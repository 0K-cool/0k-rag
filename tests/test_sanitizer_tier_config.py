"""
Tests for wiring tier config from .0k-rag.yml into the Sanitizer.

The callers (rag/cli/index.py, mcp_server/ok_rag_server.py) read tier config via
Sanitizer.tier_kwargs_from_config() and splat it into the constructor. This pins
that seam: absent config keeps the strict default (no silent loosening on
upgrade), present config drives the tiers, and a config-shaped path_tiers list
(exactly what YAML produces) routes redaction correctly end to end.
"""

from rag.indexing.sanitizer import Sanitizer

_AWS = "AKIA" + "IOSFODNN7" + "EXAMPLE"
SAMPLE = f"soc@example.org 185.220.101.44 evil.net leaked {_AWS}"


def test_absent_tier_config_yields_no_kwargs_and_strict_default():
    # A config with no tier keys must not opt in — Sanitizer stays strict.
    cfg = {"sanitize_ner_skip_paths": ["output/research/"]}
    kwargs = Sanitizer.tier_kwargs_from_config(cfg)
    assert kwargs == {}
    s = Sanitizer(enable_ner=False, **kwargs)
    out = s.sanitize(SAMPLE, "anything.md").sanitized_text
    for tok in ["soc@example.org", "185.220.101.44", "evil.net", _AWS]:
        assert tok not in out, f"strict default must redact {tok!r}"


def test_present_tier_config_is_extracted():
    cfg = {
        "sanitize_default_tier": "standard",
        "sanitize_path_tiers": [
            {"path": "output/threat-intel/", "tier": "intel"},
            {"path": "output/client-work/", "tier": "strict"},
        ],
    }
    kwargs = Sanitizer.tier_kwargs_from_config(cfg)
    assert kwargs["default_tier"] == "standard"
    assert kwargs["path_tiers"] == cfg["sanitize_path_tiers"]


def test_config_shaped_path_tiers_route_end_to_end():
    # The YAML-produced list of dicts must drive the Sanitizer directly.
    cfg = {
        "sanitize_default_tier": "standard",
        "sanitize_path_tiers": [
            {"path": "output/threat-intel/", "tier": "intel"},
            {"path": "output/client-work/", "tier": "strict"},
        ],
    }
    s = Sanitizer(enable_ner=False, **Sanitizer.tier_kwargs_from_config(cfg))

    intel = s.sanitize(SAMPLE, "output/threat-intel/hf.md").sanitized_text
    assert "soc@example.org" in intel and "185.220.101.44" in intel, "intel keeps IOCs+PII"
    assert _AWS not in intel, "secrets still redact on intel"

    client = s.sanitize(SAMPLE, "output/client-work/acme/r.md").sanitized_text
    for tok in ["soc@example.org", "185.220.101.44", "evil.net", _AWS]:
        assert tok not in client, f"client-work forces strict; {tok!r} must redact"


def test_empty_path_tiers_list_is_omitted():
    # A falsy list must not be passed (kept minimal); default still applies.
    kwargs = Sanitizer.tier_kwargs_from_config({"sanitize_path_tiers": []})
    assert "path_tiers" not in kwargs


def test_bad_config_values_fail_closed_via_constructor():
    # The helper does not validate; the constructor must fail these closed.
    cfg = {"sanitize_default_tier": "typo", "sanitize_path_tiers": [{"path": "output/", "tier": "bogus"}]}
    s = Sanitizer(enable_ner=False, **Sanitizer.tier_kwargs_from_config(cfg))
    out = s.sanitize(SAMPLE, "output/x.md").sanitized_text
    assert "evil.net" not in out, "bad tier config must fall back to strict, not open"


def test_scalar_path_tiers_does_not_crash_and_falls_closed():
    # YAML `sanitize_path_tiers: true` forwards a truthy scalar; the constructor
    # must reject it (not crash on `for rule in True`) and stay strict.
    cfg = {"sanitize_default_tier": "standard", "sanitize_path_tiers": True}
    s = Sanitizer(enable_ner=False, **Sanitizer.tier_kwargs_from_config(cfg))
    # standard default still applies (scalar path_tiers dropped, not fatal),
    # and construction did not raise.
    out = s.sanitize(SAMPLE, "output/threat-intel/x.md").sanitized_text
    assert _AWS not in out, "secrets still redact after scalar path_tiers dropped"
