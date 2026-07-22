"""
Effect-level tests for the Sanitizer path-tier feature.

The sanitizer historically redacted every regex category and ran NER on all
input. That destroys IOCs (domains, IPs, URLs) and public person/org names in
curated threat-intel content, where those are the intelligence, not PII.

This adds opt-in path tiers. The tests assert the REAL Sanitizer's redaction
outcome per tier — not that a flag is set — because a control you cannot prove
fires must be assumed broken.

Non-negotiable invariants pinned here:
  - Real secrets (ssn, credit_card, aws_key, azure_key, api_key) redact on
    EVERY tier, including the most permissive.
  - Client patterns redact on every tier.
  - The default with no configuration is byte-identical to the old behavior
    (strict), so upgrading without touching config never loosens posture.

Secret-shaped and client tokens are built from fragments / are synthetic, so no
real secret or client name is committed to this public repo — which is the very
thing the sanitizer, and the Governor guarding this repo, exist to prevent.
"""

import pytest

from rag.indexing.sanitizer import Sanitizer

# Built from fragments so no literal secret sits in the file. Each still matches
# its regex. AWS = AKIA + 16 upper/digit; SSN ddd-dd-dddd; card 4-4-4-4;
# azure = 88 base64 chars + "=="; api_key = api_key="<32+ chars>".
_AWS = "AKIA" + "IOSFODNN7" + "EXAMPLE"          # AWS's own documented example key
_SSN = "123" + "-45" + "-6789"
_CARD = "4111" + " 1111" + " 1111" + " 1111"     # fragment-built so no PAN literal on disk
_AZURE = ("A1b2" * 22) + "=="                     # 88 chars + == matches azure_key
_APIKEY = 'api_key="' + ("k9X2" * 8) + '"'        # 32-char value matches api_key
_IPV6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

# Synthetic client name — NOT a real client. Injected as a test client pattern
# so we verify the mechanism ("client patterns always redact") without embedding
# a real client name in a public repo.
_CLIENT_NAME = "Northwind Test Clinic"
_CLIENT_PATTERN = {"synthetic_test_client": r"(?i)northwind\s+test\s+clinic"}

SAMPLE = (
    f"Contact soc@example.org or call 787-555-0100. "
    f"C2 beacon to evil-c2.net and 185.220.101.44 over ipv6 {_IPV6}, "
    f"mac 00:1b:44:11:3a:b7, staging at https://cdn-abuse.example/payload. "
    f"Leaked {_AWS} and SSN {_SSN}, card {_CARD}, azure {_AZURE}, {_APIKEY}. "
    f"Report drafted at {_CLIENT_NAME} by Caleb Sima."
)

IOC_TOKENS = ["evil-c2.net", "185.220.101.44", _IPV6, "00:1b:44:11:3a:b7", "cdn-abuse.example"]
PII_TOKENS = ["soc@example.org", "787-555-0100"]
SECRET_TOKENS = [_AWS, _SSN, _CARD, _AZURE, _APIKEY]


def _make(tier=None, path_tiers=None):
    # NER disabled except where NER routing is the thing under test — it needs
    # the spaCy model and is orthogonal to regex-category routing. A synthetic
    # client pattern is injected so client-redaction is verified without a real
    # client name on disk.
    s = Sanitizer(enable_ner=False, default_tier=tier, path_tiers=path_tiers)
    s.CLIENT_PATTERNS = dict(_CLIENT_PATTERN)
    return s


def kept(text, token):
    return token in text


def redacted(text, token):
    return token not in text


# ---------------------------------------------------------------------------
# Backward compatibility — the load-bearing safety property
# ---------------------------------------------------------------------------

def test_no_config_default_redacts_everything_like_before():
    """Sanitizer() with no tier config must behave like the old strict default:
    every category redacted. Upgrading without config must not loosen posture."""
    s = _make()
    out = s.sanitize(SAMPLE, file_path="some/arbitrary/note.md").sanitized_text
    for tok in IOC_TOKENS + PII_TOKENS + SECRET_TOKENS + [_CLIENT_NAME]:
        assert redacted(out, tok), f"strict default must redact {tok!r}"


def test_explicit_strict_matches_no_config():
    text = "beacon to evil-c2.net from soc@example.org"
    assert (
        _make().sanitize(text, "x.md").sanitized_text
        == _make(tier="strict").sanitize(text, "x.md").sanitized_text
    )


# ---------------------------------------------------------------------------
# standard tier — the configured global default
# ---------------------------------------------------------------------------

def test_standard_keeps_iocs_redacts_pii_and_secrets():
    s = _make(tier="standard")
    out = s.sanitize(SAMPLE, "note.md").sanitized_text
    for tok in IOC_TOKENS:
        assert kept(out, tok), f"standard must keep IOC {tok!r}"
    for tok in PII_TOKENS:
        assert redacted(out, tok), f"standard must redact PII {tok!r}"
    for tok in SECRET_TOKENS:
        assert redacted(out, tok), f"standard must ALWAYS redact secret {tok!r}"
    assert redacted(out, _CLIENT_NAME), "standard must ALWAYS redact client patterns"


# ---------------------------------------------------------------------------
# intel tier — output/threat-intel/, output/research/
# ---------------------------------------------------------------------------

def test_intel_keeps_iocs_and_pii_but_still_redacts_secrets_and_clients():
    s = _make(tier="standard", path_tiers=[{"path": "output/threat-intel/", "tier": "intel"}])
    out = s.sanitize(SAMPLE, "output/threat-intel/hf-breach.md").sanitized_text
    for tok in IOC_TOKENS + PII_TOKENS:
        assert kept(out, tok), f"intel must keep {tok!r}"
    for tok in SECRET_TOKENS:
        assert redacted(out, tok), f"intel must STILL redact secret {tok!r}"
    assert redacted(out, _CLIENT_NAME), "intel must STILL redact client patterns"


def test_intel_tier_only_applies_on_matching_path():
    s = _make(tier="standard", path_tiers=[{"path": "output/threat-intel/", "tier": "intel"}])
    out = s.sanitize(SAMPLE, "output/personal/diary.md").sanitized_text
    assert redacted(out, "soc@example.org"), "non-intel path uses standard default → PII redacted"
    assert kept(out, "evil-c2.net")


# ---------------------------------------------------------------------------
# strict path override — full redaction preserved exactly
# ---------------------------------------------------------------------------

def test_client_path_forces_full_redaction_even_under_standard_default():
    s = _make(tier="standard", path_tiers=[{"path": "output/client-work/", "tier": "strict"}])
    out = s.sanitize(SAMPLE, "output/client-work/acme/report.md").sanitized_text
    for tok in IOC_TOKENS + PII_TOKENS + SECRET_TOKENS + [_CLIENT_NAME]:
        assert redacted(out, tok), f"client-work/strict must redact {tok!r}"


# ---------------------------------------------------------------------------
# Tier resolution ordering — the leak all four review agents flagged.
# Resolution must be MOST-SPECIFIC-path-wins, independent of config order, so a
# broad permissive rule listed first cannot shadow a narrow strict rule.
# ---------------------------------------------------------------------------

def test_narrow_strict_rule_wins_over_broad_intel_prefix_regardless_of_order():
    # Broad intel rule listed FIRST, narrow strict rule second. First-match-wins
    # would resolve a client-work file to intel and leak client email/phone.
    s = _make(tier="standard", path_tiers=[
        {"path": "output/",             "tier": "intel"},
        {"path": "output/client-work/", "tier": "strict"},
    ])
    out = s.sanitize(SAMPLE, "output/client-work/acme/report.md").sanitized_text
    for tok in IOC_TOKENS + PII_TOKENS + SECRET_TOKENS + [_CLIENT_NAME]:
        assert redacted(out, tok), f"client-work must resolve strict, not intel; leaked {tok!r}"


def test_specific_permissive_carveout_still_overrides_broad_strict():
    # The inverse the fix must PRESERVE: a narrow intel carve-out under a broad
    # strict default must still resolve intel (most-specific wins, not strictest).
    s = _make(tier="standard", path_tiers=[
        {"path": "output/",              "tier": "strict"},
        {"path": "output/threat-intel/", "tier": "intel"},
    ])
    out = s.sanitize(SAMPLE, "output/threat-intel/hf.md").sanitized_text
    assert kept(out, "evil-c2.net"), "narrow intel carve-out must win over broad strict"
    assert kept(out, "soc@example.org")
    for tok in SECRET_TOKENS:
        assert redacted(out, tok), "secrets still redact even in the carve-out"


# ---------------------------------------------------------------------------
# NER routing per tier (needs the spaCy model; skipped if unavailable)
# ---------------------------------------------------------------------------

def _ner_available():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ner_available(), reason="spaCy model not installed")
def test_strict_runs_ner_standard_skips_it():
    person = "Caleb Sima noted the guardrail issue."
    strict = Sanitizer(enable_ner=True, default_tier="strict")
    standard = Sanitizer(enable_ner=True, default_tier="standard")
    assert "Caleb Sima" not in strict.sanitize(person, "x.md").sanitized_text
    assert "Caleb Sima" in standard.sanitize(person, "x.md").sanitized_text


# ---------------------------------------------------------------------------
# Config robustness — a bad tier name must fail CLOSED (strict), never open.
# ---------------------------------------------------------------------------

def test_unknown_tier_falls_back_to_strict_not_open():
    s = _make(tier="nonsense-typo")
    out = s.sanitize(SAMPLE, "note.md").sanitized_text
    for tok in IOC_TOKENS + PII_TOKENS + SECRET_TOKENS:
        assert redacted(out, tok), f"unknown tier must fail CLOSED (strict), leaked {tok!r}"


@pytest.mark.parametrize("bad", [["standard"], {"tier": "standard"}, 123, {"path": "x"}])
def test_non_string_config_fails_closed_without_crashing(bad):
    """A YAML config can hand us a list/dict; membership tests must not raise,
    and the result must be strict, not open."""
    s = _make(tier=bad)
    out = s.sanitize(SAMPLE, "note.md").sanitized_text
    assert redacted(out, "evil-c2.net"), "non-string default_tier must fail closed to strict"
    # Malformed path_tier rules must be dropped, not crash construction.
    s2 = _make(tier="standard", path_tiers=[bad, "notadict", {"path": 1, "tier": "intel"}])
    assert s2.sanitize(SAMPLE, "output/threat-intel/x.md").sanitized_text is not None


# ---------------------------------------------------------------------------
# validate_sanitization must agree with the tier it validates
# ---------------------------------------------------------------------------

def test_validate_is_tier_aware():
    s = _make()
    # intel output keeps email + IPv4 — validation for the intel tier must PASS.
    intel_out = "beacon 185.220.101.44 from soc@example.org"
    ok_intel, fails_intel = s.validate_sanitization(intel_out, tier="intel")
    assert ok_intel, f"intel-tier text wrongly flagged: {fails_intel}"

    # The same text under strict expectation must FAIL (email + IP present).
    ok_strict, fails_strict = s.validate_sanitization(intel_out, tier="strict")
    assert not ok_strict and "Email addresses" in fails_strict and "IP address" in fails_strict

    # SSN (a secret) is caught on EVERY tier, including the most permissive.
    ssn_leak = f"ref {_SSN} kept"
    ok, fails = s.validate_sanitization(ssn_leak, tier="intel")
    assert not ok and "SSN" in fails, "secrets must be validated on every tier"


def test_validate_catches_all_five_secret_classes_on_intel():
    """The always-redact invariant must be VERIFIABLE: validate flags every
    secret class even on the most permissive tier, not just SSN."""
    s = _make()
    for leak, label in [
        (_AWS, "AWS key"), (_SSN, "SSN"), (_CARD, "Credit card"),
        (_AZURE, "Azure key"), (_APIKEY, "API key"),
    ]:
        ok, fails = s.validate_sanitization(f"leaked {leak} here", tier="intel")
        assert not ok, f"validate missed secret {label} on intel tier"


def test_validate_default_tier_is_strict_backward_compat():
    """Calling validate without a tier keeps the old strict contract."""
    s = _make()
    text = "ip 10.0.0.1 email a@b.co"
    assert s.validate_sanitization(text) == s.validate_sanitization(text, tier="strict")


def test_malformed_path_tier_with_bogus_tier_name_is_dropped():
    """A well-formed dict with an unknown tier string is dropped (not applied),
    so the path falls through to the default tier — fails closed, not open."""
    s = _make(tier="strict", path_tiers=[{"path": "output/threat-intel/", "tier": "strickt"}])
    out = s.sanitize(SAMPLE, "output/threat-intel/x.md").sanitized_text
    # bogus intel-ish rule dropped -> strict default applies -> IOCs redacted
    assert redacted(out, "evil-c2.net"), "dropped bogus rule must fall to strict, not open"
