"""
Sanitizer - Multi-layer PII/sensitive data sanitization

Implements defense-in-depth sanitization for client data:
- Layer 1: Regex patterns (emails, IPs, phones, SSNs, etc.)
- Layer 2: NER (Named Entity Recognition) for contextual PII
  - With allowlist filtering: known-safe terms skip NER redaction
- Layer 3: Manual review workflow

Per-path tiers (strict/standard/intel) scope which non-secret categories redact
and whether NER runs, via `default_tier`/`path_tiers`. Real secrets and
CLIENT_PATTERNS always redact regardless of tier. Most-specific path match wins.
See the TIERS constant.

Security Purpose (100% Local Architecture):
- Data at rest protection (disk theft, malware, backups)
- Compliance with client NDAs (PII removal requirements)
- Defense against accidental exposure (sharing, screenshots)
- Future-proofing (safe to migrate if cloud components added)

Allowlist (v1.1.0):
- External JSON config at ~/.0k-rag/config/ner-allowlist.json
- Prevents NER from redacting public security terms (OWASP, MITRE, L0-L19, etc.)
- 60s TTL cache for performance
- Client names and real PII are NOT allowlisted

From security analysis: output/research/0k-rag-security-analysis.md
"""

import re
import json
import time
import spacy
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result of sanitization process"""
    sanitized_text: str
    detected_patterns: List[str]
    redaction_count: int
    requires_review: bool


class Sanitizer:
    """Multi-layer PII and sensitive data sanitizer"""

    # Regex patterns for automatic detection
    SANITIZATION_PATTERNS = {
        # Contact information
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "url": r'https?://[^\s]+',

        # Network identifiers
        "ipv4": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "ipv6": r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b',
        "mac_address": r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
        "domain": r'\b[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}\b',

        # Personal identifiers
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',

        # Cloud/Server IDs
        "aws_key": r'AKIA[0-9A-Z]{16}',
        "azure_key": r'[a-zA-Z0-9+/]{88}==',

        # API keys (common patterns)
        "api_key": r'["\']?api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{32,})["\']?',
    }

    # Client-specific patterns (can be customized)
    CLIENT_PATTERNS = {
        "hospital_bella_vista": r'(?i)hospital\s+bella\s+vista',
        "uaa": r'(?i)university\s+of\s+alaska\s+anchorage',
    }

    # --- Sanitization tiers -------------------------------------------------
    # A tier declares which regex categories to redact and whether NER runs.
    # Real secrets and CLIENT_PATTERNS are NOT tier-configurable — they redact
    # on every tier (enforced in sanitize_regex, not here), so no configuration
    # can expose them.
    #
    #   strict   — redact everything + NER. The historical behavior and the
    #              default when no tier is configured (fail-closed).
    #   standard — redact PII (email/phone) + secrets + client; KEEP IOCs
    #              (domain/ip/url/mac); skip NER. IOCs are intelligence, not PII.
    #   intel    — redact secrets + client only; KEEP IOCs and CONTACT PII
    #              (email/phone). SSN/credit-card are classed as secrets and
    #              still redact. For curated threat-intel/research where
    #              sender/recipient and network indicators are the value.
    _PII_CATEGORIES = frozenset({"email", "phone"})
    _IOC_CATEGORIES = frozenset({"ipv4", "ipv6", "mac_address", "domain", "url"})
    # Secret categories always redact regardless of tier; listed for clarity.
    _SECRET_CATEGORIES = frozenset({"ssn", "credit_card", "aws_key", "azure_key", "api_key"})

    # Per tier: the set of NON-secret regex categories to redact, and run_ner.
    TIERS: Dict[str, Dict] = {
        "strict":   {"redact": _PII_CATEGORIES | _IOC_CATEGORIES, "run_ner": True},
        "standard": {"redact": _PII_CATEGORIES,                   "run_ner": False},
        "intel":    {"redact": frozenset(),                       "run_ner": False},
    }
    _DEFAULT_TIER = "strict"

    # Allowlist config path and cache
    _ALLOWLIST_DEFAULT_PATH = Path.home() / ".0k-rag" / "config" / "ner-allowlist.json"
    _ALLOWLIST_CACHE_TTL = 60  # seconds

    def __init__(
        self,
        enable_ner: bool = True,
        allowlist_path: Optional[str] = None,
        skip_ner_paths: Optional[List[str]] = None,
        default_tier: Optional[str] = None,
        path_tiers: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Initialize sanitizer

        Args:
            enable_ner: Enable Named Entity Recognition (spaCy)
            allowlist_path: Path to NER allowlist JSON (default: ~/.0k-rag/config/ner-allowlist.json)
            skip_ner_paths: LEGACY. Path substrings where the NER layer is
                skipped. Still honored, and OR'd with any tier that also skips
                NER. Prefer `path_tiers` for new config.
            default_tier: Sanitization tier applied when no path_tier matches.
                One of TIERS ("strict"/"standard"/"intel"). None or an unknown
                value falls back to "strict" (fail-closed) — an unreadable tier
                config must never loosen redaction.
            path_tiers: List of {"path": <substring>, "tier": <name>} rules.
                The first rule whose `path` is a substring of the file path
                wins; otherwise `default_tier` applies. Lets curated intel paths
                keep IOCs while client-work paths force full strict redaction.

        Secrets (ssn/credit_card/aws_key/azure_key/api_key) and CLIENT_PATTERNS
        always redact on every tier; they are not tier-configurable.
        """
        self.enable_ner = enable_ner
        self.nlp = None
        self._allowlist_path = Path(allowlist_path) if allowlist_path else self._ALLOWLIST_DEFAULT_PATH
        self._allowlist_cache: Set[str] = set()
        self._allowlist_cache_lower: Set[str] = set()
        self._allowlist_loaded_at: float = 0
        self._skip_ner_paths: List[str] = skip_ner_paths or []
        # Type-check before membership tests: a YAML config can hand us a
        # list/dict, and `x in self.TIERS` would raise on an unhashable value —
        # defeating the fail-closed guarantee. Validate the type first.
        valid_default = isinstance(default_tier, str) and default_tier in self.TIERS
        self._default_tier: str = default_tier if valid_default else self._DEFAULT_TIER
        if default_tier is not None and not valid_default:
            logger.warning(
                f"Unknown sanitizer default_tier {default_tier!r}; falling back to "
                f"'{self._DEFAULT_TIER}' (fail-closed)."
            )
        # Normalize path_tiers; drop malformed rules loudly rather than silently.
        # A non-list value (e.g. YAML `sanitize_path_tiers: true`) would raise on
        # the iteration below and crash indexing — reject it and fall closed.
        if path_tiers is not None and not isinstance(path_tiers, list):
            logger.warning(f"Ignoring non-list sanitizer path_tiers value: {path_tiers!r}")
            path_tiers = []
        self._path_tiers: List[Tuple[str, str]] = []
        for rule in path_tiers or []:
            if not isinstance(rule, dict):
                logger.warning(f"Ignoring malformed sanitizer path_tier rule: {rule!r}")
                continue
            path = rule.get("path")
            tier = rule.get("tier")
            if not isinstance(path, str) or not path or not isinstance(tier, str) or tier not in self.TIERS:
                logger.warning(f"Ignoring malformed sanitizer path_tier rule: {rule!r}")
                continue
            self._path_tiers.append((path, tier))

        if enable_ner:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"NER disabled: Could not load spaCy model: {e}")
                self.enable_ner = False

        # Load allowlist on init
        self._load_allowlist()

    @staticmethod
    def tier_kwargs_from_config(indexing_config: Dict) -> Dict:
        """Extract tier constructor kwargs from an `.0k-rag.yml` [indexing] dict.

        Reads `sanitize_default_tier` (str) and `sanitize_path_tiers`
        (list of {path, tier}). Returns only the keys that are present, so a
        config with neither yields `{}` and the Sanitizer keeps its strict
        default — a deployment that has not opted in is never loosened. Bad
        values are not validated here; the Sanitizer constructor fails them
        closed (unknown tier -> strict, malformed rule -> dropped).
        """
        kwargs: Dict = {}
        default_tier = indexing_config.get("sanitize_default_tier")
        if default_tier is not None:
            kwargs["default_tier"] = default_tier
        path_tiers = indexing_config.get("sanitize_path_tiers")
        if path_tiers:
            kwargs["path_tiers"] = path_tiers
        return kwargs

    def _load_allowlist(self) -> None:
        """Load NER allowlist from JSON config with TTL cache"""
        now = time.time()
        if now - self._allowlist_loaded_at < self._ALLOWLIST_CACHE_TTL and self._allowlist_cache:
            return  # Cache still valid

        try:
            if not self._allowlist_path.exists():
                logger.debug(f"No NER allowlist found at {self._allowlist_path}")
                return

            with open(self._allowlist_path) as f:
                config = json.load(f)

            allowlist = config.get("allowlist", {})
            terms: Set[str] = set()

            for category_name, category_data in allowlist.items():
                category_terms = category_data.get("terms", [])
                terms.update(category_terms)

            self._allowlist_cache = terms
            self._allowlist_cache_lower = {t.lower() for t in terms}
            self._allowlist_loaded_at = now

            logger.info(f"Loaded NER allowlist: {len(terms)} terms from {len(allowlist)} categories")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load NER allowlist: {e}")
            # Keep existing cache on error (fail-open for allowlist is safe)

    def _is_allowlisted(self, entity_text: str) -> bool:
        """Check if an NER entity matches the allowlist

        Uses exact match for short terms (likely acronyms) and
        case-insensitive match for longer terms (multi-word phrases).

        Args:
            entity_text: The entity text detected by NER

        Returns:
            True if the entity should be preserved (not redacted)
        """
        # Refresh cache if stale
        self._load_allowlist()

        # Exact match (handles acronyms like OWASP, L5, TCB)
        if entity_text in self._allowlist_cache:
            return True

        # Case-insensitive match (handles "Google" vs "google", multi-word terms)
        if entity_text.lower() in self._allowlist_cache_lower:
            return True

        return False

    def is_client_data(self, file_path: str, content: str = None) -> bool:
        """
        Detect if file/content contains client data

        Args:
            file_path: Path to file
            content: Optional file content

        Returns:
            True if client data detected
        """
        client_indicators = [
            '/client-work/',
            '/Cooperton/',
            'TTX',
            'Hospital',
            'UAA',
            'client',
            'engagement',
        ]

        # Check file path
        for indicator in client_indicators:
            if indicator.lower() in file_path.lower():
                return True

        # Check content if provided
        if content:
            for indicator in client_indicators:
                if indicator.lower() in content.lower():
                    return True

        return False

    def sanitize_regex(
        self, text: str, redact_categories: Optional[Set[str]] = None
    ) -> Tuple[str, List[str]]:
        """
        Layer 1: Regex-based sanitization.

        Args:
            text: Text to sanitize
            redact_categories: Which SANITIZATION_PATTERNS categories to redact
                for non-secret classes (PII/IOC). None = redact all (strict).
                Secret categories and CLIENT_PATTERNS ALWAYS redact regardless
                of this set — a permissive tier can never expose them.

        Returns:
            (sanitized_text, detected_patterns)
        """
        sanitized = text
        detected = []

        for pattern_name, pattern in self.SANITIZATION_PATTERNS.items():
            # Secrets always redact; other categories only if the tier says so.
            if pattern_name not in self._SECRET_CATEGORIES:
                if redact_categories is not None and pattern_name not in redact_categories:
                    continue
            matches = re.findall(pattern, text)
            if matches:
                detected.append(f"{pattern_name}: {len(matches)} occurrences")
                replacement = f"[REDACTED_{pattern_name.upper()}]"
                sanitized = re.sub(pattern, replacement, sanitized)

        # Client-specific patterns ALWAYS redact, on every tier.
        for pattern_name, pattern in self.CLIENT_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                detected.append(f"{pattern_name}: {len(matches)} occurrences")
                replacement = "[REDACTED_CLIENT]"
                sanitized = re.sub(pattern, replacement, sanitized)

        return sanitized, detected

    def sanitize_ner(self, text: str) -> Tuple[str, List[str]]:
        """
        Layer 2: Named Entity Recognition sanitization

        Applies NER to detect PERSON, ORG, GPE entities, then filters
        out allowlisted terms (public security frameworks, layer IDs, etc.)
        before redacting.

        Args:
            text: Text to sanitize

        Returns:
            (sanitized_text, detected_entities)
        """
        if not self.enable_ner or not self.nlp:
            return text, []

        try:
            doc = self.nlp(text)
            sanitized = text
            detected = []
            skipped = []

            # Redact PERSON, ORG, GPE (locations) — unless allowlisted
            entities_to_redact = {}

            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    if self._is_allowlisted(ent.text):
                        skipped.append(f"{ent.label_}: {ent.text}")
                        continue
                    entities_to_redact[ent.text] = f"[REDACTED_{ent.label_}]"
                    detected.append(f"{ent.label_}: {ent.text}")

            if skipped:
                logger.debug(f"NER allowlist preserved {len(skipped)} entities: {skipped}")

            # Replace entities (longest first to avoid partial replacements)
            for entity, replacement in sorted(entities_to_redact.items(), key=lambda x: len(x[0]), reverse=True):
                sanitized = sanitized.replace(entity, replacement)

            return sanitized, detected

        except Exception as e:
            logger.error(f"NER sanitization failed: {e}")
            return text, []

    def _path_skips_ner(self, file_path: str) -> bool:
        """Check if file_path matches any configured NER-skip path substring."""
        if not file_path or not self._skip_ner_paths:
            return False
        return any(skip in file_path for skip in self._skip_ner_paths)

    # Redaction strength per tier (higher = more redacted). Used to break ties
    # between equal-length path matches toward the safer tier.
    _TIER_STRICTNESS = {"intel": 0, "standard": 1, "strict": 2}

    def _resolve_tier(self, file_path: str) -> str:
        """Most-specific matching path_tier rule wins (longest matching path),
        NOT first-in-config-order. On equal-length ties, the stricter tier wins.
        Falls back to the default tier when nothing matches.

        Longest-match — not first-match, and not strictest-overall — is the
        correct routing semantics here (cf. .gitignore / longest-prefix routing):
          - a narrow `output/client-work/` -> strict rule beats a broad
            `output/` -> intel rule regardless of config order, so client work
            cannot be routed to a permissive tier by mis-ordering; AND
          - a legitimate `output/threat-intel/` -> intel carve-out still overrides
            a broad strict default, which strictest-overall would wrongly forbid.
        Fail-closed values were validated at init."""
        best_tier = self._default_tier
        best_len = -1
        for path, tier in self._path_tiers:
            if path not in file_path:
                continue
            plen = len(path)
            if plen > best_len or (
                plen == best_len
                and self._TIER_STRICTNESS[tier] > self._TIER_STRICTNESS[best_tier]
            ):
                best_tier = tier
                best_len = plen
        return best_tier

    def sanitize(self, text: str, file_path: str = "") -> SanitizationResult:
        """
        Complete multi-layer sanitization

        Args:
            text: Text to sanitize
            file_path: Source file path (for context + NER-skip routing)

        Returns:
            SanitizationResult object
        """
        # Resolve the tier for this path. Secrets + CLIENT_PATTERNS redact on
        # every tier; the tier only governs the non-secret PII/IOC categories
        # and whether NER runs.
        tier = self._resolve_tier(file_path)
        tier_cfg = self.TIERS[tier]
        redact_categories = set(tier_cfg["redact"]) | self._SECRET_CATEGORIES

        # Layer 1: Regex (secrets + client always; PII/IOC per tier).
        sanitized, regex_detected = self.sanitize_regex(text, redact_categories)

        # Layer 2: NER. Skipped when the tier says so, OR for a legacy
        # skip_ner_paths match. Regex has already protected real PII/secrets.
        ner_detected: List[str] = []
        if not tier_cfg["run_ner"] or self._path_skips_ner(file_path):
            logger.debug(f"NER skipped (tier={tier}) for path: {file_path}")
        else:
            sanitized, ner_detected = self.sanitize_ner(sanitized)

        # Combine detections
        all_detected = regex_detected + ner_detected
        redaction_count = len(all_detected)

        # Determine if manual review required
        requires_review = self._requires_manual_review(file_path, text, all_detected)

        return SanitizationResult(
            sanitized_text=sanitized,
            detected_patterns=all_detected,
            redaction_count=redaction_count,
            requires_review=requires_review
        )

    def _requires_manual_review(self, file_path: str, original_text: str, detected: List[str]) -> bool:
        """
        Determine if manual review is required

        Args:
            file_path: Source file path
            original_text: Original text
            detected: List of detected patterns

        Returns:
            True if manual review needed
        """
        # Always review client data
        if self.is_client_data(file_path, original_text):
            return True

        # Review if high number of redactions
        if len(detected) > 10:
            return True

        # Review if specific high-risk patterns found
        high_risk_patterns = ["hospital", "university", "client", "engagement", "ssn", "credit_card"]
        for pattern in high_risk_patterns:
            if any(pattern in d.lower() for d in detected):
                return True

        return False

    def validate_sanitization(
        self, sanitized_text: str, tier: str = "strict"
    ) -> Tuple[bool, List[str]]:
        """
        Spot-check that no tier-redacted PII/secret pattern remains.

        This is a REGEX SPOT-CHECK against a declared tier, NOT an exhaustive
        proof of clean output and NOT an independent guarantee — it re-derives
        the expected redaction from the caller-supplied `tier`, so it cannot
        detect a tier misconfiguration, and NER-class entities (PERSON/ORG/GPE)
        are not checked at all (regex cannot see them).

        Tier-aware: `standard` keeps IPv4, `intel` keeps email/phone/IPv4, so
        those are only checked on tiers that redact them. ALL five secret
        categories (ssn/credit_card/aws_key/azure_key/api_key) are checked on
        every tier, matching the always-redact invariant. Defaults to `strict`.

        Args:
            sanitized_text: Sanitized text to validate
            tier: The tier the text was sanitized with. Unknown → strict.

        Returns:
            (is_valid, list_of_failures)
        """
        redact = set(self.TIERS.get(tier, self.TIERS[self._DEFAULT_TIER])["redact"])

        # (pattern, description, category). Checked when the tier redacts the
        # category, OR always for secrets. Secret patterns mirror
        # SANITIZATION_PATTERNS so a leak of any secret class is caught on any
        # tier — the always-redact invariant made verifiable.
        pii_tests = [
            (r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', "Email addresses", "email"),
            (r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', "Phone", "phone"),
            (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', "IP address", "ipv4"),
            # secrets — always checked (category ∈ _SECRET_CATEGORIES)
            (self.SANITIZATION_PATTERNS["ssn"], "SSN", "ssn"),
            (self.SANITIZATION_PATTERNS["credit_card"], "Credit card", "credit_card"),
            (self.SANITIZATION_PATTERNS["aws_key"], "AWS key", "aws_key"),
            (self.SANITIZATION_PATTERNS["azure_key"], "Azure key", "azure_key"),
            (self.SANITIZATION_PATTERNS["api_key"], "API key", "api_key"),
        ]

        failures = []
        for pattern, description, category in pii_tests:
            if category not in self._SECRET_CATEGORIES and category not in redact:
                continue
            if re.search(pattern, sanitized_text):
                failures.append(description)

        return (len(failures) == 0, failures)

    def get_stats(self) -> Dict:
        """Get sanitizer statistics"""
        return {
            'regex_patterns': len(self.SANITIZATION_PATTERNS),
            'client_patterns': len(self.CLIENT_PATTERNS),
            'ner_enabled': self.enable_ner,
            'allowlist_terms': len(self._allowlist_cache),
            'allowlist_path': str(self._allowlist_path)
        }
