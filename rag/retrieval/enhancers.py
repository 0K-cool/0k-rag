"""
Retrieval Enhancers — Post-fusion score boosting for improved recall.

Three enhancers that re-score candidates between RRF fusion and BGE reranking:
1. Temporal Boosting — date-aware proximity scoring
2. Preference Detection — bridges vocabulary gap for preference queries
3. Entity Boosting — quoted phrases and proper noun emphasis

These are production improvements to the retrieval pipeline, not benchmark hacks.
Inspired by MemPalace's hybrid v1-v4 progression but adapted for our architecture.

Slots into pipeline between Step 3 (fusion) and Step 4 (reranking).
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# =============================================================================
# Enhancer 1: Temporal Boosting
# =============================================================================

# Month names and abbreviations for date extraction
MONTH_PATTERNS = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
    'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12,
}

# Temporal signal patterns in queries
TEMPORAL_QUERY_PATTERNS = [
    # "in February", "in March 2024"
    re.compile(r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b', re.I),
    # "last January", "last March"
    re.compile(r'\blast\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b', re.I),
    # "first in February", "earlier in March"
    re.compile(r'\b(?:first|earlier|later|recently)\s+in\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b', re.I),
    # Explicit dates: "on March 15", "March 15, 2024"
    re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:,?\s*(\d{4}))?\b', re.I),
    # ISO dates: "2024-03-15"
    re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
    # Relative: "last week", "last month", "yesterday"
    re.compile(r'\b(last\s+(?:week|month|year)|yesterday|this\s+(?:week|month))\b', re.I),
]

# Date patterns in document content
CONTENT_DATE_PATTERNS = [
    re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:,?\s*(\d{4}))?\b', re.I),
    re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
    re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b'),
]


def extract_temporal_signals(query: str) -> List[Tuple[int, Optional[int]]]:
    """Extract month/year signals from a query. Returns list of (month, year|None)."""
    signals = []
    for pattern in TEMPORAL_QUERY_PATTERNS:
        for match in pattern.finditer(query):
            groups = match.groups()
            if groups:
                month_str = groups[0].lower()
                if month_str in MONTH_PATTERNS:
                    month = MONTH_PATTERNS[month_str]
                    year = int(groups[1]) if len(groups) > 1 and groups[1] and groups[1].isdigit() else None
                    signals.append((month, year))
    return signals


def content_has_temporal_match(content: str, signals: List[Tuple[int, Optional[int]]]) -> float:
    """Check if content mentions dates matching the temporal signals. Returns proximity score 0-1."""
    if not signals:
        return 0.0

    best_score = 0.0
    content_lower = content.lower()

    for target_month, target_year in signals:
        # Check for month name mentions
        for month_name, month_num in MONTH_PATTERNS.items():
            if month_num == target_month and month_name in content_lower:
                # Month match — strong signal
                score = 0.8
                if target_year:
                    if str(target_year) in content:
                        score = 1.0  # Month + year match
                best_score = max(best_score, score)

    return best_score


def apply_temporal_boost(
    results: List[Dict],
    query: str,
    max_boost: float = 0.4,
    score_key: str = 'rrf_score'
) -> List[Dict]:
    """
    Boost results that match temporal signals in the query.

    Args:
        results: Results with score_key field
        query: Original query string
        max_boost: Maximum score multiplier (0.4 = 40% boost, matching MemPalace v2)
        score_key: Which score field to boost ('rrf_score' or 'rerank_score')

    Returns:
        Results with adjusted scores and temporal_boost metadata
    """
    signals = extract_temporal_signals(query)
    if not signals:
        return results

    logger.debug(f"Temporal signals detected: {signals}")

    for result in results:
        content = result.get('original_chunk', '') + ' ' + result.get('generated_context', '')
        proximity = content_has_temporal_match(content, signals)

        if proximity > 0:
            boost = max_boost * proximity
            original_score = result.get(score_key, 0)
            result[score_key] = original_score * (1.0 + boost)
            result['temporal_boost'] = boost
            result['temporal_proximity'] = proximity
        else:
            result['temporal_boost'] = 0.0

    results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
    return results


# =============================================================================
# Enhancer 2: Preference Pattern Detection
# =============================================================================

# Patterns that indicate preference-related content in documents
PREFERENCE_CONTENT_PATTERNS = [
    re.compile(r'\b(?:i|my)\s+(?:really\s+)?(?:prefer|like|love|enjoy|favor|choose)\b', re.I),
    re.compile(r'\b(?:i|my)\s+(?:usually|always|typically|normally|often|tend\s+to)\b', re.I),
    re.compile(r"\b(?:i|my)\s+(?:don't|do\s+not|never|hate|dislike|can't\s+stand)\b", re.I),
    re.compile(r'\b(?:my\s+favorite|my\s+go-to|my\s+preferred)\b', re.I),
    re.compile(r'\b(?:i\'m\s+a\s+fan\s+of|i\'m\s+into|i\'m\s+passionate\s+about)\b', re.I),
    re.compile(r'\b(?:i\s+(?:would|\'d)\s+rather)\b', re.I),
    re.compile(r'\b(?:best\s+(?:way|thing|part)\s+(?:is|about))\b', re.I),
    re.compile(r'\b(?:i\s+(?:switched|moved|changed)\s+(?:to|from))\b', re.I),
    re.compile(r'\b(?:i\s+(?:recommend|suggest|use)\s+\w+\s+(?:for|because|since))\b', re.I),
    re.compile(r'\b(?:nothing\s+beats|there\'s\s+nothing\s+like)\b', re.I),
    re.compile(r'\b(?:i\s+find\s+(?:it|that|this)\s+(?:better|worse|easier|harder))\b', re.I),
    re.compile(r'\b(?:i\'ve\s+(?:always|been)\s+(?:a|into))\b', re.I),
    re.compile(r'\b(?:i\s+(?:grew\s+up|was\s+raised)\s+(?:with|on|eating|doing))\b', re.I),
    re.compile(r'\b(?:my\s+(?:hobby|hobbies|interest|interests|passion))\b', re.I),
    re.compile(r'\b(?:i\s+(?:collect|practice|study|play)\s+\w+\s+(?:regularly|daily|weekly))\b', re.I),
    re.compile(r'\b(?:i\s+(?:can\'t|cannot)\s+(?:live|go|do)\s+without)\b', re.I),
]

# Patterns that indicate a preference-seeking query
PREFERENCE_QUERY_PATTERNS = [
    re.compile(r'\b(?:what\s+(?:do|did)\s+i\s+(?:prefer|like|enjoy|love|favor))\b', re.I),
    re.compile(r'\b(?:what(?:\'s|\s+is)\s+my\s+(?:favorite|preferred|go-to))\b', re.I),
    re.compile(r'\b(?:do\s+i\s+(?:like|prefer|enjoy|love))\b', re.I),
    re.compile(r'\b(?:what\s+(?:kind|type|sort)\s+of\s+\w+\s+do\s+i)\b', re.I),
    re.compile(r'\b(?:how\s+do\s+i\s+(?:feel|think)\s+about)\b', re.I),
    re.compile(r'\b(?:what\s+are\s+my\s+(?:hobbies|interests|preferences))\b', re.I),
    re.compile(r'\b(?:what\s+(?:do|did)\s+i\s+(?:usually|typically|normally|often))\b', re.I),
]


def is_preference_query(query: str) -> bool:
    """Detect if the query is seeking preference information."""
    for pattern in PREFERENCE_QUERY_PATTERNS:
        if pattern.search(query):
            return True
    return False


def content_preference_score(content: str) -> float:
    """Score how strongly content expresses preferences. Returns 0-1."""
    matches = 0
    for pattern in PREFERENCE_CONTENT_PATTERNS:
        if pattern.search(content):
            matches += 1
    # Normalize: 3+ pattern matches = max score
    return min(matches / 3.0, 1.0)


def apply_preference_boost(
    results: List[Dict],
    query: str,
    max_boost: float = 0.5,
    score_key: str = 'rrf_score'
) -> List[Dict]:
    """
    Boost results containing preference language when query seeks preferences.

    Args:
        results: Results with score_key field
        query: Original query string
        max_boost: Maximum score multiplier (0.5 = 50% boost)
        score_key: Which score field to boost

    Returns:
        Results with adjusted scores and preference_boost metadata
    """
    if not is_preference_query(query):
        for r in results:
            r['preference_boost'] = 0.0
        return results

    logger.debug("Preference query detected — applying preference boost")

    for result in results:
        content = result.get('original_chunk', '')
        pref_score = content_preference_score(content)

        if pref_score > 0:
            boost = max_boost * pref_score
            original_score = result.get(score_key, 0)
            result[score_key] = original_score * (1.0 + boost)
            result['preference_boost'] = boost
            result['preference_score'] = pref_score
        else:
            result['preference_boost'] = 0.0

    results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
    return results


# =============================================================================
# Enhancer 3: Entity Boosting (Quoted Phrases + Proper Nouns)
# =============================================================================

QUOTED_PHRASE_PATTERN = re.compile(r"""['"]([^'"]{3,50})['"]""")
PROPER_NOUN_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')

# Common words that look like proper nouns but aren't
PROPER_NOUN_EXCLUSIONS = {
    'What', 'When', 'Where', 'Which', 'Who', 'How', 'Why', 'The', 'This',
    'That', 'There', 'Then', 'Than', 'Their', 'They', 'These', 'Those',
    'Have', 'Has', 'Had', 'Was', 'Were', 'Been', 'Being', 'Would', 'Could',
    'Should', 'Will', 'Can', 'May', 'Might', 'Must', 'Shall', 'Did', 'Does',
    'After', 'Before', 'During', 'Until', 'Since', 'About', 'Between',
    'January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December',
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    'First', 'Last', 'Next', 'Also', 'Just', 'Still', 'Even', 'However',
}


def extract_entities(query: str) -> Tuple[List[str], List[str]]:
    """Extract quoted phrases and proper nouns from query."""
    # Quoted phrases
    quoted = [m.group(1) for m in QUOTED_PHRASE_PATTERN.finditer(query)]

    # Proper nouns (excluding common words and months)
    proper_nouns = []
    for m in PROPER_NOUN_PATTERN.finditer(query):
        noun = m.group(1)
        # Skip if starts at beginning of sentence (likely just capitalized)
        pos = m.start()
        if pos == 0:
            continue
        # Check preceding char — if it's a period/question mark + space, skip
        if pos >= 2 and query[pos - 2] in '.?!':
            continue
        # Skip exclusions
        if noun.split()[0] in PROPER_NOUN_EXCLUSIONS:
            continue
        proper_nouns.append(noun)

    return quoted, proper_nouns


def apply_entity_boost(
    results: List[Dict],
    query: str,
    quoted_boost: float = 0.6,
    noun_boost: float = 0.4,
    score_key: str = 'rrf_score'
) -> List[Dict]:
    """
    Boost results containing quoted phrases or proper nouns from the query.

    Args:
        results: Results with score_key field
        query: Original query string
        quoted_boost: Boost for exact quoted phrase matches (0.6 = 60%)
        noun_boost: Boost for proper noun matches (0.4 = 40%)
        score_key: Which score field to boost

    Returns:
        Results with adjusted scores and entity_boost metadata
    """
    quoted_phrases, proper_nouns = extract_entities(query)

    if not quoted_phrases and not proper_nouns:
        for r in results:
            r['entity_boost'] = 0.0
        return results

    logger.debug(f"Entities detected — quoted: {quoted_phrases}, nouns: {proper_nouns}")

    for result in results:
        content = result.get('original_chunk', '')
        content_lower = content.lower()
        total_boost = 0.0

        # Check quoted phrases (exact match, case-insensitive)
        for phrase in quoted_phrases:
            if phrase.lower() in content_lower:
                total_boost = max(total_boost, quoted_boost)

        # Check proper nouns
        for noun in proper_nouns:
            if noun.lower() in content_lower:
                total_boost = max(total_boost, noun_boost)

        if total_boost > 0:
            original_score = result.get(score_key, 0)
            result[score_key] = original_score * (1.0 + total_boost)
            result['entity_boost'] = total_boost
        else:
            result['entity_boost'] = 0.0

    results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
    return results


# =============================================================================
# Combined Enhancer Pipeline
# =============================================================================

def apply_all_enhancers(
    results: List[Dict],
    query: str,
    score_key: str = 'rrf_score',
    enable_temporal: bool = True,
    enable_preference: bool = True,
    enable_entity: bool = True,
    temporal_max_boost: float = 0.4,
    preference_max_boost: float = 0.5,
    quoted_boost: float = 0.6,
    noun_boost: float = 0.4,
    verbose: bool = False
) -> List[Dict]:
    """
    Apply all retrieval enhancers in sequence.

    Order matters: temporal → preference → entity (most specific last).

    Args:
        results: Fused results from RRF
        query: Original query string
        enable_*: Toggle individual enhancers
        *_boost: Boost parameters per enhancer
        verbose: Log enhancement details

    Returns:
        Enhanced and re-sorted results
    """
    enhanced = results

    if enable_temporal:
        enhanced = apply_temporal_boost(enhanced, query, temporal_max_boost, score_key)
        if verbose:
            boosted = sum(1 for r in enhanced if r.get('temporal_boost', 0) > 0)
            if boosted:
                logger.info(f"  Temporal boost applied to {boosted} results")

    if enable_preference:
        enhanced = apply_preference_boost(enhanced, query, preference_max_boost, score_key)
        if verbose:
            boosted = sum(1 for r in enhanced if r.get('preference_boost', 0) > 0)
            if boosted:
                logger.info(f"  Preference boost applied to {boosted} results")

    if enable_entity:
        enhanced = apply_entity_boost(enhanced, query, quoted_boost, noun_boost, score_key)
        if verbose:
            boosted = sum(1 for r in enhanced if r.get('entity_boost', 0) > 0)
            if boosted:
                logger.info(f"  Entity boost applied to {boosted} results")

    return enhanced
