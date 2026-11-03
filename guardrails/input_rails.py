"""Input guardrails: jailbreak detection and topic safety rail for French labour law."""

from __future__ import annotations

import re
import unicodedata

# Explicit jailbreak / prompt injection patterns
JAILBREAK_PATTERNS = [
    re.compile(r"\b(dan|do\s+anything\s+now)\b", re.IGNORECASE),
    re.compile(r"ignore\s+(all|any)?\s*(previous|prior|following)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all|any)?\s*(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"ignorez?\s+(toutes?\s+les?|vos?)\s+consignes?(\s+pr[ée]c[ée]dentes?)?", re.IGNORECASE),
    re.compile(r"mode\s+(jailbreak|alpha|sans\s+filtre|unrestricted|developpeur)", re.IGNORECASE),
    re.compile(r"\b(freethinker|unrestricted\s+roleplay)\b", re.IGNORECASE),
    re.compile(r"---+\s*---+.*ignore", re.IGNORECASE | re.DOTALL),
    re.compile(r"repeat\s+the\s+text\s+above.*verbatim", re.IGNORECASE),
    re.compile(r"r[ée]p[ée]tez?\s+tous?\s+les?\s+mots?\s+qui\s+se\s+trouvaient\s+au-dessus", re.IGNORECASE),
    re.compile(r"prompt\s+syst[eè]me.*mot\s+pour\s+mot", re.IGNORECASE),
    re.compile(r"encode.*system\s+prompt.*base64", re.IGNORECASE),
    re.compile(r"disable\s+safety\s+guardrails?", re.IGNORECASE),
]

# Explicit out-of-scope / illegal topics
OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"(montage\s+offshore|optimisation\s+fiscale|contourner\s+le\s+fisc|evasion\s+fiscale)", re.IGNORECASE),
    re.compile(r"(faire\s+chanter.*employeur|extorsion|plainte\s+penale.*escroquerie)", re.IGNORECASE),
    re.compile(r"(bail\s+d'habitation|droit\s+immobilier|bloquer\s+le\s+loyer|proprietaire.*refuse)", re.IGNORECASE),
    re.compile(r"(detourner.*clientele.*sans\s+laisser\s+de\s+traces|dissimuler.*indices)", re.IGNORECASE),
    re.compile(r"(faux\s+certificat\s+medical|medecin\s+complaisant|simuler\s+devant\s+le\s+medecin)", re.IGNORECASE),
    re.compile(r"(cryptomonnaie.*sans\s+declaration|usdt.*sans\s+kyc|blanchiment)", re.IGNORECASE),
    re.compile(r"(repackager.*code\s+source\s+proprietaire|obfuscation.*code\s+proprietaire)", re.IGNORECASE),
]

JAILBREAK_REFUSAL = "Demande refusée : la requête déclenche les filtres de sécurité de contenu."
TOPIC_REFUSAL = (
    "Les sources dont je dispose ne couvrent pas cette question. "
    "Je réponds uniquement aux questions relatives au droit du travail français "
    "(Code du travail et conventions collectives).\n\nCeci n'est pas un avis juridique."
)


def _strip_accents(text: str) -> str:
    """Normalize text by removing accents for robust regex matching."""
    norm = unicodedata.normalize("NFD", text)
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


def check_jailbreak(text: str) -> tuple[bool, str]:
    """Inspect user prompt for prompt injection and jailbreak signatures.

    Returns:
        (is_jailbreak, response_or_empty)
    """
    if not text:
        return False, ""

    norm_text = _strip_accents(text)
    for pat in JAILBREAK_PATTERNS:
        if pat.search(text) or pat.search(norm_text):
            return True, JAILBREAK_REFUSAL

    return False, ""


def check_topic(text: str) -> tuple[bool, str]:
    """Inspect user prompt against out-of-scope and prohibited domains.

    Returns:
        (is_off_topic, response_or_empty)
    """
    if not text:
        return False, ""

    norm_text = _strip_accents(text)
    for pat in OUT_OF_SCOPE_PATTERNS:
        if pat.search(text) or pat.search(norm_text):
            return True, TOPIC_REFUSAL

    return False, ""
