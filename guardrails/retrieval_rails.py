"""Retrieval rail that strips instruction-like text from retrieved chunks."""

from __future__ import annotations

import re

INSTRUCTION_PATTERNS = [
    re.compile(r"ignore\b.*\binstructions?", re.IGNORECASE),
    re.compile(r"disregard\b.*\binstructions?", re.IGNORECASE),
    re.compile(r"ignorez?\b.*\bconsignes?", re.IGNORECASE),
    re.compile(r"ne\s+tenez?\s+pas\s+compte\s+des?\s+instructions?", re.IGNORECASE),
    re.compile(r"^(system|admin|directive|command|instruction)\s*:\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"---+\s*---+", re.IGNORECASE),
    re.compile(r"<\s*\|?\s*im_(start|end)\s*\|?\s*>", re.IGNORECASE),
    re.compile(r"###\s*instruction\s*:", re.IGNORECASE),
    re.compile(r"curl\s+-[sS]?\s+https?://", re.IGNORECASE),
    re.compile(r"execute\s+the\s+(following|embedded)\s+script", re.IGNORECASE),
    re.compile(r"ex[ée]cutez?\s+le\s+script", re.IGNORECASE),
    re.compile(r"(exfiltrat|token|mot de passe|password)\s*:\s*[a-zA-Z0-9_\-]+", re.IGNORECASE),
]

SANITIZED_NOTICE = "[Contenu d'instruction non sollicité neutralisé]"


def sanitize_retrieved_chunk(text: str) -> tuple[str, bool]:
    """Sanitize a retrieved chunk by stripping instruction-like text.

    Returns:
        (sanitized_text, was_modified)
    """
    if not text:
        return "", False

    modified = False
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line_stripped = line.strip()
        matched = False
        for pat in INSTRUCTION_PATTERNS:
            if pat.search(line_stripped):
                matched = True
                modified = True
                break

        if matched:
            cleaned_lines.append(SANITIZED_NOTICE)
        else:
            cleaned_lines.append(line)

    sanitized = "\n".join(cleaned_lines)

    # Secondary pass over full text for multi-line delimiter hacks
    for pat in INSTRUCTION_PATTERNS[:4]:
        if pat.search(sanitized):
            sanitized = pat.sub(SANITIZED_NOTICE, sanitized)
            modified = True

    return sanitized, modified


def sanitize_search_results(results: list[dict]) -> list[dict]:
    """Sanitize all text fields in search hit dictionaries."""
    sanitized_hits = []
    for hit in results:
        hit_copy = dict(hit)
        if "texte" in hit_copy and isinstance(hit_copy["texte"], str):
            clean_text, _ = sanitize_retrieved_chunk(hit_copy["texte"])
            hit_copy["texte"] = clean_text
        if "text" in hit_copy and isinstance(hit_copy["text"], str):
            clean_text, _ = sanitize_retrieved_chunk(hit_copy["text"])
            hit_copy["text"] = clean_text
        sanitized_hits.append(hit_copy)
    return sanitized_hits
