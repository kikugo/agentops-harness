"""Tool argument validation policy executed outside the model in code."""

from __future__ import annotations

import re
from typing import Any

VALID_STATUTS = {"normal", "faute_grave", "faute_lourde"}
VALID_SOURCES = {"code", "IDCC 1486", "IDCC 1979", None}

# Legal article format (e.g. L1234-1, R1234-2, D1234-3, L1233-65, 12, etc.)
ARTICLE_NUM_RE = re.compile(r"^[LRDA]?\s*\d+([-\.]\d+)*$", re.IGNORECASE)

# Dangerous injection / traversal signatures
DANGEROUS_PATTERNS = [
    re.compile(r"(--|/\*|\*/|;|DROP\s+TABLE|SELECT\s+|UNION\s+SELECT|OR\s+['\"]?1['\"]?\s*=\s*['\"]?1)", re.IGNORECASE),
    re.compile(r"(\.\./|\.\.\\|/etc/|/app/|cat\s+/)", re.IGNORECASE),
    re.compile(r"(<script|</script|javascript:|alert\()", re.IGNORECASE),
]


class ToolValidationError(ValueError):
    """Raised when a tool argument fails range or format validation."""

    def __init__(self, tool_name: str, argument: str, value: Any, message: str) -> None:
        super().__init__(f"Tool policy violation in '{tool_name}' for argument '{argument}'={value!r}: {message}")
        self.tool_name = tool_name
        self.argument = argument
        self.value = value
        self.message = message


def validate_preavis_args(anciennete_mois: Any, statut: str = "normal") -> dict[str, Any]:
    """Validate arguments for preavis calculator."""
    if isinstance(anciennete_mois, bool) or not isinstance(anciennete_mois, int | float):
        raise ToolValidationError("preavis", "anciennete_mois", anciennete_mois, "Must be an integer number of months")

    val_int = int(anciennete_mois)
    if val_int < 0:
        raise ToolValidationError("preavis", "anciennete_mois", anciennete_mois, "Seniority cannot be negative")

    if val_int > 1200:  # Max 100 years career limit to prevent integer overflow
        raise ToolValidationError("preavis", "anciennete_mois", anciennete_mois, "Seniority exceeds realistic human maximum")

    if statut not in VALID_STATUTS:
        raise ToolValidationError(
            "preavis", "statut", statut, f"Invalid status: must be one of {sorted(VALID_STATUTS)}"
        )

    return {"anciennete_mois": val_int, "statut": statut}


def validate_indemnite_args(
    anciennete_mois: Any, salaire_ref: Any, statut: str = "normal"
) -> dict[str, Any]:
    """Validate arguments for indemnite_licenciement calculator."""
    if isinstance(anciennete_mois, bool) or not isinstance(anciennete_mois, int | float):
        raise ToolValidationError(
            "indemnite_licenciement", "anciennete_mois", anciennete_mois, "Must be an integer number of months"
        )

    mois_int = int(anciennete_mois)
    if mois_int < 0:
        raise ToolValidationError(
            "indemnite_licenciement", "anciennete_mois", anciennete_mois, "Seniority cannot be negative"
        )

    if mois_int > 1200:
        raise ToolValidationError(
            "indemnite_licenciement", "anciennete_mois", anciennete_mois, "Seniority exceeds realistic maximum"
        )

    if isinstance(salaire_ref, bool) or not isinstance(salaire_ref, int | float):
        raise ToolValidationError(
            "indemnite_licenciement", "salaire_ref", salaire_ref, "Must be a numeric salary in euros"
        )

    sal_float = float(salaire_ref)
    if sal_float < 0:
        raise ToolValidationError(
            "indemnite_licenciement", "salaire_ref", salaire_ref, "Reference salary cannot be negative"
        )

    if statut not in VALID_STATUTS:
        raise ToolValidationError(
            "indemnite_licenciement", "statut", statut, f"Invalid status: must be one of {sorted(VALID_STATUTS)}"
        )

    return {"anciennete_mois": mois_int, "salaire_ref": sal_float, "statut": statut}


def validate_get_article_args(num: Any, source: str | None = None) -> dict[str, Any]:
    """Validate arguments for get_article tool."""
    if not isinstance(num, str):
        raise ToolValidationError("get_article", "num", num, "Article number must be a string")

    clean_num = num.strip()
    if not clean_num:
        raise ToolValidationError("get_article", "num", num, "Article number cannot be empty")

    for pat in DANGEROUS_PATTERNS:
        if pat.search(clean_num):
            raise ToolValidationError(
                "get_article", "num", num, "Article identifier contains prohibited SQL, script, or path traversal patterns"
            )

    if not ARTICLE_NUM_RE.match(clean_num):
        raise ToolValidationError(
            "get_article", "num", num, f"Invalid article identifier format: {clean_num!r}"
        )

    if source is not None and source not in VALID_SOURCES:
        raise ToolValidationError("get_article", "source", source, f"Invalid source: must be one of {VALID_SOURCES}")

    return {"num": clean_num, "source": source}


def validate_search_code_args(query: Any, source: str | None = None) -> dict[str, Any]:
    """Validate arguments for search_code tool."""
    if not isinstance(query, str):
        raise ToolValidationError("search_code", "query", query, "Query must be a string")

    clean_query = query.strip()
    if not clean_query:
        raise ToolValidationError("search_code", "query", query, "Query cannot be empty")

    if len(clean_query) > 1000:
        raise ToolValidationError("search_code", "query", query, "Query exceeds maximum allowed length of 1000 characters")

    for pat in DANGEROUS_PATTERNS:
        if pat.search(clean_query):
            raise ToolValidationError(
                "search_code", "query", query, "Query contains prohibited command injection or script syntax"
            )

    if source is not None and source not in VALID_SOURCES:
        raise ToolValidationError("search_code", "source", source, f"Invalid source: must be one of {VALID_SOURCES}")

    return {"query": clean_query, "source": source}
