"""Output guardrails: Presidio PII masking and citation grounding verification."""

from __future__ import annotations

import logging
import re
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

# French NIR (Numéro de Sécurité Sociale) pattern: 1 85 05 75 ...
NIR_PATTERN = re.compile(r"\b[12]\s*\d{2}\s*(0[1-9]|1[0-2])\s*\d{2}\s*\d{3}\s*\d{3}(\s*\d{2})?\b")

# French Phone format (e.g. 06 01 02 03 04, 01.23.45.67.89, +33 6 ...)
FR_PHONE_PATTERN = re.compile(r"\b(0[1-9](?:[\s\.-]?\d{2}){4}|\+33[\s\.-]?[1-9](?:[\s\.-]?\d{2}){4})\b")

# Regex to detect article citations in text (e.g. L1234-1, R1234-2, article L. 1221-1, etc.)
CITATION_REGEX = re.compile(
    r"\b(article\s+)?([LRDA]\.?\s*\d+([-\.]\d+)*|IDCC\s*\d+)\b", re.IGNORECASE
)


# Targeted PII entities to mask in responses without stripping legitimate citations or URLs
TARGET_PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "FR_NIR",
    "IBAN_CODE",
    "CREDIT_CARD",
    "PERSON",
]


class PresidioMasker:
    """Masks Personally Identifiable Information (PII) using Microsoft Presidio."""

    def __init__(self) -> None:
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_sm"},
                {"lang_code": "fr", "model_name": "fr_core_news_sm"},
            ],
        }
        try:
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=["en", "fr"])
        except Exception as ex:  # noqa: BLE001
            logger.warning(f"Failed to initialize spaCy NLP engine for Presidio ({ex}); falling back to default")
            self.analyzer = AnalyzerEngine(supported_languages=["en", "fr"])

        # Add custom French NIR and Phone recognizers
        nir_recognizer = PatternRecognizer(
            supported_entity="FR_NIR",
            patterns=[Pattern(name="fr_nir_pattern", regex=NIR_PATTERN.pattern, score=0.85)],
            supported_language="fr",
        )
        self.analyzer.registry.add_recognizer(nir_recognizer)

        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[Pattern(name="fr_phone_pattern", regex=FR_PHONE_PATTERN.pattern, score=0.85)],
            supported_language="fr",
        )
        self.analyzer.registry.add_recognizer(phone_recognizer)
        self.anonymizer = AnonymizerEngine()

    def mask(self, text: str, language: str = "fr") -> str:
        """Analyze text and replace detected PII with anonymized tags."""
        if not text:
            return ""

        lang = "fr" if language.lower().startswith("fr") else "en"
        results = self.analyzer.analyze(
            text=text,
            language=lang,
            entities=TARGET_PII_ENTITIES,
            score_threshold=0.7,
        )
        if not results:
            return text

        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text


def verify_citation_grounding(
    answer: str,
    retrieved_citations: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> tuple[str, bool]:
    """Check that every substantive answer cites at least one article returned by search_code.

    Returns:
        (grounded_answer, is_grounded)
    """
    if not answer:
        return "", True

    # Safe refusals do not need article citations
    lower_ans = answer.lower()
    if any(
        phrase in lower_ans
        for phrase in [
            "ne couvrent pas",
            "demande refusée",
            "je ne peux pas",
            "pas un avis juridique",
            "hors périmètre",
        ]
    ) and len(answer) < 300:
        return answer, True

    # Extract available article numbers from retrieved citations and tool calls
    available_nums: set[str] = set()
    if retrieved_citations:
        for c in retrieved_citations:
            num = c.get("num")
            if num:
                available_nums.add(num.upper().replace(" ", "").replace(".", ""))

    if tool_calls:
        for tc in tool_calls:
            for num in tc.get("nums", []):
                available_nums.add(num.upper().replace(" ", "").replace(".", ""))

    # Check citations mentioned in answer text
    cited_in_text = CITATION_REGEX.findall(answer)
    cleaned_cites = {
        c[1].upper().replace(" ", "").replace(".", "") for c in cited_in_text if c[1]
    }

    # Answer is grounded if it cites at least one available article or valid article number
    if cleaned_cites:
        # Check intersection with available articles if citations were returned
        if available_nums and not (cleaned_cites & available_nums):
            # Cites an article, but none of the retrieved ones
            warning = (
                "\n\n[Avertissement de vérification : l'article cité n'a pas été formellement "
                "validé par le moteur de recherche interne.]"
            )
            return answer + warning, False
        return answer, True

    # No citations found in substantive text
    if len(answer.strip()) > 50:
        disclaimer = (
            "\n\n[Note de conformité : Cette réponse n'a pas pu être recoupée directement "
            "avec un article précis issu des sources vérifiées.]"
        )
        return answer + disclaimer, False

    return answer, True
