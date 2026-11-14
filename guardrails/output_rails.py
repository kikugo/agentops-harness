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
# No \b before "+33": a word boundary never sits between a space and "+".
FR_PHONE_PATTERN = re.compile(r"(?:\b0[1-9](?:[\s\.-]?\d{2}){4}|\+33[\s\.-]?[1-9](?:[\s\.-]?\d{2}){4})\b")

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


# Words and phrases in a labour-law answer that spaCy's small NER models tag as PERSON.
# Passed to Presidio's analyze(allow_list=...), matched as whole-span regexes.
LEGAL_ALLOW_LIST = [
    r"(?:l|d|n|qu|s|c|j)['’].*",  # elided French words: n’, l’article, d’indemnité, L’outil
    r"(?:article|art\.)\s.*",
    r"[LRD]\.?\s?\d[\d\s.-]*\**",  # article numbers, e.g. L1234-1, R. 1234-2
    r"(?:code|conseil|cour|tribunal|convention|accord|syntec|hcr|urssaf|cse|dreets).*",
    r"préavis|indemnité|licenciement|démission|salarié|employeur|ancienneté|faute",
]

# Lower-case name particles allowed inside a person name ("Jean de La Fontaine").
NAME_PARTICLES = {"de", "du", "des", "la", "le", "van", "von", "da", "di", "del", "ben", "el"}
_NAME_WORD = re.compile(r"^[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:-[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)*$")


def plausible_person(span: str) -> bool:
    """Shape check for a PERSON span: two or more capitalised name words, no digits.

    The spaCy small models flag sentence-initial words ("Cela"), elisions ("n’") and legal
    terms ("préavis", "l’article L1234-1") as people. A full name has at least two capitalised
    words; a lone surname is missed, which is the trade-off chosen here.
    """
    if any(ch.isdigit() for ch in span) or "@" in span or "*" in span:
        return False
    words = span.split()
    name_words = [w for w in words if w.lower() not in NAME_PARTICLES]
    if len(name_words) < 2:
        return False
    return all(_NAME_WORD.match(w) for w in name_words)


class PresidioMasker:
    """Masks Personally Identifiable Information (PII) using Microsoft Presidio.

    strict_person=False reproduces the first version (every spaCy PERSON hit is masked);
    it is kept only so scripts/replay_output_rails.py can measure the difference.
    """

    def __init__(self, strict_person: bool = True) -> None:
        self.strict_person = strict_person
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
        results = self.analyze(text, lang)
        if not results:
            return text

        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text

    def analyze(self, text: str, lang: str) -> list:
        """Presidio results after the legal allow list and the PERSON shape check."""
        if not self.strict_person:
            return self.analyzer.analyze(
                text=text, language=lang, entities=TARGET_PII_ENTITIES, score_threshold=0.7
            )
        results = self.analyzer.analyze(
            text=text,
            language=lang,
            entities=TARGET_PII_ENTITIES,
            score_threshold=0.7,
            allow_list=LEGAL_ALLOW_LIST,
            allow_list_match="regex",
            regex_flags=re.IGNORECASE | re.DOTALL,
        )
        kept = []
        for r in results:
            if r.entity_type == "PERSON":
                if not plausible_person(text[r.start : r.end]):
                    continue
                # An e-mail address is masked as EMAIL_ADDRESS, not as a person.
                if any(o.entity_type != "PERSON" and o.start <= r.start and r.end <= o.end for o in results):
                    continue
            kept.append(r)
        return kept


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
