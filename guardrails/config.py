"""Configuration settings for runtime guardrails."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class GuardrailsConfig:
    enabled: bool = True
    jailbreak_detection: bool = True
    topic_rail: bool = True
    retrieval_rail: bool = True
    tool_policy: bool = True
    pii_masking: bool = True
    citation_check: bool = True

    # PII entities to detect and redact
    pii_entities: list[str] = field(
        default_factory=lambda: [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "IBAN_CODE",
            "CRYPTO",
            "IP_ADDRESS",
        ]
    )

    # Allowed in-domain themes (French labour law)
    allowed_domains: list[str] = field(
        default_factory=lambda: [
            "droit du travail",
            "code du travail",
            "convention collective",
            "contrat de travail",
            "cdi",
            "cdd",
            "periode d'essai",
            "licenciement",
            "rupture conventionnelle",
            "demission",
            "preavis",
            "indemnite de licenciement",
            "conges payes",
            "rtt",
            "heures supplementaires",
            "temps de travail",
            "teletravail",
            "arret maladie",
            "inaptitude",
            "medecine du travail",
            "harcelement",
            "cse",
            "prud'hommes",
            "inspection du travail",
            "syntec",
            "hcr",
            "titres-restaurant",
        ]
    )

    # Explicitly blocked out-of-scope themes
    blocked_domains: list[str] = field(
        default_factory=lambda: [
            "optimisation fiscale",
            "evasion fiscale",
            "montage offshore",
            "impot sur les societes",
            "chanteur",
            "faire chanter",
            "extorsion",
            "plainte penale pour escroquerie",
            "bail d'habitation",
            "droit immobilier",
            "bloquer le loyer",
            "proprietaire",
            "detourner la clientele",
            "supprimer les traces",
            "fausses factures",
            "cryptomonnaie",
            "blanchiment",
            "usdt",
            "sans kyc",
            "faux certificat",
            "script bash",
            "piratage",
            "injection sql",
        ]
    )

    @classmethod
    def from_env(cls) -> GuardrailsConfig:
        return cls(
            enabled=os.getenv("ENABLE_GUARDRAILS", "true").lower() in ("true", "1", "yes"),
            jailbreak_detection=os.getenv("GUARDRAIL_JAILBREAK", "true").lower() in ("true", "1", "yes"),
            topic_rail=os.getenv("GUARDRAIL_TOPIC", "true").lower() in ("true", "1", "yes"),
            retrieval_rail=os.getenv("GUARDRAIL_RETRIEVAL", "true").lower() in ("true", "1", "yes"),
            tool_policy=os.getenv("GUARDRAIL_TOOL_POLICY", "true").lower() in ("true", "1", "yes"),
            pii_masking=os.getenv("GUARDRAIL_PII", "true").lower() in ("true", "1", "yes"),
            citation_check=os.getenv("GUARDRAIL_CITATION", "true").lower() in ("true", "1", "yes"),
        )
