"""Unified Guardrails pipeline orchestrating input, retrieval, tool, and output rails."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from guardrails.config import GuardrailsConfig
from guardrails.input_rails import check_jailbreak, check_topic
from guardrails.output_rails import PresidioMasker, verify_citation_grounding
from guardrails.retrieval_rails import sanitize_search_results

logger = logging.getLogger(__name__)


class GuardrailPipeline:
    """Full guardrail execution engine placed in front of and behind the C1 agent."""

    def __init__(self, config: GuardrailsConfig | None = None) -> None:
        self.config = config or GuardrailsConfig.from_env()
        self.masker = PresidioMasker() if self.config.pii_masking else None

    async def process_request(
        self,
        question: str,
        target_caller: Callable[[str], Any],
        language: str = "fr",
    ) -> dict[str, Any]:
        """Process a user question through input rails, target execution, and output rails.

        Args:
            question: The user input text.
            target_caller: Async callable taking (question: str) returning a response dict.
            language: Query language ("fr" or "en").

        Returns:
            Dictionary containing:
                answer: final guarded answer
                citations: citations returned
                tool_calls: tool calls made
                latency_ms: total execution latency
                rails_triggered: list of triggered rail names
                refusal: boolean indicating if question was refused
        """
        start = time.perf_counter()
        rails_triggered: list[str] = []

        if not self.config.enabled:
            raw_res = await target_caller(question)
            latency = round((time.perf_counter() - start) * 1000)
            return {
                "answer": raw_res.get("answer", ""),
                "citations": raw_res.get("citations", []),
                "tool_calls": raw_res.get("tool_calls", []),
                "latency_ms": latency,
                "rails_triggered": [],
                "refusal": False,
            }

        # 1. Input Rail: Jailbreak Detection
        if self.config.jailbreak_detection:
            is_jb, jb_reply = check_jailbreak(question)
            if is_jb:
                rails_triggered.append("input.jailbreak")
                latency = round((time.perf_counter() - start) * 1000)
                return {
                    "answer": jb_reply,
                    "citations": [],
                    "tool_calls": [],
                    "latency_ms": latency,
                    "rails_triggered": rails_triggered,
                    "refusal": True,
                }

        # 2. Input Rail: Topic Rail (French labour law scope)
        if self.config.topic_rail:
            is_off_topic, topic_reply = check_topic(question)
            if is_off_topic:
                rails_triggered.append("input.topic")
                latency = round((time.perf_counter() - start) * 1000)
                return {
                    "answer": topic_reply,
                    "citations": [],
                    "tool_calls": [],
                    "latency_ms": latency,
                    "rails_triggered": rails_triggered,
                    "refusal": True,
                }

        # 3. Call Target Agent
        target_res = await target_caller(question)
        answer = target_res.get("answer", "")
        citations = target_res.get("citations", [])
        tool_calls = target_res.get("tool_calls", [])

        # 4. Retrieval Rail: Sanitize search results if present
        if self.config.retrieval_rail and citations:
            citations = sanitize_search_results(citations)

        # 5. Output Rail: Citation Grounding Check
        if self.config.citation_check and answer:
            grounded_answer, is_grounded = verify_citation_grounding(
                answer=answer,
                retrieved_citations=citations,
                tool_calls=tool_calls,
            )
            if not is_grounded:
                rails_triggered.append("output.citation_check")
            answer = grounded_answer

        # 6. Output Rail: Presidio PII Masking
        if self.config.pii_masking and self.masker and answer:
            masked_answer = self.masker.mask(answer, language=language)
            if masked_answer != answer:
                rails_triggered.append("output.pii_masking")
                answer = masked_answer

        latency = round((time.perf_counter() - start) * 1000)
        is_refusal = (
            "demande refusée" in answer.lower()
            or "ne couvrent pas cette question" in answer.lower()
        )

        return {
            "answer": answer,
            "citations": citations,
            "tool_calls": tool_calls,
            "latency_ms": latency,
            "rails_triggered": rails_triggered,
            "refusal": is_refusal,
        }
