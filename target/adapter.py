"""OpenAI-compatible /v1/chat/completions in front of C1's POST /ask, for garak and PyRIT.

C1 is stateless and takes one question of at most 2000 characters, so earlier turns are folded
into that question as a transcript (oldest dropped first). Sampling parameters are ignored:
the agent picks its own.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request

load_dotenv()
from fastapi.responses import JSONResponse
from opentelemetry import propagate, trace
from pydantic import BaseModel

from guardrails.input_rails import check_jailbreak, check_topic
from guardrails.output_rails import verify_citation_grounding
from guardrails.pipeline import GuardrailPipeline
from guardrails.retrieval_rails import sanitize_search_results
from target import tracing

MAX_QUESTION = 2000  # C1's AskRequest.question max_length
MIN_QUESTION = 3
ROLE_LABELS = {"user": "Utilisateur", "assistant": "Assistant", "system": "Système",
               "developer": "Système"}  # fmt: skip
HISTORY_HEADER = "Conversation précédente :"
QUESTION_HEADER = "Question :"

tracer = trace.get_tracer("adapter")  # proxy: follows whichever provider tracing.setup installs


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    # temperature, max_tokens, stop, n... are accepted and ignored


def text_of(content) -> str:
    """Plain text of a message; list content keeps only its text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))


def flatten(messages: list[ChatMessage], limit: int = MAX_QUESTION) -> tuple[str, int]:
    """One question for /ask and the number of earlier turns dropped to fit `limit`.

    The last user message is the question. Earlier messages become a labelled transcript;
    the oldest are dropped first. The question itself is cut only if it alone is too long.
    """
    turns = [(m.role, text_of(m.content).strip()) for m in messages]
    turns = [(r, t) for r, t in turns if t]
    last_user = max((i for i, (r, _) in enumerate(turns) if r == "user"), default=None)
    if last_user is None:
        return "", 0
    question = turns[last_user][1]
    history = [f"{ROLE_LABELS.get(r, r)} : {t}" for r, t in turns[:last_user]]
    if not history:
        return question[:limit], 0

    def build(lines: list[str]) -> str:
        return "\n".join([HISTORY_HEADER, *lines, "", QUESTION_HEADER, question])

    dropped = 0
    while history and len(build(history)) > limit:
        history.pop(0)
        dropped += 1
    if not history:
        return question[:limit], dropped
    return build(history), dropped


def question_truncated(messages: list[ChatMessage], limit: int = MAX_QUESTION) -> bool:
    users = [text_of(m.content).strip() for m in messages if m.role == "user"]
    users = [u for u in users if u]
    return bool(users) and len(users[-1]) > limit


def openai_error(status: int, message: str, type_: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": type_}})


def completion(model: str, answer: str, extra: dict) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
        # C1's /ask doesn't report token usage
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "c1": extra,
    }


def targets_from_env() -> dict[str, str]:
    """Model name -> C1 base URL."""
    c1_base = os.environ.get("C1_URL") or "http://127.0.0.1:8001"
    targets = {
        "c1": c1_base,
        "c1-guarded": c1_base,
    }
    redteam_base = os.environ.get("C1_REDTEAM_URL") or "http://127.0.0.1:8002"
    if redteam_base:
        targets["c1-redteam"] = redteam_base
        targets["c1-redteam-guarded"] = redteam_base
    return targets


def create_app(
    targets: dict[str, str] | None = None,
    api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 120.0,
) -> FastAPI:
    targets = targets if targets is not None else targets_from_env()
    api_key = api_key if api_key is not None else os.environ.get("ADAPTER_API_KEY", "")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(transport=transport, timeout=timeout)
        app.state.pipeline = GuardrailPipeline()
        yield
        await app.state.http.aclose()

    app = FastAPI(title="C1 OpenAI-compatible adapter", lifespan=lifespan)

    @app.middleware("http")
    async def check_key(request: Request, call_next):
        authorised = request.headers.get("authorization") == f"Bearer {api_key}"
        if request.url.path.startswith("/v1/") and api_key and not authorised:
            return openai_error(401, "invalid API key", "authentication_error")
        return await call_next(request)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list",
                "data": [{"id": m, "object": "model", "owned_by": "c1"} for m in targets]}  # fmt: skip

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request):
        if body.model not in targets:
            return openai_error(404, f"unknown model {body.model!r}; use one of {list(targets)}")
        question, dropped = flatten(body.messages)
        if len(question) < MIN_QUESTION:
            return openai_error(400, f"the last user message needs at least {MIN_QUESTION} chars")
        attack_id = request.headers.get("x-attack-id", "")
        campaign = request.headers.get("x-campaign", "")
        attrs = tracing.attack_attributes(attack_id, campaign, body.model)
        # Root span of the Langfuse trace; its attributes travel to C1 as baggage.
        is_guarded = (
            "guarded" in body.model
            or request.headers.get("x-guardrails", "").lower() in ("true", "1")
            or os.environ.get("ENABLE_GUARDRAILS", "").lower() in ("true", "1")
        )

        # 1. Input Guardrails
        if is_guarded:
            is_jb, jb_reply = check_jailbreak(question)
            if is_jb:
                extra = {
                    "citations": [],
                    "tool_calls": [],
                    "trace_id": "",
                    "latency_ms": 2,
                    "question_sent": question,
                    "turns_dropped": dropped,
                    "question_truncated": False,
                    "attack_id": attack_id,
                    "campaign": campaign,
                    "content_filtered": True,
                    "refusal_source": "guardrail_jailbreak",
                    "guardrails_applied": True,
                    "rails_triggered": ["input.jailbreak"],
                }
                return completion(body.model, jb_reply, extra)

            is_topic, topic_reply = check_topic(question)
            if is_topic:
                extra = {
                    "citations": [],
                    "tool_calls": [],
                    "trace_id": "",
                    "latency_ms": 2,
                    "question_sent": question,
                    "turns_dropped": dropped,
                    "question_truncated": False,
                    "attack_id": attack_id,
                    "campaign": campaign,
                    "content_filtered": True,
                    "refusal_source": "guardrail_topic",
                    "guardrails_applied": True,
                    "rails_triggered": ["input.topic"],
                }
                return completion(body.model, topic_reply, extra)

        with tracing.baggage_scope(attrs), tracer.start_as_current_span(
            attrs["langfuse.trace.name"], kind=trace.SpanKind.SERVER, attributes=attrs
        ) as span:
            span.set_attribute("langfuse.trace.tags", [t for t in (campaign, body.model) if t])
            span.set_attribute("langfuse.observation.input", question)
            headers: dict[str, str] = {}
            propagate.inject(headers)
            start = time.perf_counter()
            try:
                resp = await request.app.state.http.post(
                    f"{targets[body.model]}/ask", json={"question": question}, headers=headers
                )
            except httpx.HTTPError as err:
                span.set_status(trace.StatusCode.ERROR, type(err).__name__)
                return openai_error(502, f"C1 unreachable: {type(err).__name__}", "api_error")
            latency_ms = round((time.perf_counter() - start) * 1000)
            if resp.status_code == 422:
                span.set_status(trace.StatusCode.ERROR, "C1 422")
                return openai_error(400, f"C1 rejected the question: {resp.text[:500]}")
            if resp.status_code != 200:
                span.set_status(trace.StatusCode.ERROR, f"C1 {resp.status_code}")
                return openai_error(502, f"C1 returned {resp.status_code}", "api_error")
            data = resp.json()
            if data.get("content_filtered"):
                span.set_attribute("c1.content_filtered", True)
                if data.get("refusal_source"):
                    span.set_attribute("c1.refusal_source", data["refusal_source"])
            span.set_attribute("langfuse.observation.output", data.get("answer", ""))

        answer = data.get("answer", "")
        citations = data.get("citations", [])
        tool_calls = data.get("tool_calls", [])
        rails_triggered = []

        # 2. Retrieval & Output Guardrails
        if is_guarded:
            if citations:
                citations = sanitize_search_results(citations)

            if answer:
                grounded_answer, is_grounded = verify_citation_grounding(
                    answer, retrieved_citations=citations, tool_calls=tool_calls
                )
                if not is_grounded:
                    rails_triggered.append("output.citation_check")
                answer = grounded_answer

                pipeline = getattr(request.app.state, "pipeline", None)
                if pipeline and pipeline.masker:
                    masked_answer = pipeline.masker.mask(answer)
                    if masked_answer != answer:
                        rails_triggered.append("output.pii_masking")
                    answer = masked_answer

        extra = {
            "citations": citations,
            "tool_calls": tool_calls,
            "trace_id": data.get("trace_id", ""),  # same id as the adapter's trace when traced
            "latency_ms": latency_ms,
            "question_sent": question,
            "turns_dropped": dropped,
            "question_truncated": question_truncated(body.messages),
            "attack_id": attack_id,
            "campaign": campaign,
            "content_filtered": data.get("content_filtered", False),
            "refusal_source": data.get("refusal_source"),
            "guardrails_applied": is_guarded,
            "rails_triggered": rails_triggered,
        }
        return completion(body.model, answer, extra)

    return app


tracing.setup("adapter")
app = create_app()
