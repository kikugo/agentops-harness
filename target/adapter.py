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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MAX_QUESTION = 2000  # C1's AskRequest.question max_length
MIN_QUESTION = 3
ROLE_LABELS = {"user": "Utilisateur", "assistant": "Assistant", "system": "Système",
               "developer": "Système"}  # fmt: skip
HISTORY_HEADER = "Conversation précédente :"
QUESTION_HEADER = "Question :"


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
    targets = {"c1": os.environ.get("C1_URL", "http://c1:8000")}
    if os.environ.get("C1_REDTEAM_URL"):
        targets["c1-redteam"] = os.environ["C1_REDTEAM_URL"]
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
        start = time.perf_counter()
        try:
            resp = await request.app.state.http.post(
                f"{targets[body.model]}/ask", json={"question": question}
            )
        except httpx.HTTPError as err:
            return openai_error(502, f"C1 unreachable: {type(err).__name__}", "api_error")
        latency_ms = round((time.perf_counter() - start) * 1000)
        if resp.status_code == 422:
            return openai_error(400, f"C1 rejected the question: {resp.text[:500]}")
        if resp.status_code != 200:
            return openai_error(502, f"C1 returned {resp.status_code}", "api_error")
        data = resp.json()
        extra = {
            "citations": data.get("citations", []),
            "tool_calls": data.get("tool_calls", []),
            "trace_id": data.get("trace_id", ""),
            "latency_ms": latency_ms,
            "question_sent": question,
            "turns_dropped": dropped,
            "question_truncated": question_truncated(body.messages),
            "attack_id": request.headers.get("x-attack-id", ""),
            "campaign": request.headers.get("x-campaign", ""),
        }
        return completion(body.model, data.get("answer", ""), extra)

    return app


app = create_app()
