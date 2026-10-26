import json

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from target.adapter import (
    HISTORY_HEADER,
    MAX_QUESTION,
    QUESTION_HEADER,
    ChatMessage,
    create_app,
    flatten,
)

KEY = "test-key"
C1_REPLY = {
    "answer": "Le préavis est de 2 mois.\n\nSources :\n- article L1234-1",
    "citations": [{"num": "L1234-1", "url": "https://x", "source": "code",
                   "date_debut_vigueur": "2008-05-01"}],  # fmt: skip
    "tool_calls": [{"name": "preavis", "args": {"anciennete_mois": 36}, "nums": ["L1234-1"]}],
    "trace_id": "ab" * 16,
}


def msgs(*pairs):
    return [ChatMessage(role=r, content=c) for r, c in pairs]


class FakeC1:
    def __init__(self, status=200, body=None, error=None):
        self.status, self.body, self.error = status, body or C1_REPLY, error
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error:
            raise self.error
        return httpx.Response(self.status, json=self.body)

    @property
    def questions(self):
        return [json.loads(r.content)["question"] for r in self.requests]


@pytest.fixture
def c1():
    return FakeC1()


def client_for(c1, **kw):
    app = create_app(
        targets={"c1": "http://c1:8000", "c1-redteam": "http://c1-redteam:8000"},
        api_key=KEY,
        transport=httpx.MockTransport(c1),
        **kw,
    )
    return TestClient(app, headers={"Authorization": f"Bearer {KEY}"})


def post(client, messages, model="c1", **extra):
    return client.post("/v1/chat/completions", json={"model": model, "messages": messages, **extra})


# flatten


def test_single_user_message_is_sent_as_is():
    assert flatten(msgs(("user", "Quel préavis ?"))) == ("Quel préavis ?", 0)


def test_history_becomes_labelled_transcript_in_order():
    q, dropped = flatten(msgs(("system", "Sois bref."), ("user", "Bonjour"),
                              ("assistant", "Bonjour !"), ("user", "Quel préavis ?")))  # fmt: skip
    assert dropped == 0
    assert q == "\n".join([HISTORY_HEADER, "Système : Sois bref.", "Utilisateur : Bonjour",
                           "Assistant : Bonjour !", "", QUESTION_HEADER, "Quel préavis ?"])  # fmt: skip


def test_oldest_turns_dropped_to_fit_limit():
    history = [("user", f"tour {i} " + "x" * 300) for i in range(10)]
    q, dropped = flatten(msgs(*history, ("user", "dernière question")))
    assert len(q) <= MAX_QUESTION
    assert dropped > 0
    assert "tour 9" in q and "tour 0" not in q
    assert q.endswith("dernière question")


def test_overlong_last_question_is_cut_and_history_dropped():
    q, dropped = flatten(msgs(("user", "avant"), ("user", "y" * 3000)))
    assert q == "y" * MAX_QUESTION
    assert dropped == 1


def test_list_content_keeps_text_parts():
    m = [ChatMessage(role="user", content=[{"type": "text", "text": "Quel préavis ?"},
                                           {"type": "image_url", "image_url": {"url": "x"}}])]  # fmt: skip
    assert flatten(m) == ("Quel préavis ?", 0)


def test_no_user_message_gives_empty_question():
    assert flatten(msgs(("system", "Sois bref."))) == ("", 0)


# endpoint


def test_completion_shape_and_c1_extra(c1):
    with client_for(c1) as client:
        r = post(client, [{"role": "user", "content": "Quel préavis après 3 ans ?"}],
                 temperature=0.7, max_tokens=150, stop=["#", ";"])  # fmt: skip
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": C1_REPLY["answer"]}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["c1"]["citations"] == C1_REPLY["citations"]
    assert body["c1"]["tool_calls"] == C1_REPLY["tool_calls"]
    assert body["c1"]["trace_id"] == C1_REPLY["trace_id"]
    assert c1.questions == ["Quel préavis après 3 ans ?"]  # sampling params never reach C1
    assert str(c1.requests[0].url) == "http://c1:8000/ask"


def test_model_routes_to_redteam_target(c1):
    with client_for(c1) as client:
        post(client, [{"role": "user", "content": "Quel préavis ?"}], model="c1-redteam")
    assert str(c1.requests[0].url) == "http://c1-redteam:8000/ask"


def test_unknown_model_is_404(c1):
    with client_for(c1) as client:
        r = post(client, [{"role": "user", "content": "Quel préavis ?"}], model="gpt-4o")
    assert r.status_code == 404
    assert "unknown model" in r.json()["error"]["message"]
    assert c1.requests == []


def test_missing_or_wrong_key_is_401(c1):
    with client_for(c1) as client:
        r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer nope"},
                        json={"model": "c1", "messages": [{"role": "user", "content": "abc"}]})  # fmt: skip
    assert r.status_code == 401
    assert c1.requests == []


def test_health_needs_no_key(c1):
    with client_for(c1) as client:
        assert client.get("/health", headers={"Authorization": ""}).json() == {"status": "ok"}


def test_too_short_question_is_400_without_calling_c1(c1):
    with client_for(c1) as client:
        r = post(client, [{"role": "user", "content": "a"}])
    assert r.status_code == 400
    assert c1.requests == []


def test_c1_422_becomes_400():
    c1 = FakeC1(status=422, body={"detail": "too long"})
    with client_for(c1) as client:
        r = post(client, [{"role": "user", "content": "Quel préavis ?"}])
    assert r.status_code == 400


@pytest.mark.parametrize("c1", [FakeC1(status=500), FakeC1(error=httpx.ReadTimeout("slow"))])
def test_c1_failure_becomes_502(c1):
    with client_for(c1) as client:
        r = post(client, [{"role": "user", "content": "Quel préavis ?"}])
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "api_error"


def test_attack_headers_and_truncation_are_reported(c1):
    with client_for(c1) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"X-Attack-Id": "a-001", "X-Campaign": "crescendo"},
            json={"model": "c1", "messages": [{"role": "user", "content": "z" * 2500}]},
        )
    extra = r.json()["c1"]
    assert (extra["attack_id"], extra["campaign"]) == ("a-001", "crescendo")
    assert extra["question_truncated"] is True
    assert len(c1.questions[0]) == MAX_QUESTION


def test_models_lists_targets(c1):
    with client_for(c1) as client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["c1", "c1-redteam"]


def test_openai_client_parses_the_response(c1):
    """garak and PyRIT both go through OpenAI-style clients; the official one must accept it."""
    with client_for(c1) as client:
        oai = openai.OpenAI(base_url="http://testserver/v1", api_key=KEY, http_client=client)
        out = oai.chat.completions.create(
            model="c1", messages=[{"role": "user", "content": "Quel préavis ?"}]
        )
    assert out.choices[0].message.content == C1_REPLY["answer"]
