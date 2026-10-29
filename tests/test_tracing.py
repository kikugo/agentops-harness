import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry import baggage, propagate, trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from target import c1_otel, tracing
from target.adapter import create_app

KEY = "k"


@pytest.fixture(scope="module")
def exporter():
    """One global provider for the module (OTel allows setting it once per process)."""
    exp = InMemorySpanExporter()
    provider = tracing.build_provider("test", exp)
    trace.set_tracer_provider(provider)
    # build_provider batches; flush before reading
    exp.flush = provider.force_flush
    return exp


@pytest.fixture
def spans(exporter):
    exporter.clear()

    def read():
        exporter.flush()
        return exporter.get_finished_spans()

    return read


def test_attack_attributes():
    attrs = tracing.attack_attributes("a-7", "crescendo", "c1")
    assert attrs == {
        "langfuse.trace.name": "attack a-7",
        "langfuse.trace.metadata.target": "c1",
        "langfuse.trace.metadata.attack_id": "a-7",
        "langfuse.trace.metadata.campaign": "crescendo",
        "langfuse.session.id": "crescendo",
    }
    assert tracing.attack_attributes("", "", "c1") == {
        "langfuse.trace.name": "request",
        "langfuse.trace.metadata.target": "c1",
    }


def test_setup_is_a_noop_without_endpoint(monkeypatch):
    monkeypatch.delenv(tracing.ENDPOINT_VAR, raising=False)
    assert tracing.setup("x") is False
    assert c1_otel.instrument() is False


def test_adapter_root_span_and_propagated_context(spans):
    seen = {}

    def fake_c1(request: httpx.Request) -> httpx.Response:
        ctx = propagate.extract(dict(request.headers))
        seen["trace_id"] = trace.get_current_span(ctx).get_span_context().trace_id
        seen["attack_id"] = baggage.get_baggage("langfuse.trace.metadata.attack_id", ctx)
        seen["session"] = baggage.get_baggage("langfuse.session.id", ctx)
        return httpx.Response(200, json={"answer": "Réponse.", "citations": [],
                                         "tool_calls": [], "trace_id": "x"})  # fmt: skip

    app = create_app(targets={"c1": "http://c1:8000"}, api_key=KEY,
                     transport=httpx.MockTransport(fake_c1))  # fmt: skip
    with TestClient(app, headers={"Authorization": f"Bearer {KEY}"}) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"X-Attack-Id": "a-7", "X-Campaign": "crescendo"},
            json={"model": "c1", "messages": [{"role": "user", "content": "Quel préavis ?"}]},
        )
    assert r.status_code == 200
    (root,) = [s for s in spans() if s.name == "attack a-7"]
    assert root.attributes["langfuse.trace.metadata.attack_id"] == "a-7"
    assert root.attributes["langfuse.session.id"] == "crescendo"
    assert tuple(root.attributes["langfuse.trace.tags"]) == ("crescendo", "c1")
    assert root.attributes["langfuse.observation.input"] == "Quel préavis ?"
    assert root.attributes["langfuse.observation.output"] == "Réponse."
    # C1 receives the same trace and the attack attributes as baggage
    assert seen == {"trace_id": root.context.trace_id, "attack_id": "a-7", "session": "crescendo"}


def test_baggage_is_copied_to_child_spans_and_search_is_a_retriever(spans):
    tracer = trace.get_tracer("t")
    attrs = tracing.attack_attributes("a-8", "tool-abuse", "c1")
    names = ("_SearchClientOperationsMixin.search", "_SearchClientOperationsMixin._search_post",
             "execute_tool search_code")  # fmt: skip
    with tracing.baggage_scope(attrs), tracer.start_as_current_span("root"):
        for name in names:
            with tracer.start_as_current_span(name):
                pass
    by_name = {s.name: s for s in spans()}
    for name in names:
        assert by_name[name].attributes["langfuse.trace.metadata.attack_id"] == "a-8"
        assert by_name[name].attributes["langfuse.session.id"] == "tool-abuse"
    types = {n: by_name[n].attributes.get("langfuse.observation.type") for n in names}
    assert types == dict(zip(names, ("retriever", None, None), strict=True))


def test_failed_c1_call_marks_the_root_span_as_error(spans):
    app = create_app(targets={"c1": "http://c1:8000"}, api_key=KEY,
                     transport=httpx.MockTransport(lambda r: httpx.Response(500)))  # fmt: skip
    with TestClient(app, headers={"Authorization": f"Bearer {KEY}"}) as client:
        client.post("/v1/chat/completions",
                    json={"model": "c1", "messages": [{"role": "user", "content": "abc"}]})  # fmt: skip
    (root,) = [s for s in spans() if s.name == "request"]
    assert root.status.status_code is trace.StatusCode.ERROR
