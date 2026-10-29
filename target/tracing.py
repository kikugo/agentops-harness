"""OTLP tracing to Langfuse, shared by the adapter and the C1 entrypoint.

Off unless OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is set, so tests and plain runs export nothing.
Docs: langfuse.com/integrations/native/opentelemetry (HTTP only, Basic auth in
OTEL_EXPORTER_OTLP_TRACES_HEADERS, langfuse.* attributes, baggage on every span).
"""

import os
from contextlib import contextmanager

from opentelemetry import baggage, context, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

ENDPOINT_VAR = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
def langfuse_keys(key: str) -> bool:
    return key.startswith("langfuse.")


def is_search_query(name: str) -> bool:
    """azure-search-documents 12 names its query span `_SearchClientOperationsMixin.search`
    (the HTTP call below it is `..._search_post`)."""
    return "SearchClient" in name and name.endswith(".search")


class RetrieverTagger(SpanProcessor):
    """Marks Azure AI Search query spans as Langfuse retriever observations."""

    def on_start(self, span, parent_context=None) -> None:
        if is_search_query(span.name):
            span.set_attribute("langfuse.observation.type", "retriever")


def build_provider(service_name: str, exporter: SpanExporter) -> TracerProvider:
    from opentelemetry.processor.baggage import BaggageSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    # Order matters: copy baggage and tag spans before the batch processor exports them.
    provider.add_span_processor(BaggageSpanProcessor(langfuse_keys))
    provider.add_span_processor(RetrieverTagger())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def setup(service_name: str, exporter: SpanExporter | None = None) -> bool:
    """Install a global provider exporting to Langfuse. Returns False (no-op) when unconfigured."""
    if exporter is None:
        if not os.environ.get(ENDPOINT_VAR):
            return False
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()  # endpoint and headers from the OTEL_* env vars
    trace.set_tracer_provider(build_provider(service_name, exporter))
    return True


def attack_attributes(attack_id: str, campaign: str, target: str) -> dict:
    """Trace-level Langfuse attributes for one adapter request."""
    attrs = {
        "langfuse.trace.name": f"attack {attack_id}" if attack_id else "request",
        "langfuse.trace.metadata.target": target,
    }
    if attack_id:
        attrs["langfuse.trace.metadata.attack_id"] = attack_id
    if campaign:
        attrs["langfuse.trace.metadata.campaign"] = campaign
        attrs["langfuse.session.id"] = campaign
    return attrs


@contextmanager
def baggage_scope(attrs: dict):
    """Make `attrs` current baggage: copied to every span below and sent on to C1."""
    ctx = context.get_current()
    for key, value in attrs.items():
        ctx = baggage.set_baggage(key, value, ctx)
    token = context.attach(ctx)
    try:
        yield
    finally:
        context.detach(token)
