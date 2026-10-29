"""Entrypoint for the C1 container: tracing to Langfuse without changing C1's code.

Sets up the provider before copilot.api is imported, so C1 keeps it (at the pinned commit it
only installs its own TracerProvider when none exists). Without OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
nothing is instrumented and C1 runs exactly as it does on its own.
"""

import os

from target import tracing


def instrument() -> bool:
    if not tracing.setup(os.environ.get("OTEL_SERVICE_NAME", "c1")):
        return False
    from azure.core.settings import settings
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    FastAPIInstrumentor().instrument()  # reads the adapter's traceparent and baggage
    HTTPXClientInstrumentor().instrument()  # Foundry and embedding calls
    settings.tracing_implementation = "opentelemetry"  # Azure AI Search queries

    # invoke_agent, chat and execute_tool spans; prompts and tool args stay in local Langfuse
    from agent_framework.observability import enable_instrumentation

    enable_instrumentation(enable_sensitive_data=os.environ.get("ENABLE_SENSITIVE_DATA") == "true")
    return True


def main() -> None:
    import uvicorn

    instrument()
    uvicorn.run("copilot.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
