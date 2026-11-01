"""Entrypoint for the C1 container: tracing to Langfuse without changing C1's code.

Sets up the provider before copilot.api is imported, so C1 keeps it (at the pinned commit it
only installs its own TracerProvider when none exists). Without OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
nothing is instrumented and C1 runs exactly as it does on its own.
"""

import logging
import os

from target import tracing

logger = logging.getLogger("c1")


def patch_app_content_filter() -> None:
    """Handle upstream content filter exceptions from Azure OpenAI as safe refusals."""
    try:
        from copilot.api import app
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.exception_handler(Exception)
        async def handle_content_filter(request: Request, exc: Exception):
            exc_str = str(exc)
            exc_type = type(exc).__name__
            if (
                "contentfilter" in exc_type.lower()
                or "content_filter" in exc_str.lower()
                or "content management policy" in exc_str.lower()
                or "responsibleaipolicyviolation" in exc_str.lower()
            ):
                return JSONResponse(
                    status_code=200,
                    content={
                        "question": "",
                        "answer": "Demande refusée : la requête déclenche les filtres de sécurité de contenu.",
                        "citations": [],
                        "tool_calls": [],
                        "trace_id": "",
                        "content_filtered": True,
                        "refusal_source": "azure_content_filter",
                    },
                )
            logger.exception("Unhandled error in C1: %s (%s)", exc_type, exc_str)
            from fastapi.exception_handlers import http_exception_handler
            from starlette.exceptions import HTTPException as StarletteHTTPException

            if isinstance(exc, StarletteHTTPException):
                return await http_exception_handler(request, exc)
            return JSONResponse(
                status_code=500,
                content={"detail": f"{exc_type}: {exc_str[:500]}" if exc_str else exc_type},
            )
    except ImportError:
        pass


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
    patch_app_content_filter()
    from copilot.api import app

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
