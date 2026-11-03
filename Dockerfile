# OpenAI-compatible adapter (target/adapter.py)
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY target ./target
COPY guardrails ./guardrails

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8080
CMD ["uvicorn", "target.adapter:app", "--host", "0.0.0.0", "--port", "8080"]
