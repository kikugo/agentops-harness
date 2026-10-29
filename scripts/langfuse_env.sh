#!/usr/bin/env bash
# Appends the local Langfuse secrets and project keys to .env. Prints names, never values.
# Safe to rerun: a variable already in .env is left alone.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE=.env
touch "$ENV_FILE"

rand() { openssl rand -hex "${1:-24}"; }
has() { grep -q "^$1=." "$ENV_FILE"; }
add() {
  if has "$1"; then echo "kept  $1"; else
    sed -i.bak "/^$1=$/d" "$ENV_FILE" && rm -f "$ENV_FILE.bak"
    printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"; echo "added $1"
  fi
}
get() { grep "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2-; }

add POSTGRES_PASSWORD "$(rand)"
add DATABASE_URL "postgresql://postgres:$(get POSTGRES_PASSWORD)@postgres:5432/postgres"
add SALT "$(rand)"
add ENCRYPTION_KEY "$(rand 32)"
add NEXTAUTH_SECRET "$(rand)"
add CLICKHOUSE_PASSWORD "$(rand)"
add REDIS_AUTH "$(rand)"
add MINIO_ROOT_PASSWORD "$(rand)"
for k in EVENT_UPLOAD MEDIA_UPLOAD BATCH_EXPORT; do
  add "LANGFUSE_S3_${k}_SECRET_ACCESS_KEY" "$(get MINIO_ROOT_PASSWORD)"
done
add TELEMETRY_ENABLED false

# Headless init: org, project and API keys exist on first start, no sign-up needed.
add LANGFUSE_INIT_ORG_ID agentops
add LANGFUSE_INIT_ORG_NAME agentops
add LANGFUSE_INIT_PROJECT_ID agentops
add LANGFUSE_INIT_PROJECT_NAME agentops-harness
add LANGFUSE_INIT_PROJECT_PUBLIC_KEY "pk-lf-$(rand 16)"
add LANGFUSE_INIT_PROJECT_SECRET_KEY "sk-lf-$(rand 16)"
add LANGFUSE_INIT_USER_EMAIL admin@agentops.localhost
add LANGFUSE_INIT_USER_NAME admin
add LANGFUSE_INIT_USER_PASSWORD "$(rand 12)"

# OTLP auth for the C1 and adapter exporters (space in "Basic ..." is %20 per the OTEL spec)
auth=$(printf '%s:%s' "$(get LANGFUSE_INIT_PROJECT_PUBLIC_KEY)" "$(get LANGFUSE_INIT_PROJECT_SECRET_KEY)" | base64 | tr -d '\n')
add OTEL_EXPORTER_OTLP_TRACES_HEADERS "Authorization=Basic%20${auth},x-langfuse-ingestion-version=4"
