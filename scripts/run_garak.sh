#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Verify adapter endpoint is reachable
if ! curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "Error: adapter is not responding on http://127.0.0.1:8080/health"
    echo "Make sure containers are up: docker compose up -d"
    exit 1
fi

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -z "${ADAPTER_API_KEY:-}" ]; then
    echo "Error: ADAPTER_API_KEY is not set in .env or environment"
    exit 1
fi

export OPENAICOMPATIBLE_API_KEY="${ADAPTER_API_KEY}"

mkdir -p results/garak

# Default spec covers prompt injection, jailbreaks, leakage and agent probes per C3 guide
DEFAULT_SPEC="probes.dan,probes.promptinject,probes.leakreplay"
SPEC="${1:-${DEFAULT_SPEC}}"

echo "==> Target: C1 via OpenAI-compatible adapter (127.0.0.1:8080/v1)"
echo "==> Running garak with spec: ${SPEC}"

# Detect how to invoke garak
if command -v garak &> /dev/null; then
    GARAK_BIN="garak"
elif command -v uv &> /dev/null; then
    GARAK_BIN="uv run garak"
else
    GARAK_BIN="python -m garak"
fi

${GARAK_BIN} --config garak.yaml \
             -m openai.OpenAICompatible \
             -n c1 \
             --spec "${SPEC}" \
             --report_prefix "results/garak/garak_baseline"

echo "==> Generating summary from latest report in results/garak/..."
uv run python scripts/summarize_garak.py results/garak --output results/garak_summary.json

echo "==> Done. Results saved to results/garak_summary.json"
