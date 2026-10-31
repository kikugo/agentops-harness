#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Verify adapter endpoint is reachable and returning adapter health payload
HEALTH_RESP="$(curl -sf http://127.0.0.1:8080/health 2>/dev/null || true)"
if [[ "${HEALTH_RESP}" != *'"status":"ok"'* && "${HEALTH_RESP}" != *'"status": "ok"'* ]]; then
    echo "Error: adapter is not responding on http://127.0.0.1:8080/health with expected status."
    echo "Current response: ${HEALTH_RESP:0:100}"
    echo "Make sure the adapter container is running on port 8080."
    echo "If port 8080 is occupied by another service (e.g. reel-history), stop it first."
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

GENERATIONS="${GENERATIONS:-1}"

echo "==> Target: C1 via OpenAI-compatible adapter (127.0.0.1:8080/v1)"
echo "==> Running garak with spec: ${SPEC} (generations: ${GENERATIONS})"

# Detect how to invoke garak (garak 0.17.1 runs isolated to avoid openai<3 conflict)
if command -v uvx &> /dev/null; then
    GARAK_BIN="uvx --from garak==0.17.1 garak"
elif command -v garak &> /dev/null; then
    GARAK_BIN="garak"
elif [ -x ".venv/bin/garak" ]; then
    GARAK_BIN="uv run garak"
else
    echo "Error: uvx or garak 0.17.1 is required to run garak without dependency conflicts."
    exit 1
fi

${GARAK_BIN} --config garak.yaml \
             --target_type openai.OpenAICompatible \
             --target_name c1 \
             --generations "${GENERATIONS}" \
             --spec "${SPEC}" \
             --report_prefix "${REPO_ROOT}/results/garak/garak_baseline"

echo "==> Generating summary from latest report in results/garak/..."
uv run python scripts/summarize_garak.py results/garak --output results/garak_summary.json

echo "==> Done. Results saved to results/garak_summary.json"
