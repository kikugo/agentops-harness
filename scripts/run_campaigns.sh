#!/usr/bin/env bash
set -euo pipefail

# Scripts directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Running PyRIT Red-Teaming Campaigns against C1"

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Ensure output directory exists
mkdir -p results/pyrit

# Check that adapter is reachable
if ! curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "ERROR: Adapter endpoint (http://127.0.0.1:8080) is not answering."
    echo "Make sure docker compose is running: docker compose up -d"
    exit 1
fi

echo "==> Step 1: Executing campaigns (crescendo, system_prompt, indirect_injection, tool_abuse)..."
uv run python -m campaigns.run_all "$@"

echo "==> Step 2: Sampling 50 attempts into results/manual_check.csv for manual labeling..."
uv run python -m scripts.sample_manual_check \
    --input results/pyrit/ \
    --output results/manual_check.csv \
    --sample-size 50

echo ""
echo "=========================================================================="
echo "Campaigns finished!"
echo "Next step: Open 'results/manual_check.csv' and fill the 'human_label' column"
echo "with 1 (attack succeeded / breach) or 0 (defended / refusal) for the 50 rows."
echo ""
echo "Once labeled, compute the agreement metrics by running:"
echo "  uv run python -m scripts.judge_agreement"
echo "=========================================================================="
