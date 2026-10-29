"""Send one question through the adapter and check its trace landed in Langfuse.

Costs one C1 question (a few thousand gpt-5.4-mini tokens). Writes results/trace_check.json.
Usage: uv run python scripts/check_trace.py
"""

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ADAPTER = "http://127.0.0.1:8080"
LANGFUSE = "http://127.0.0.1:3000"
# needs search_code, so the trace has a retrieval span as well as the agent, model and tool
QUESTION = "Combien de jours de congés payés un salarié acquiert-il par mois de travail ?"
CAMPAIGN = "phase1-smoke"
OUT = Path("results/trace_check.json")

# what must be in the trace -> how to recognise it (Langfuse observation type or name prefix)
EXPECTED = {
    "adapter root": lambda o: o["name"].startswith("attack "),
    "C1 request": lambda o: o["name"] == "POST /ask",
    "agent run": lambda o: o["type"] == "AGENT",
    "generation": lambda o: o["type"] == "GENERATION",
    "tool call": lambda o: o["type"] == "TOOL",
    "retriever": lambda o: o["type"] == "RETRIEVER",
}
# FastAPI's own spans in the adapter start before the attack baggage is set
UNTAGGED_SCOPES = {"fastapi"}


def ask(attack_id: str) -> dict:
    headers = {
        "Authorization": f"Bearer {os.environ['ADAPTER_API_KEY']}",
        "X-Attack-Id": attack_id,
        "X-Campaign": CAMPAIGN,
    }
    resp = httpx.post(
        f"{ADAPTER}/v1/chat/completions",
        headers=headers,
        json={"model": "c1", "messages": [{"role": "user", "content": QUESTION}]},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_observations(trace_id: str, tries: int = 30) -> list[dict]:
    """Langfuse v4 serves traces through the v2 observations API (the legacy one is gone)."""
    auth = (os.environ["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"],
            os.environ["LANGFUSE_INIT_PROJECT_SECRET_KEY"])  # fmt: skip
    params = {"traceId": trace_id, "fields": "core,basic,metadata", "limit": 100}
    seen = 0
    for _ in range(tries):
        r = httpx.get(f"{LANGFUSE}/api/public/v2/observations", params=params, auth=auth,
                      timeout=30)  # fmt: skip
        data = r.json().get("data", []) if r.status_code == 200 else []
        if data and len(data) == seen:  # stopped growing: the batch exporters have flushed
            return data
        seen = len(data)
        time.sleep(5)
    raise SystemExit(f"trace {trace_id} incomplete in Langfuse after {tries * 5}s")


def check(observations: list[dict], attack_id: str, campaign: str) -> dict:
    found = {label: [o["name"] for o in observations if match(o)]
             for label, match in EXPECTED.items()}  # fmt: skip
    tagged = [o for o in observations
              if (o.get("metadata") or {}).get("scope.name") not in UNTAGGED_SCOPES]  # fmt: skip
    untagged = [o["name"] for o in tagged
                if (o.get("metadata") or {}).get("attack_id") != attack_id
                or o.get("sessionId") != campaign]  # fmt: skip
    missing = [label for label, hits in found.items() if not hits]
    return {
        "passed": not missing and not untagged,
        "missing": missing,
        "spans_without_attack_tags": untagged,
        "span_counts": {label: len(hits) for label, hits in found.items()},
        "observations": sorted(f"{o['type']} {o['name']}" for o in observations),
    }


def main() -> int:
    load_dotenv(".env")
    attack_id = f"trace-smoke-{int(time.time())}"
    out = ask(attack_id)
    trace_id = out["c1"]["trace_id"]
    result = {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "attack_id": attack_id,
        "campaign": CAMPAIGN,
        "trace_id": trace_id,
        "tools_called": [c["name"] for c in out["c1"]["tool_calls"]],
        **check(fetch_observations(trace_id), attack_id, CAMPAIGN),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "missing", "span_counts")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
