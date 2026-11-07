"""Generate Markdown before/after comparison tables in README.md from results/*.json.

Rereads results/baseline.json, results/guarded.json, results/legitimate_baseline.json,
and results/legitimate_guarded.json to populate README.md between:
<!-- RESULTS:START --> and <!-- RESULTS:END -->.

Usage:
    uv run python scripts/generate_report.py [--readme README.md]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("generate_report")

CATEGORY_NAMES = {
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage",
    "LLM08": "Vector & Embedding Weaknesses",
    "LLM09": "Misinformation & Tool Misuse",
}

def change_pts(before: float, after: float) -> str:
    """Signed change from before to after in percentage points; an increase shows as +."""
    pts = (after - before) * 100
    return "0.00 pts" if abs(pts) < 0.005 else f"{pts:+.2f} pts"


START_TAG = "<!-- RESULTS:START -->"
END_TAG = "<!-- RESULTS:END -->"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_markdown_report(
    baseline: dict[str, Any],
    guarded: dict[str, Any],
    legit_base: dict[str, Any],
    legit_guard: dict[str, Any],
) -> str:
    """Construct the before-and-after tables from structured results."""
    lines: list[str] = []

    lines.append("### Attack Success Rate Before and After Guardrails")
    lines.append("")
    lines.append("| OWASP Category | Vulnerability | Attacks | Baseline ASR | Guarded ASR | Change |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")

    cats = sorted(set(baseline.get("by_category", {}).keys()) | set(guarded.get("by_category", {}).keys()))
    for cat in cats:
        b_cat = baseline.get("by_category", {}).get(cat, {})
        g_cat = guarded.get("by_category", {}).get(cat, {})
        name = CATEGORY_NAMES.get(cat, cat)
        tot = b_cat.get("total", g_cat.get("total", 0))

        b_asr = b_cat.get("attack_success_rate", 0.0)
        g_asr = g_cat.get("attack_success_rate", 0.0)
        lines.append(f"| **{cat}** | {name} | {tot} | {b_asr:.2%} | {g_asr:.2%} | {change_pts(b_asr, g_asr)} |")

    # Overall line
    b_tot_asr = baseline.get("overall", {}).get("attack_success_rate", 0.0)
    g_tot_asr = guarded.get("overall", {}).get("attack_success_rate", 0.0)
    tot_attacks = baseline.get("total_attacks", 0)
    tot_change = change_pts(b_tot_asr, g_tot_asr)
    lines.append(f"| **Total** | **All Categories** | **{tot_attacks}** | **{b_tot_asr:.2%}** | **{g_tot_asr:.2%}** | **{tot_change}** |")

    lines.append("")
    lines.append("#### Language Slice Breakdown")
    lines.append("")
    lines.append("| Language | Attacks | Baseline ASR | Guarded ASR | Change |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for lang in ("en", "fr"):
        b_l = baseline.get("by_language", {}).get(lang, {})
        g_l = guarded.get("by_language", {}).get(lang, {})
        b_asr = b_l.get("attack_success_rate", 0.0)
        g_asr = g_l.get("attack_success_rate", 0.0)
        label = "English (en)" if lang == "en" else "French (fr)"
        tot = b_l.get("total", g_l.get("total", 0))
        lines.append(f"| **{label}** | {tot} | {b_asr:.2%} | {g_asr:.2%} | {change_pts(b_asr, g_asr)} |")

    lines.append("")
    lines.append("### Guardrail Cost and Impact on Legitimate Traffic")
    lines.append("")
    lines.append("| Metric | Without Rails (Baseline) | With Rails (Guarded) | Overhead / Impact |")
    lines.append("| :--- | :---: | :---: | :---: |")

    # False refusal rate
    b_frr = legit_base.get("false_refusal_rate", 0.0)
    g_frr = legit_guard.get("false_refusal_rate", 0.0)
    frr_diff = g_frr - b_frr
    frr_diff_str = f"+{frr_diff:.2%}" if frr_diff > 0 else f"{frr_diff:.2%}"
    lines.append(f"| False Refusal Rate (100 Qs) | {b_frr:.2%} | {g_frr:.2%} | {frr_diff_str} |")

    # Latencies
    b_p50 = legit_base.get("latency_ms", {}).get("p50", 0.0)
    g_p50 = legit_guard.get("latency_ms", {}).get("p50", 0.0)
    p50_diff = g_p50 - b_p50
    lines.append(f"| Latency p50 | {b_p50:.1f} ms | {g_p50:.1f} ms | {p50_diff:+.1f} ms |")

    b_p95 = legit_base.get("latency_ms", {}).get("p95", 0.0)
    g_p95 = legit_guard.get("latency_ms", {}).get("p95", 0.0)
    p95_diff = g_p95 - b_p95
    lines.append(f"| Latency p95 | {b_p95:.1f} ms | {g_p95:.1f} ms | {p95_diff:+.1f} ms |")

    # Tokens
    b_tok50 = legit_base.get("tokens", {}).get("p50", 0.0)
    g_tok50 = legit_guard.get("tokens", {}).get("p50", 0.0)
    lines.append(f"| Tokens p50 | {b_tok50:.0f} | {g_tok50:.0f} | {g_tok50 - b_tok50:+.0f} |")

    b_tok95 = legit_base.get("tokens", {}).get("p95", 0.0)
    g_tok95 = legit_guard.get("tokens", {}).get("p95", 0.0)
    lines.append(f"| Tokens p95 | {b_tok95:.0f} | {g_tok95:.0f} | {g_tok95 - b_tok95:+.0f} |")

    return "\n".join(lines)


def update_readme(
    readme_path: Path,
    report_markdown: str,
) -> bool:
    """Insert or update the generated tables between START and END markers in README.md."""
    if not readme_path.is_file():
        raise FileNotFoundError(f"README not found at: {readme_path}")

    content = readme_path.read_text(encoding="utf-8")
    block = f"{START_TAG}\n{report_markdown}\n{END_TAG}"

    if START_TAG in content and END_TAG in content:
        before = content.split(START_TAG)[0]
        after = content.split(END_TAG)[1]
        new_content = f"{before}{block}{after}"
    else:
        # Append to the end of README if markers don't exist yet
        new_content = f"{content.rstrip()}\n\n## Results\n\n{block}\n"

    readme_path.write_text(new_content, encoding="utf-8")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate before/after tables in README from results.")
    parser.add_argument("--readme", type=Path, default=Path("README.md"), help="Path to README.md")
    parser.add_argument("--baseline", type=Path, default=Path("results/baseline.json"), help="Baseline attacks JSON")
    parser.add_argument("--guarded", type=Path, default=Path("results/guarded.json"), help="Guarded attacks JSON")
    parser.add_argument(
        "--legit-base",
        type=Path,
        default=Path("results/legitimate_baseline.json"),
        help="Legitimate baseline JSON",
    )
    parser.add_argument(
        "--legit-guard",
        type=Path,
        default=Path("results/legitimate_guarded.json"),
        help="Legitimate guarded JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline = load_json(args.baseline)
        guarded = load_json(args.guarded)
        legit_base = load_json(args.legit_base)
        legit_guard = load_json(args.legit_guard)

        report_md = build_markdown_report(baseline, guarded, legit_base, legit_guard)
        update_readme(args.readme, report_md)
        logger.info(f"Successfully updated {args.readme} with latest results from {args.baseline.parent}/")
        print("\n" + report_md + "\n")
        return 0
    except Exception:
        logger.exception("Failed to generate report")
        return 1


if __name__ == "__main__":
    sys.exit(main())
