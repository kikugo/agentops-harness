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


def asr_cell(m: dict[str, Any]) -> str:
    """ASR with its counts, e.g. "2.82% (2/71)"; scored excludes errors."""
    asr = m.get("attack_success_rate", 0.0)
    if "scored" not in m:
        return f"{asr:.2%}"
    return f"{asr:.2%} ({m.get('successful_attacks', 0)}/{m['scored']})"


def build_markdown_report(
    baseline: dict[str, Any],
    guarded: dict[str, Any],
    legit_base: dict[str, Any],
    legit_guard: dict[str, Any],
    offline: dict[str, Any] | None = None,
    pii_replay: dict[str, Any] | None = None,
    judge: dict[str, Any] | None = None,
) -> str:
    """Construct the before-and-after tables from structured results."""
    lines: list[str] = []
    rescored = baseline.get("rescored_at") or guarded.get("rescored_at")
    if rescored:
        lines.append(
            f"These tables come from the attack and question runs of {baseline.get('generated_at', '?')[:10]}, "
            f"rescored on {rescored[:10]} with `scripts/rescore_results.py` (no new model calls). "
            "Attacks or questions that ended in an error are reported apart and left out of the rates."
        )
        lines.append("")

    lines.append("### Attack Success Rate Before and After Guardrails")
    lines.append("")
    lines.append("| OWASP Category | Vulnerability | Attacks | Errors (base / guarded) | Baseline ASR | Guarded ASR | Change |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    def err(m: dict[str, Any]) -> str:
        return str(m.get("errors", 0))

    cats = sorted(set(baseline.get("by_category", {}).keys()) | set(guarded.get("by_category", {}).keys()))
    for cat in cats:
        b_cat = baseline.get("by_category", {}).get(cat, {})
        g_cat = guarded.get("by_category", {}).get(cat, {})
        name = CATEGORY_NAMES.get(cat, cat)
        tot = b_cat.get("total", g_cat.get("total", 0))
        b_asr = b_cat.get("attack_success_rate", 0.0)
        g_asr = g_cat.get("attack_success_rate", 0.0)
        lines.append(
            f"| **{cat}** | {name} | {tot} | {err(b_cat)} / {err(g_cat)} | {asr_cell(b_cat)} | "
            f"{asr_cell(g_cat)} | {change_pts(b_asr, g_asr)} |"
        )

    b_o, g_o = baseline.get("overall", {}), guarded.get("overall", {})
    tot_change = change_pts(b_o.get("attack_success_rate", 0.0), g_o.get("attack_success_rate", 0.0))
    lines.append(
        f"| **Total** | **All Categories** | **{baseline.get('total_attacks', 0)}** | "
        f"**{err(b_o)} / {err(g_o)}** | **{asr_cell(b_o)}** | **{asr_cell(g_o)}** | **{tot_change}** |"
    )
    errored = [a["id"] for a in guarded.get("attacks", []) if a.get("outcome") == "error"]
    errored += [a["id"] for a in baseline.get("attacks", []) if a.get("outcome") == "error" and a["id"] not in errored]
    if errored:
        lines.append("")
        lines.append(
            f"Errors: {', '.join(f'`{i}`' for i in errored)}. The judge's own reply was blocked by its "
            "provider's content filter, so the attack was never scored. It is not counted as defended."
            if all("content filter" in (a.get("rationale", "") + " ").lower() or "content filtering" in a.get("rationale", "")
                   for a in guarded.get("attacks", []) + baseline.get("attacks", []) if a.get("id") in errored)
            else f"Errors: {', '.join(f'`{i}`' for i in errored)} could not be scored and are not counted as defended."
        )

    lines.append("")
    lines.append("#### Language Slice Breakdown")
    lines.append("")
    lines.append("| Language | Attacks | Baseline ASR | Guarded ASR | Change |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for lang in ("en", "fr"):
        b_l = baseline.get("by_language", {}).get(lang, {})
        g_l = guarded.get("by_language", {}).get(lang, {})
        label = "English (en)" if lang == "en" else "French (fr)"
        tot = b_l.get("total", g_l.get("total", 0))
        lines.append(
            f"| **{label}** | {tot} | {asr_cell(b_l)} | {asr_cell(g_l)} | "
            f"{change_pts(b_l.get('attack_success_rate', 0.0), g_l.get('attack_success_rate', 0.0))} |"
        )

    if offline:
        g = offline["guarded"]
        lines.append("")
        lines.append("### What the Rails Stop on Their Own")
        lines.append("")
        lines.append(
            "`scripts/offline_gate.py` sends the same attacks through the adapter and rails to a stub agent "
            "that complies with every attack, so the agent's own refusals play no part. "
            f"Without rails every attack succeeds ({offline['unguarded']['attack_success_rate']:.0%}); "
            f"with rails {len(g['let_through'])} of {g['scored']} still get through. "
            "Most of the low guarded ASR above comes from C1 refusing, not from the rails."
        )
        lines.append("")
        lines.append("| OWASP Category | Attacks stopped by the rails alone |")
        lines.append("| :--- | :---: |")
        for cat, asr in g["by_category"].items():
            lines.append(f"| **{cat}** | {1 - asr:.0%} |")

    lines.append("")
    lines.append("### Guardrail Cost and Impact on Legitimate Traffic")
    lines.append("")
    lines.append("| Metric | Without Rails (Baseline) | With Rails (Guarded) | Overhead / Impact |")
    lines.append("| :--- | :---: | :---: | :---: |")

    n_q = legit_base.get("total_evaluated", 100)
    if "errors" in legit_base or "errors" in legit_guard:
        lines.append(f"| Errors ({n_q} Qs, left out of the rates) | {legit_base.get('errors', 0)} | {legit_guard.get('errors', 0)} | |")

    b_frr = legit_base.get("false_refusal_rate", 0.0)
    g_frr = legit_guard.get("false_refusal_rate", 0.0)
    frr_diff = g_frr - b_frr
    frr_diff_str = f"+{frr_diff:.2%}" if frr_diff > 0 else f"{frr_diff:.2%}"
    lines.append(f"| False Refusal Rate (100 Qs) | {b_frr:.2%} | {g_frr:.2%} | {frr_diff_str} |")

    if offline:
        og = offline["golden"]
        lines.append(
            f"| Answerable questions refused by the input rails alone (offline check) | | "
            f"{og['false_refusals']}/{og['answerable']} | |"
        )

    if "answers_changed_by_rails" in legit_guard:
        floor = "" if legit_guard.get("full_answers_stored") else " (at least; only the first 120 characters were stored)"
        lines.append(
            f"| Answers with a PII placeholder (`<PERSON>` etc.){floor} | "
            f"{legit_base.get('answers_with_pii_placeholder', 0)} | {legit_guard.get('answers_with_pii_placeholder', 0)} | "
            f"{', '.join(legit_guard.get('changed_ids', [])) or 'none'} |"
        )

    def end_to_end(d: dict[str, Any]) -> bool:
        return d.get("latency_measured", "end_to_end_client") != "c1_call_only"

    if end_to_end(legit_base) and end_to_end(legit_guard):
        b_p50 = legit_base.get("latency_ms", {}).get("p50", 0.0)
        g_p50 = legit_guard.get("latency_ms", {}).get("p50", 0.0)
        lines.append(f"| Latency p50 | {b_p50:.1f} ms | {g_p50:.1f} ms | {g_p50 - b_p50:+.1f} ms |")
        b_p95 = legit_base.get("latency_ms", {}).get("p95", 0.0)
        g_p95 = legit_guard.get("latency_ms", {}).get("p95", 0.0)
        lines.append(f"| Latency p95 | {b_p95:.1f} ms | {g_p95:.1f} ms | {g_p95 - b_p95:+.1f} ms |")
        rails = legit_guard.get("rails_ms")
        if rails:
            lines.append(f"| Time in rails p50 / p95 | | {rails['p50']:.1f} / {rails['p95']:.1f} ms | |")

    b_tok50 = legit_base.get("tokens", {}).get("p50", 0.0)
    g_tok50 = legit_guard.get("tokens", {}).get("p50", 0.0)
    lines.append(f"| Tokens p50 (estimated, chars / 3.5) | {b_tok50:.0f} | {g_tok50:.0f} | {g_tok50 - b_tok50:+.0f} |")
    b_tok95 = legit_base.get("tokens", {}).get("p95", 0.0)
    g_tok95 = legit_guard.get("tokens", {}).get("p95", 0.0)
    lines.append(f"| Tokens p95 (estimated, chars / 3.5) | {b_tok95:.0f} | {g_tok95:.0f} | {g_tok95 - b_tok95:+.0f} |")

    lines.append("")
    lines.append(
        "A refusal is a keyword match on the answer. The two false-refusal rates come from separate runs of "
        "a non-deterministic agent; the offline row shows what the input rails refuse by themselves."
    )

    if not (end_to_end(legit_base) and end_to_end(legit_guard)):
        lines.append("")
        lines.append(
            "Latency is not reported for these runs: they timed only the C1 call, not the rails, and the two "
            "columns came from separate runs. `scripts/evaluate_legitimate.py` now times each request end to end "
            "and records the time spent in rails; the next run will fill these rows."
        )

    if pii_replay:
        old, new = pii_replay["old"], pii_replay["new"]
        lines.append("")
        lines.append("#### PII Rail Fix")
        lines.append("")
        lines.append(
            "The first PII rail masked every spaCy PERSON hit, and the small French model tags words such as "
            "`n’` and `préavis` as people. The rail now keeps a PERSON only if it looks like a full name, "
            "and skips legal terms and article numbers. `scripts/replay_output_rails.py` replays both versions "
            f"over the {pii_replay['answers']} stored unguarded answers"
            + ("" if pii_replay.get("full_answers") else " (first 120 characters each)")
            + f" and {pii_replay['planted_pii_lines']} lines with planted personal data."
        )
        lines.append("")
        lines.append("| PII rail | Legitimate answers changed | Planted PII lines masked |")
        lines.append("| :--- | :---: | :---: |")
        n, k = pii_replay["answers"], pii_replay["planted_pii_lines"]
        lines.append(f"| Before | {old['answers_changed']}/{n} | {old['planted_masked']}/{k} |")
        lines.append(f"| After | {new['answers_changed']}/{n} | {new['planted_masked']}/{k} |")
        lines.append("")
        lines.append(
            "The guarded rows above still show the old rail, because the campaigns have not been rerun since. "
            "The after rail also misses a surname on its own: a full name needs two capitalised words."
        )

    if judge:
        cm = judge["confusion_matrix"]
        lines.append("")
        lines.append("### Judge Reliability")
        lines.append("")
        lines.append(
            f"I labelled {judge['total_evaluated']} PyRIT campaign rows by hand (`results/manual_check.csv`) and "
            "compared them with the judge's verdicts (`scripts/judge_agreement.py`)."
        )
        lines.append("")
        lines.append("| | Human: breach | Human: no breach |")
        lines.append("| :--- | :---: | :---: |")
        lines.append(f"| **Judge: breach** | {cm['tp']} | {cm['fp']} |")
        lines.append(f"| **Judge: no breach** | {cm['fn']} | {cm['tn']} |")
        lines.append("")
        misses = [d["id"] for d in judge.get("disagreements", []) if d["kind"] == "missed_breach"]
        lines.append(
            f"Raw agreement is {judge['percent_agreement']:.0f}%, but it says little: the judge scored "
            f"{judge.get('judge_positive_rows', cm['tp'] + cm['fp'])} rows as breaches, the sample held "
            f"{judge.get('human_positive_rows', cm['tp'] + cm['fn'])} real breach(es), and the judge missed "
            + (f"{len(misses)} ({', '.join(f'`{m}`' for m in misses)})" if misses else "none")
            + f". Cohen's kappa is {judge['cohen_kappa']:.2f} and recall {judge['recall']:.2f}. "
            "With so few breaches in the sample, these rows cannot show that the judge catches breaches; "
            "the attack success rates above may undercount."
        )

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
    parser.add_argument("--offline", type=Path, default=Path("results/offline/report.json"))
    parser.add_argument("--pii-replay", type=Path, default=Path("results/pii_rail_replay.json"))
    parser.add_argument("--judge", type=Path, default=Path("results/judge_agreement.json"))
    return parser.parse_args(argv)


def load_optional(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline = load_json(args.baseline)
        guarded = load_json(args.guarded)
        legit_base = load_json(args.legit_base)
        legit_guard = load_json(args.legit_guard)

        report_md = build_markdown_report(
            baseline,
            guarded,
            legit_base,
            legit_guard,
            offline=load_optional(args.offline),
            pii_replay=load_optional(args.pii_replay),
            judge=load_optional(args.judge),
        )
        update_readme(args.readme, report_md)
        logger.info(f"Successfully updated {args.readme} with latest results from {args.baseline.parent}/")
        print("\n" + report_md + "\n")
        return 0
    except Exception:
        logger.exception("Failed to generate report")
        return 1


if __name__ == "__main__":
    sys.exit(main())
