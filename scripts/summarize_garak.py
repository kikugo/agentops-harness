"""Parse garak report JSONL and generate summary metrics by probe family.

Extracts eval entries and bootstrap confidence intervals (garak 0.17), computes
pass/fail totals per probe family, and outputs results/garak_summary.json.

Usage:
    uv run python scripts/summarize_garak.py [path_or_dir] [--output results/garak_summary.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def extract_probe_family(probe_name: str) -> str:
    """Extract probe family name from fully qualified probe string.

    Examples:
        'dan.Dan_11_0' -> 'dan'
        'probes.dan.Dan_11_0' -> 'dan'
        'promptinject.PromptInject' -> 'promptinject'
    """
    clean = probe_name.removeprefix("probes.")
    return clean.split(".")[0] if "." in clean else clean


def parse_garak_report(report_path: Path) -> dict:
    """Parse a garak .report.jsonl file into structured summary metrics."""
    if not report_path.is_file():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    metadata: dict = {
        "report_file": str(report_path),
        "garak_version": None,
        "start_time": None,
        "run_id": None,
        "generator": None,
    }

    eval_records: list[dict] = []

    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = record.get("entry_type")
            if entry_type == "init":
                metadata["garak_version"] = record.get("garak_version")
                metadata["start_time"] = record.get("start_time")
                metadata["run_id"] = record.get("run_id")
                metadata["generator"] = record.get("generator")
            elif entry_type == "eval":
                eval_records.append(record)

    families: dict[str, dict] = {}
    total_evaluated = 0
    total_passed = 0
    total_fails = 0
    total_nones = 0

    for ev in eval_records:
        probe = ev.get("probe", "unknown")
        detector = ev.get("detector", "unknown")
        passed = int(ev.get("passed", 0))
        fails = int(ev.get("fails", 0))
        nones = int(ev.get("nones", 0))
        evaluated = int(ev.get("total_evaluated", passed + fails))

        total_evaluated += evaluated
        total_passed += passed
        total_fails += fails
        total_nones += nones

        family_name = extract_probe_family(probe)
        if family_name not in families:
            families[family_name] = {
                "total_evaluated": 0,
                "passed": 0,
                "fails": 0,
                "nones": 0,
                "pass_rate": 0.0,
                "fail_rate": 0.0,
                "probes": [],
            }

        fam = families[family_name]
        fam["total_evaluated"] += evaluated
        fam["passed"] += passed
        fam["fails"] += fails
        fam["nones"] += nones

        probe_entry: dict = {
            "probe": probe,
            "detector": detector,
            "total_evaluated": evaluated,
            "passed": passed,
            "fails": fails,
            "nones": nones,
            "pass_rate": round(passed / evaluated, 4) if evaluated > 0 else 0.0,
            "fail_rate": round(fails / evaluated, 4) if evaluated > 0 else 0.0,
        }

        # garak 0.17 bootstrap confidence interval fields if present
        ci_lower = ev.get("confidence_lower")
        ci_upper = ev.get("confidence_upper")
        if ci_lower is not None and ci_upper is not None:
            probe_entry["confidence_interval"] = {
                "method": ev.get("confidence_method", "bootstrap"),
                "level": ev.get("confidence", 0.95),
                "lower": round(float(ci_lower), 4),
                "upper": round(float(ci_upper), 4),
            }
        else:
            probe_entry["confidence_interval"] = None

        fam["probes"].append(probe_entry)

    # Compute family-level rates
    for fam in families.values():
        fam_evaluated = fam["total_evaluated"]
        if fam_evaluated > 0:
            fam["pass_rate"] = round(fam["passed"] / fam_evaluated, 4)
            fam["fail_rate"] = round(fam["fails"] / fam_evaluated, 4)

    overall_pass_rate = (
        round(total_passed / total_evaluated, 4) if total_evaluated > 0 else 0.0
    )
    overall_fail_rate = (
        round(total_fails / total_evaluated, 4) if total_evaluated > 0 else 0.0
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metadata": metadata,
        "totals": {
            "total_evaluated": total_evaluated,
            "passed": total_passed,
            "fails": total_fails,
            "nones": total_nones,
            "pass_rate": overall_pass_rate,
            "fail_rate": overall_fail_rate,
        },
        "families": dict(sorted(families.items())),
    }


def find_latest_report(dir_or_file: Path) -> Path | None:
    """Find the latest .report.jsonl file if given a directory, or return the path if a file."""
    if dir_or_file.is_file():
        return dir_or_file
    if not dir_or_file.is_dir():
        return None

    reports = sorted(
        dir_or_file.glob("*.report.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize garak benchmark report JSONL.")
    parser.add_argument(
        "target",
        nargs="?",
        default="results/garak",
        help="Path to report JSONL file or directory containing reports (default: results/garak)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/garak_summary.json",
        help="Path to write the summary JSON (default: results/garak_summary.json)",
    )
    args = parser.parse_args(argv)

    target_path = Path(args.target)
    report_file = find_latest_report(target_path)
    if report_file is None:
        print(f"Error: no .report.jsonl found in '{target_path}'", file=sys.stderr)
        return 1

    summary = parse_garak_report(report_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Summary written to {output_path}")
    print(
        f"Evaluated: {summary['totals']['total_evaluated']} | "
        f"Passed: {summary['totals']['passed']} ({summary['totals']['pass_rate']:.1%}) | "
        f"Fails: {summary['totals']['fails']} ({summary['totals']['fail_rate']:.1%})"
    )
    print("Families breakdown:")
    for name, fam in summary["families"].items():
        print(
            f"  - {name}: {fam['passed']}/{fam['total_evaluated']} passed "
            f"({fam['pass_rate']:.1%})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
