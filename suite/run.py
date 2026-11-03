"""CLI entry point for running the attack suite against C1 and evaluating baseline ASR."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from suite.config import SuiteConfig
from suite.runner import run_suite_async

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("suite.run")


def print_summary_table(summary) -> None:
    """Print a clean Markdown-like summary table to stdout."""
    print("\n" + "=" * 70)
    print("ATTACK SUITE EVALUATION BASELINE REPORT")
    print("=" * 70)
    print(f"Generated at: {summary.generated_at}")
    print(f"Suite file:   {summary.suite_file}")
    print(f"Total attacks evaluated: {summary.total_attacks}")
    print(f"Overall Attack Success Rate (ASR): {summary.overall.attack_success_rate:.2%}")
    print(f"Successful attacks: {summary.overall.successful_attacks} / {summary.overall.total}")
    print("-" * 70)
    print("BREAKDOWN BY OWASP CATEGORY:")
    print(f"{'Category':<10} | {'Total':<6} | {'Passed/Defended':<16} | {'Breached/Succ':<14} | {'ASR':<8}")
    print("-" * 70)
    for cat, m in summary.by_category.items():
        print(
            f"{cat:<10} | {m.total:<6} | {m.defended_attacks:<16} | "
            f"{m.successful_attacks:<14} | {m.attack_success_rate:.2%}"
        )
    print("-" * 70)
    print("BREAKDOWN BY LANGUAGE:")
    print(f"{'Language':<10} | {'Total':<6} | {'Passed/Defended':<16} | {'Breached/Succ':<14} | {'ASR':<8}")
    print("-" * 70)
    for lang, m in summary.by_language.items():
        print(
            f"{lang:<10} | {m.total:<6} | {m.defended_attacks:<16} | "
            f"{m.successful_attacks:<14} | {m.attack_success_rate:.2%}"
        )
    print("=" * 70 + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attack suite against C1 adapter and compute baseline ASR."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("attacks/suite.yaml"),
        help="Path to attacks/suite.yaml file (default: attacks/suite.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline.json"),
        help="Path to write results JSON (default: results/baseline.json)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only run the 30-attack smoke subset (for CI gate testing)",
    )
    parser.add_argument(
        "--guarded",
        action="store_true",
        help="Run attacks against guarded model target (e.g. c1-guarded)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter attacks by OWASP category (e.g. LLM01, LLM07)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit execution to the first N attack items",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SuiteConfig.from_env(
        suite_path=args.suite, results_path=args.output, guarded=args.guarded
    )

    try:
        summary = asyncio.run(
            run_suite_async(
                config=config,
                filter_smoke=args.smoke,
                filter_category=args.category,
                limit=args.limit,
                output_path=args.output,
            )
        )
        print_summary_table(summary)
        return 0
    except Exception:
        logger.exception("Suite run failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
