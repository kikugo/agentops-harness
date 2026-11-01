"""Sample 50 scored conversation attempts into results/manual_check.csv for manual labeling."""

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("sample_manual_check")

CSV_FIELDS = [
    "id",
    "campaign",
    "prompt_preview",
    "response_preview",
    "judge_score",
    "judge_rationale",
    "human_label",
]


def _clean_preview(text: str, max_len: int = 160) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


def extract_item(record: dict[str, Any]) -> dict[str, Any]:
    turns = record.get("turns", [])
    user_turns = [t.get("content", "") for t in turns if t.get("role") == "user"]
    assistant_turns = [t.get("content", "") for t in turns if t.get("role") == "assistant"]

    prompt_preview = _clean_preview(user_turns[-1] if user_turns else "")
    response_preview = _clean_preview(assistant_turns[-1] if assistant_turns else "")

    score_val = record.get("score")
    if score_val is not None:
        judge_score = 1 if float(score_val) >= 0.5 else 0
    else:
        judge_score = 1 if record.get("achieved") else 0

    return {
        "id": record.get("id") or record.get("attack_id", "unknown"),
        "campaign": record.get("campaign", "unknown"),
        "prompt_preview": prompt_preview,
        "response_preview": response_preview,
        "judge_score": judge_score,
        "judge_rationale": _clean_preview(record.get("rationale", ""), max_len=200),
        "human_label": "",  # Strictly empty: for Karthik to fill manually
    }


def sample_attempts(
    input_sources: list[Path | str],
    output_csv: Path | str,
    sample_size: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for source in input_sources:
        path = Path(source)
        if not path.exists():
            continue
        if path.is_dir():
            for f in sorted(path.glob("*.jsonl")):
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
        else:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in provided input sources: {input_sources}")

    # Deduplicate by id
    unique_by_id: dict[str, dict[str, Any]] = {}
    for r in records:
        item = extract_item(r)
        unique_by_id[item["id"]] = item
    pool = list(unique_by_id.values())

    if len(pool) <= sample_size:
        sampled = pool
    else:
        # Stratified sampling across (campaign, judge_score)
        strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for item in pool:
            strata[(item["campaign"], item["judge_score"])].append(item)

        num_strata = len(strata)
        quota_per_stratum = max(1, sample_size // num_strata)

        sampled = []
        leftovers = []
        for items in strata.values():
            shuffled = list(items)
            rng.shuffle(shuffled)
            take = min(quota_per_stratum, len(shuffled))
            sampled.extend(shuffled[:take])
            leftovers.extend(shuffled[take:])

        remaining = sample_size - len(sampled)
        if remaining > 0 and leftovers:
            rng.shuffle(leftovers)
            sampled.extend(leftovers[:remaining])

    rng.shuffle(sampled)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sampled)

    logger.info(f"Sampled {len(sampled)} items into {out_path}")
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample scored PyRIT attempts into CSV for human evaluation"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["results/pyrit/"],
        help="Input JSONL files or directories (default: results/pyrit/)",
    )
    parser.add_argument(
        "--output",
        default="results/manual_check.csv",
        help="Output CSV path (default: results/manual_check.csv)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=50, help="Number of items to sample (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sample_attempts(
        input_sources=args.input,
        output_csv=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
