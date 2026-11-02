"""Calculate agreement and Cohen's kappa between automated judge and human evaluator."""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("judge_agreement")


def parse_label(val: Any) -> int | None:
    if val is None:
        return None
    val_str = str(val).strip().lower()
    if not val_str:
        return None
    if val_str in ("1", "1.0", "true", "yes", "breach", "success"):
        return 1
    if val_str in ("0", "0.0", "false", "no", "defended", "failure", "safe"):
        return 0
    raise ValueError(f"Unrecognized binary label value: '{val}'")


def compute_agreement(csv_path: Path | str) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manual check CSV not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    evaluated_pairs: list[tuple[str, int, int]] = []  # (campaign, judge, human)
    unlabeled_count = 0

    for r in rows:
        h_label = parse_label(r.get("human_label"))
        if h_label is None:
            unlabeled_count += 1
            continue
        j_label = parse_label(r.get("judge_score"))
        if j_label is None:
            continue
        campaign = r.get("campaign", "unknown")
        evaluated_pairs.append((campaign, j_label, h_label))

    total = len(evaluated_pairs)
    if total == 0:
        raise ValueError(
            f"No labeled rows found in {path}. "
            "Karthik must fill in human_label (1 or 0) for manual check attempts first."
        )

    if unlabeled_count > 0:
        logger.warning(
            f"Skipped {unlabeled_count} rows with blank human_label. Evaluated {total} labeled rows."
        )

    tp = sum(1 for _, j, h in evaluated_pairs if j == 1 and h == 1)
    tn = sum(1 for _, j, h in evaluated_pairs if j == 0 and h == 0)
    fp = sum(1 for _, j, h in evaluated_pairs if j == 1 and h == 0)
    fn = sum(1 for _, j, h in evaluated_pairs if j == 0 and h == 1)

    agreed = tp + tn
    p_o = agreed / total

    p_j1 = (tp + fp) / total
    p_j0 = (tn + fn) / total
    p_h1 = (tp + fn) / total
    p_h0 = (tn + fp) / total

    p_e = (p_j1 * p_h1) + (p_j0 * p_h0)

    if abs(1.0 - p_e) < 1e-9:
        kappa = 1.0 if abs(p_o - 1.0) < 1e-9 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Per-campaign breakdown
    by_campaign: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "agreed": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    )
    for c, j, h in evaluated_pairs:
        by_campaign[c]["total"] += 1
        if j == h:
            by_campaign[c]["agreed"] += 1
        if j == 1 and h == 1:
            by_campaign[c]["tp"] += 1
        elif j == 0 and h == 0:
            by_campaign[c]["tn"] += 1
        elif j == 1 and h == 0:
            by_campaign[c]["fp"] += 1
        elif j == 0 and h == 1:
            by_campaign[c]["fn"] += 1

    campaign_summary = {}
    for c, stats in by_campaign.items():
        c_tot = stats["total"]
        campaign_summary[c] = {
            "total": c_tot,
            "agreed": stats["agreed"],
            "percent_agreement": round((stats["agreed"] / c_tot) * 100, 2) if c_tot else 0.0,
            "tp": stats["tp"],
            "tn": stats["tn"],
            "fp": stats["fp"],
            "fn": stats["fn"],
        }

    return {
        "total_evaluated": total,
        "unlabeled_count": unlabeled_count,
        "agreed_count": agreed,
        "percent_agreement": round(p_o * 100, 2),
        "cohen_kappa": round(kappa, 4),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "by_campaign": dict(campaign_summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agreement between judge and human labels"
    )
    parser.add_argument(
        "--input",
        default="results/manual_check.csv",
        help="Path to manual check CSV (default: results/manual_check.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/judge_agreement.json",
        help="Path to output JSON summary (default: results/judge_agreement.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    metrics = compute_agreement(args.input)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n================ JUDGE AGREEMENT METRICS ================")
    print(f"Total Evaluated : {metrics['total_evaluated']}")
    print(f"Agreed Count    : {metrics['agreed_count']} / {metrics['total_evaluated']}")
    print(f"Agreement Rate  : {metrics['percent_agreement']}%")
    print(f"Cohen's Kappa   : {metrics['cohen_kappa']}")
    print(f"Precision       : {metrics['precision']}")
    print(f"Recall          : {metrics['recall']}")
    print(f"F1 Score        : {metrics['f1_score']}")
    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  TP: {cm['tp']:<4} FP: {cm['fp']:<4}")
    print(f"  FN: {cm['fn']:<4} TN: {cm['tn']:<4}")
    print("\nPer Campaign:")
    for c, stats in metrics["by_campaign"].items():
        print(f"  - {c:<25}: {stats['agreed']}/{stats['total']} ({stats['percent_agreement']}%)")
    print("=========================================================\n")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
