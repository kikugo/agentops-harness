import csv
from pathlib import Path

import pytest

from scripts.judge_agreement import compute_agreement, parse_label
from scripts.sample_manual_check import sample_attempts


def test_parse_label():
    assert parse_label("1") == 1
    assert parse_label(1) == 1
    assert parse_label("true") == 1
    assert parse_label("TRUE") == 1
    assert parse_label("yes") == 1
    assert parse_label("0") == 0
    assert parse_label(0) == 0
    assert parse_label("false") == 0
    assert parse_label("no") == 0
    assert parse_label("") is None
    assert parse_label(None) is None


def test_sample_attempts_from_jsonl(tmp_path):
    fixture_jsonl = Path("tests/fixtures/sample_campaign_conversations.jsonl")
    out_csv = tmp_path / "manual_check.csv"

    sampled = sample_attempts(
        input_sources=[fixture_jsonl],
        output_csv=out_csv,
        sample_size=10,
        seed=42,
    )

    assert len(sampled) == 10
    assert out_csv.exists()

    with open(out_csv, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 10
        for row in reader:
            assert row["id"]
            assert row["campaign"]
            assert row["prompt_preview"]
            assert row["response_preview"]
            assert row["judge_score"] in ("0", "1", "0.0", "1.0")
            assert row["human_label"] == ""  # MUST be empty for Karthik to fill


def test_compute_agreement_on_fixture():
    fixture_csv = Path("tests/fixtures/sample_manual_check.csv")
    metrics = compute_agreement(fixture_csv)

    assert metrics["total_evaluated"] == 20
    assert metrics["agreed_count"] == 18
    assert metrics["percent_agreement"] == pytest.approx(90.0, rel=1e-3)
    assert metrics["cohen_kappa"] == pytest.approx(0.7917, rel=1e-3)

    cm = metrics["confusion_matrix"]
    assert cm["tp"] == 7
    assert cm["tn"] == 11
    assert cm["fp"] == 1
    assert cm["fn"] == 1

    assert metrics["precision"] == pytest.approx(0.875, rel=1e-3)
    assert metrics["recall"] == pytest.approx(0.875, rel=1e-3)
    assert metrics["f1_score"] == pytest.approx(0.875, rel=1e-3)

    # Per-campaign breakdown
    by_campaign = metrics["by_campaign"]
    assert "crescendo" in by_campaign
    assert "system_prompt_extraction" in by_campaign
    assert "indirect_prompt_injection" in by_campaign
    assert "tool_abuse" in by_campaign


def test_compute_agreement_perfect_agreement(tmp_path):
    csv_file = tmp_path / "perfect.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "campaign",
                "prompt_preview",
                "response_preview",
                "judge_score",
                "judge_rationale",
                "human_label",
            ]
        )
        writer.writerow(["1", "crescendo", "p", "r", "1", "rationale", "1"])
        writer.writerow(["2", "crescendo", "p", "r", "0", "rationale", "0"])

    metrics = compute_agreement(csv_file)
    assert metrics["percent_agreement"] == 100.0
    assert metrics["cohen_kappa"] == 1.0


def test_compute_agreement_empty_human_labels(tmp_path):
    csv_file = tmp_path / "empty.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "campaign",
                "prompt_preview",
                "response_preview",
                "judge_score",
                "judge_rationale",
                "human_label",
            ]
        )
        writer.writerow(["1", "crescendo", "p", "r", "1", "rationale", ""])

    with pytest.raises(ValueError, match="No labeled rows found"):
        compute_agreement(csv_file)
