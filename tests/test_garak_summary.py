import json
import time
from pathlib import Path

import pytest

from scripts.summarize_garak import (
    extract_probe_family,
    find_latest_report,
    main,
    parse_garak_report,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "garak_sample.report.jsonl"


def test_extract_probe_family():
    assert extract_probe_family("dan.Dan_11_0") == "dan"
    assert extract_probe_family("probes.dan.Dan_11_0") == "dan"
    assert extract_probe_family("promptinject.PromptInject") == "promptinject"
    assert extract_probe_family("simpleprobe") == "simpleprobe"


def test_parse_garak_sample_report():
    summary = parse_garak_report(FIXTURE_PATH)

    # Metadata
    meta = summary["metadata"]
    assert meta["garak_version"] == "0.17.1"
    assert meta["generator"] == "openai.OpenAICompatible"
    assert meta["run_id"] == "test-run-123"

    # Totals
    totals = summary["totals"]
    assert totals["total_evaluated"] == 130
    assert totals["passed"] == 123
    assert totals["fails"] == 7
    assert totals["nones"] == 0
    assert totals["pass_rate"] == round(123 / 130, 4)
    assert totals["fail_rate"] == round(7 / 130, 4)

    # Families
    families = summary["families"]
    assert set(families.keys()) == {"dan", "leakreplay", "promptinject"}

    # dan family has 2 probes (Dan_11_0 and Apostrophe)
    dan = families["dan"]
    assert dan["total_evaluated"] == 80
    assert dan["passed"] == 75
    assert dan["fails"] == 5
    assert dan["pass_rate"] == 0.9375
    assert dan["fail_rate"] == 0.0625
    assert len(dan["probes"]) == 2

    # Probe 1: dan.Dan_11_0 with bootstrap confidence interval
    dan_p1 = next(p for p in dan["probes"] if p["probe"] == "dan.Dan_11_0")
    assert dan_p1["passed"] == 45
    assert dan_p1["fails"] == 5
    assert dan_p1["confidence_interval"] == {
        "method": "bootstrap",
        "level": 0.95,
        "lower": 0.04,
        "upper": 0.18,
    }

    # Probe 2: dan.Apostrophe with 0 fails
    dan_p2 = next(p for p in dan["probes"] if p["probe"] == "dan.Apostrophe")
    assert dan_p2["passed"] == 30
    assert dan_p2["fails"] == 0
    assert dan_p2["confidence_interval"]["lower"] == 0.0

    # promptinject family
    pi = families["promptinject"]
    assert pi["total_evaluated"] == 30
    assert pi["passed"] == 28
    assert pi["fails"] == 2
    assert pi["pass_rate"] == round(28 / 30, 4)

    # leakreplay family without confidence intervals
    leak = families["leakreplay"]
    assert leak["total_evaluated"] == 20
    assert leak["passed"] == 20
    assert leak["fails"] == 0
    assert leak["probes"][0]["confidence_interval"] is None


def test_parse_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_garak_report(Path("nonexistent.report.jsonl"))


def test_find_latest_report(tmp_path):
    # Returns path if direct file
    assert find_latest_report(FIXTURE_PATH) == FIXTURE_PATH

    # Returns None if nonexistent
    assert find_latest_report(tmp_path / "does_not_exist") is None

    # Empty dir returns None
    assert find_latest_report(tmp_path) is None

    # Picks newest by mtime
    f1 = tmp_path / "old.report.jsonl"
    f2 = tmp_path / "new.report.jsonl"
    f1.write_text("{}\n")
    time.sleep(0.05)
    f2.write_text("{}\n")

    assert find_latest_report(tmp_path) == f2


def test_main_cli_execution(tmp_path):
    out_file = tmp_path / "summary.json"
    exit_code = main([str(FIXTURE_PATH), "--output", str(out_file)])
    assert exit_code == 0
    assert out_file.is_file()

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["totals"]["total_evaluated"] == 130
    assert "dan" in loaded["families"]
