"""Unit tests for legitimate questions evaluation and report generator."""

from __future__ import annotations

from pathlib import Path

from scripts.evaluate_legitimate import estimate_tokens, is_refusal_response, load_golden_questions
from scripts.generate_report import build_markdown_report, change_pts, update_readme


def test_is_refusal_response():
    assert is_refusal_response("Demande refusée : la requête déclenche les filtres de sécurité.") is True
    assert is_refusal_response("Les sources dont je dispose ne couvrent pas cette question.") is True
    assert is_refusal_response("Je ne peux pas répondre à cette demande.") is True
    assert is_refusal_response("Selon l'article L1234-1, la durée du préavis est de 2 mois.") is False


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("Bonjour") == 2
    assert estimate_tokens("Une phrase un peu plus longue en français.") > 5


def test_load_golden_questions():
    golden_path = Path("vendor/droit-du-travail-copilot/eval/golden.jsonl")
    questions = load_golden_questions(golden_path)
    assert len(questions) == 100
    assert "question" in questions[0]
    assert "id" in questions[0]


def test_build_markdown_report():
    baseline = {
        "total_attacks": 72,
        "overall": {"attack_success_rate": 0.0139},
        "by_category": {
            "LLM01": {"total": 12, "attack_success_rate": 0.0},
            "LLM09": {"total": 12, "attack_success_rate": 0.0833},
        },
        "by_language": {
            "en": {"total": 36, "attack_success_rate": 0.0},
            "fr": {"total": 36, "attack_success_rate": 0.0278},
        },
    }
    guarded = {
        "total_attacks": 72,
        "overall": {"attack_success_rate": 0.0},
        "by_category": {
            "LLM01": {"total": 12, "attack_success_rate": 0.0},
            "LLM09": {"total": 12, "attack_success_rate": 0.0},
        },
        "by_language": {
            "en": {"total": 36, "attack_success_rate": 0.0},
            "fr": {"total": 36, "attack_success_rate": 0.0},
        },
    }
    legit_base = {
        "false_refusal_rate": 0.02,
        "latency_ms": {"p50": 850.0, "p95": 1450.0},
        "tokens": {"p50": 210, "p95": 420},
    }
    legit_guard = {
        "false_refusal_rate": 0.02,
        "latency_ms": {"p50": 890.0, "p95": 1520.0},
        "tokens": {"p50": 210, "p95": 420},
    }

    report = build_markdown_report(baseline, guarded, legit_base, legit_guard)
    assert "Attack Success Rate Before and After Guardrails" in report
    assert "LLM09" in report
    assert "False Refusal Rate (100 Qs)" in report
    assert "850.0 ms" in report
    assert "890.0 ms" in report


def test_update_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Test\n\nIntro.\n", encoding="utf-8")

    report_text = "### Results Table\nData here."
    update_readme(readme, report_text)

    content = readme.read_text(encoding="utf-8")
    assert "<!-- RESULTS:START -->" in content
    assert "<!-- RESULTS:END -->" in content
    assert "### Results Table" in content

    # Test idempotence on update
    new_report = "### Updated Table\nNew data."
    update_readme(readme, new_report)
    updated_content = readme.read_text(encoding="utf-8")
    assert "### Updated Table" in updated_content
    assert "Data here" not in updated_content


def test_change_pts_signs_increases_and_decreases():
    assert change_pts(0.0, 0.0833) == "+8.33 pts"
    assert change_pts(0.0278, 0.0) == "-2.78 pts"
    assert change_pts(0.05, 0.05) == "0.00 pts"
