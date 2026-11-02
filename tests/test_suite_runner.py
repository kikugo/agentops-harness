"""Unit tests for attack suite parsing, evaluation, metrics aggregation, and runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from suite.config import SuiteConfig
from suite.models import (
    AttackEvaluationResult,
    AttackItem,
)
from suite.run import parse_args
from suite.runner import (
    compute_metrics,
    execute_conversation,
    judge_conversation,
    load_suite,
    run_suite_async,
)


def test_load_sample_suite():
    sample_path = Path("tests/fixtures/sample_suite.yaml")
    items = load_suite(sample_path)
    assert len(items) == 6
    assert items[0].id == "SAMPLE-LLM01"
    assert items[0].owasp_category == "LLM01"
    assert items[0].language == "en"
    assert items[0].target_model == "c1"
    assert items[0].smoke is True
    assert len(items[0].turns) == 1


def test_load_main_suite_integrity():
    main_path = Path("attacks/suite.yaml")
    items = load_suite(main_path)
    assert len(items) == 72

    # Check 6 OWASP categories are each represented by 12 items
    expected_categories = {"LLM01", "LLM02", "LLM06", "LLM07", "LLM08", "LLM09"}
    categories = {i.owasp_category for i in items}
    assert categories == expected_categories

    for cat in expected_categories:
        cat_items = [i for i in items if i.owasp_category == cat]
        assert len(cat_items) == 12

    # Check language breakdown
    fr_items = [i for i in items if i.language == "fr"]
    en_items = [i for i in items if i.language == "en"]
    assert len(fr_items) == 36
    assert len(en_items) == 36

    # Check smoke subset
    smoke_items = [i for i in items if i.smoke]
    assert len(smoke_items) == 30

    # Check LLM08 targets c1-redteam
    llm08_items = [i for i in items if i.owasp_category == "LLM08"]
    assert all(i.target_model == "c1-redteam" for i in llm08_items)


def test_compute_metrics_empty():
    summary = compute_metrics([], suite_file="empty.yaml")
    assert summary.total_attacks == 0
    assert summary.overall.total == 0
    assert summary.overall.attack_success_rate == 0.0
    assert summary.by_category == {}
    assert summary.by_language == {}


def test_compute_metrics_calculation():
    results = [
        AttackEvaluationResult(
            id="atk-1",
            owasp_category="LLM01",
            category_name="Prompt Injection",
            language="en",
            target_model="c1",
            smoke=True,
            turns=["attack 1"],
            score=1.0,
            achieved=True,
            rationale="Injection succeeded",
            response_preview="bad response",
        ),
        AttackEvaluationResult(
            id="atk-2",
            owasp_category="LLM01",
            category_name="Prompt Injection",
            language="fr",
            target_model="c1",
            smoke=True,
            turns=["attack 2"],
            score=0.0,
            achieved=False,
            rationale="Injection refused",
            response_preview="refusal",
        ),
        AttackEvaluationResult(
            id="atk-3",
            owasp_category="LLM07",
            category_name="System Prompt Leakage",
            language="en",
            target_model="c1",
            smoke=False,
            turns=["attack 3"],
            score=0.0,
            achieved=False,
            rationale="Prompt preserved",
            response_preview="refusal",
        ),
    ]

    summary = compute_metrics(results, suite_file="dummy.yaml")
    assert summary.total_attacks == 3
    assert summary.overall.total == 3
    assert summary.overall.successful_attacks == 1
    assert summary.overall.defended_attacks == 2
    assert summary.overall.attack_success_rate == pytest.approx(0.3333, rel=1e-3)

    assert summary.by_category["LLM01"].total == 2
    assert summary.by_category["LLM01"].successful_attacks == 1
    assert summary.by_category["LLM01"].attack_success_rate == 0.5

    assert summary.by_category["LLM07"].total == 1
    assert summary.by_category["LLM07"].successful_attacks == 0
    assert summary.by_category["LLM07"].attack_success_rate == 0.0

    assert summary.by_language["en"].total == 2
    assert summary.by_language["en"].successful_attacks == 1
    assert summary.by_language["en"].attack_success_rate == 0.5

    assert summary.by_language["fr"].total == 1
    assert summary.by_language["fr"].successful_attacks == 0
    assert summary.by_language["fr"].attack_success_rate == 0.0


def test_execute_conversation_mock():
    import asyncio

    config = SuiteConfig.from_env()
    item = AttackItem(
        id="mock-1",
        owasp_category="LLM01",
        category_name="Prompt Injection",
        language="en",
        target_model="c1",
        smoke=True,
        turns=["turn 1", "turn 2"],
        expected_safe_behavior="refuse",
    )

    async def mock_requester(model: str, messages: list[dict[str, str]], headers: dict[str, str]):
        return f"echo: {messages[-1]['content']}"

    conv = asyncio.run(execute_conversation(config, item, custom_requester=mock_requester))
    assert len(conv) == 4
    assert conv[0] == {"role": "user", "content": "turn 1"}
    assert conv[1] == {"role": "assistant", "content": "echo: turn 1"}
    assert conv[2] == {"role": "user", "content": "turn 2"}
    assert conv[3] == {"role": "assistant", "content": "echo: turn 2"}


def test_judge_conversation_mock():
    import asyncio

    config = SuiteConfig.from_env()
    item = AttackItem(
        id="mock-1",
        owasp_category="LLM01",
        category_name="Prompt Injection",
        language="en",
        target_model="c1",
        smoke=True,
        turns=["turn 1"],
        expected_safe_behavior="refuse",
    )
    conversation = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "I refuse this request."},
    ]

    async def mock_judge(it: AttackItem, c: list[dict[str, str]]):
        return 0.0, False, "Successfully defended against injection."

    score, achieved, rationale = asyncio.run(
        judge_conversation(config, item, conversation, custom_judge=mock_judge)
    )
    assert score == 0.0
    assert achieved is False
    assert "defended" in rationale


def test_run_suite_async_end_to_end_mock(tmp_path):
    import asyncio

    sample_suite = Path("tests/fixtures/sample_suite.yaml")
    out_file = tmp_path / "baseline.json"
    config = SuiteConfig.from_env(suite_path=sample_suite, results_path=out_file)

    async def mock_requester(model: str, messages: list[dict[str, str]], headers: dict[str, str]):
        return "Demande refusée : hors périmètre."

    async def mock_judge(it: AttackItem, c: list[dict[str, str]]):
        # Let SAMPLE-LLM06 be breached for testing
        if it.id == "SAMPLE-LLM06":
            return 1.0, True, "Executed command"
        return 0.0, False, "Safely defended"

    summary = asyncio.run(
        run_suite_async(
            config=config,
            filter_smoke=False,
            output_path=out_file,
            custom_requester=mock_requester,
            custom_judge=mock_judge,
        )
    )

    assert out_file.exists()
    assert summary.total_attacks == 6
    assert summary.overall.successful_attacks == 1
    assert summary.overall.defended_attacks == 5
    assert summary.overall.attack_success_rate == pytest.approx(1 / 6, rel=1e-3)

    # Test smoke filter
    smoke_summary = asyncio.run(
        run_suite_async(
            config=config,
            filter_smoke=True,
            output_path=out_file,
            custom_requester=mock_requester,
            custom_judge=mock_judge,
        )
    )
    assert smoke_summary.total_attacks == 3


def test_parse_cli_args():
    args = parse_args(["--suite", "attacks/suite.yaml", "--smoke", "--category", "LLM01", "--limit", "5"])
    assert args.suite == Path("attacks/suite.yaml")
    assert args.smoke is True
    assert args.category == "LLM01"
    assert args.limit == 5
