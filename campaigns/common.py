"""Common utilities, target factories, and logging for PyRIT campaigns."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pyrit.memory import CentralMemory, SQLiteMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import SelfAskTrueFalseScorer
from pyrit.score.true_false.self_ask_true_false_scorer import (
    TrueFalseQuestion,
    render_true_false_system_prompt,
)

from campaigns.config import CampaignConfig


def init_memory(db_path: Path | str | None = None) -> SQLiteMemory:
    """Initialize CentralMemory with an SQLiteMemory backend if not already set."""
    try:
        mem = CentralMemory.get_memory_instance()
        if isinstance(mem, SQLiteMemory):
            return mem
    except ValueError:
        pass

    if db_path is None or db_path == ":memory:":
        target_path = ":memory:"
    else:
        path_obj = Path(db_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        target_path = str(path_obj)

    memory_instance = SQLiteMemory(db_path=target_path)
    CentralMemory.set_memory_instance(memory_instance)
    return memory_instance


def create_target(
    config: CampaignConfig,
    model: str = "c1",
    campaign: str = "",
    attack_id: str = "",
) -> OpenAIChatTarget:
    """Create an OpenAIChatTarget pointing to the adapter with trace headers."""
    init_memory(config.memory_db_path)

    headers = {
        "X-Attack-Id": attack_id,
        "X-Campaign": campaign,
    }
    if config.adapter_api_key:
        headers["Authorization"] = f"Bearer {config.adapter_api_key}"

    return OpenAIChatTarget(
        model_name=model,
        endpoint=config.adapter_url,
        api_key=config.adapter_api_key or "local-adapter",
        headers=json.dumps(headers),
    )


def create_judge_target(config: CampaignConfig) -> OpenAIChatTarget:
    """Create an OpenAIChatTarget pointing to the judge LLM."""
    init_memory(config.memory_db_path)
    kwargs = {}
    if config.judge_api_key:
        kwargs["api_key"] = config.judge_api_key
    return OpenAIChatTarget(
        model_name=config.judge_model,
        endpoint=config.judge_endpoint,
        **kwargs,
    )


def create_judge_scorer(
    chat_target: OpenAIChatTarget,
    category: str,
    true_description: str,
    false_description: str,
) -> SelfAskTrueFalseScorer:
    """Instantiate a SelfAskTrueFalseScorer using the specified true/false descriptions."""
    question = TrueFalseQuestion(
        category=category,
        true_description=true_description,
        false_description=false_description,
    )
    rendered = render_true_false_system_prompt(question=question)
    return SelfAskTrueFalseScorer(
        chat_target=chat_target,
        system_prompt=rendered.value,
        question=question,
    )


def extract_turns_from_memory(conversation_id: str) -> list[dict[str, str]]:
    """Retrieve formatted conversation turns (role, content) from CentralMemory."""
    memory = CentralMemory.get_memory_instance()
    messages = memory.get_conversation_messages(conversation_id=conversation_id)
    return [{"role": m.api_role, "content": m.get_value()} for m in messages]


def record_conversation(
    filepath: Path | str,
    attack_id: str,
    campaign: str,
    model: str,
    turns: list[dict[str, Any]],
    score: float | None = None,
    rationale: str = "",
    achieved: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a full conversation run record to a JSONL file."""
    record_path = Path(filepath)
    record_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "id": attack_id,
        "attack_id": attack_id,
        "campaign": campaign,
        "model": model,
        "timestamp": datetime.now(UTC).isoformat(),
        "turns": turns,
        "score": score,
        "rationale": rationale,
        "achieved": achieved,
        "metadata": metadata or {},
    }

    with open(record_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
