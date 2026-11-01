import json

from campaigns.common import create_target, record_conversation
from campaigns.config import CampaignConfig


def test_campaign_config_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTER_API_KEY", "test-adapter-key")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    cfg = CampaignConfig.from_env()

    assert cfg.adapter_url == "http://127.0.0.1:8080/v1"
    assert cfg.adapter_api_key == "test-adapter-key"
    assert cfg.results_dir == tmp_path / "results"
    assert cfg.memory_db_path == tmp_path / "results" / "pyrit_memory.db"


def test_create_target_configures_headers(monkeypatch):
    monkeypatch.setenv("ADAPTER_API_KEY", "secret-key")
    cfg = CampaignConfig.from_env()

    target = create_target(
        config=cfg,
        model="c1",
        campaign="crescendo",
        attack_id="atk-12345",
    )

    assert target._model_name == "c1"
    assert target._endpoint == "http://127.0.0.1:8080/v1"
    assert target._headers.get("X-Attack-Id") == "atk-12345"
    assert target._headers.get("X-Campaign") == "crescendo"
    assert target._headers.get("Authorization") == "Bearer secret-key"


def test_record_conversation_writes_jsonl(tmp_path):
    output_file = tmp_path / "test_campaign.jsonl"
    turns = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
    ]
    record_conversation(
        filepath=output_file,
        attack_id="atk-001",
        campaign="crescendo",
        model="c1",
        turns=turns,
        score=1.0,
        rationale="Agent provided out of domain advice",
        achieved=True,
    )

    assert output_file.exists()
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["attack_id"] == "atk-001"
    assert record["campaign"] == "crescendo"
    assert record["model"] == "c1"
    assert record["turns"] == turns
    assert record["score"] == 1.0
    assert record["rationale"] == "Agent provided out of domain advice"
    assert record["achieved"] is True
    assert "timestamp" in record
