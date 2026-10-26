import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DUMMY = {
    "AZURE_SEARCH_ENDPOINT": "https://s.search.windows.net",
    "AZURE_OPENAI_ENDPOINT": "https://o.openai.azure.com",
    "FOUNDRY_PROJECT_ENDPOINT": "https://f.services.ai.azure.com/api/projects/p",
    "AZURE_CLIENT_ID": "id",
    "AZURE_TENANT_ID": "tenant",
    "AZURE_CLIENT_SECRET": "secret",
    "ADAPTER_API_KEY": "key",
}

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")


def compose_config(tmp_path, *args) -> dict:
    env = tmp_path / "test.env"
    env.write_text("".join(f"{k}={v}\n" for k, v in DUMMY.items()))
    out = subprocess.run(
        ["docker", "compose", "--env-file", str(env), *args, "config", "--format", "json"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(out.stdout)


def test_default_services_read_the_demo_index(tmp_path):
    cfg = compose_config(tmp_path)
    assert set(cfg["services"]) == {"c1", "adapter"}
    assert cfg["services"]["c1"]["environment"]["SEARCH_INDEX"] == "articles-v1"


def test_redteam_profile_reads_only_the_redteam_index(tmp_path):
    cfg = compose_config(tmp_path, "--profile", "redteam")
    assert cfg["services"]["c1-redteam"]["environment"]["SEARCH_INDEX"] == "articles-redteam"
    assert cfg["services"]["c1"]["environment"]["SEARCH_INDEX"] == "articles-v1"


def test_ports_bind_to_localhost_and_corpus_is_read_only(tmp_path):
    cfg = compose_config(tmp_path, "--profile", "redteam")
    for svc in cfg["services"].values():
        for port in svc.get("ports", []):
            assert port["host_ip"] == "127.0.0.1"
    for name in ("c1", "c1-redteam"):
        (vol,) = cfg["services"][name]["volumes"]
        assert vol["target"] == "/app/data/articles.jsonl" and vol["read_only"] is True
