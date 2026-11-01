"""Configuration for PyRIT campaigns targeting C1 and C1-redteam."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _normalize_azure_endpoint(url: str) -> str:
    url = url.rstrip("/")
    if ".openai.azure.com" in url and not url.endswith("/openai/v1"):
        return f"{url}/openai/v1"
    return url


@dataclass
class CampaignConfig:
    adapter_url: str
    adapter_api_key: str
    results_dir: Path
    memory_db_path: Path
    judge_endpoint: str
    judge_api_key: str
    judge_model: str
    attacker_endpoint: str
    attacker_api_key: str
    attacker_model: str

    @classmethod
    def from_env(cls) -> "CampaignConfig":
        adapter_url = os.getenv("ADAPTER_URL", "http://127.0.0.1:8080/v1")
        adapter_api_key = os.getenv("ADAPTER_API_KEY", "")

        results_dir = Path(os.getenv("RESULTS_DIR", "results/pyrit"))
        memory_db_path = Path(os.getenv("MEMORY_DB_PATH", str(results_dir / "pyrit_memory.db")))

        judge_endpoint = (
            os.getenv("JUDGE_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("OPENROUTER_URL")
            or adapter_url
        )
        judge_endpoint = _normalize_azure_endpoint(judge_endpoint)
        judge_api_key = (
            os.getenv("JUDGE_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or adapter_api_key
        )
        judge_model = os.getenv("JUDGE_MODEL") or os.getenv("CHAT_DEPLOYMENT", "c1")

        attacker_endpoint = (
            os.getenv("ATTACKER_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("OPENROUTER_URL")
            or adapter_url
        )
        attacker_endpoint = _normalize_azure_endpoint(attacker_endpoint)
        attacker_api_key = (
            os.getenv("ATTACKER_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or adapter_api_key
        )
        attacker_model = os.getenv("ATTACKER_MODEL") or os.getenv("CHAT_DEPLOYMENT", "c1")

        return cls(
            adapter_url=adapter_url,
            adapter_api_key=adapter_api_key,
            results_dir=results_dir,
            memory_db_path=memory_db_path,
            judge_endpoint=judge_endpoint,
            judge_api_key=judge_api_key,
            judge_model=judge_model,
            attacker_endpoint=attacker_endpoint,
            attacker_api_key=attacker_api_key,
            attacker_model=attacker_model,
        )
