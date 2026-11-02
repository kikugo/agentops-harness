"""Configuration for attack suite execution and judging."""

from __future__ import annotations

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
class SuiteConfig:
    adapter_url: str
    adapter_api_key: str
    judge_endpoint: str
    judge_api_key: str
    judge_model: str
    suite_path: Path
    results_path: Path
    memory_db_path: Path

    @classmethod
    def from_env(
        cls,
        suite_path: Path | str | None = None,
        results_path: Path | str | None = None,
        memory_db_path: Path | str | None = None,
    ) -> SuiteConfig:
        adapter_url = os.getenv("ADAPTER_URL", "http://127.0.0.1:8080/v1")
        adapter_api_key = os.getenv("ADAPTER_API_KEY", "")

        judge_endpoint = (
            os.getenv("JUDGE_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("OPENROUTER_URL")
            or adapter_url
        )
        judge_endpoint = _normalize_azure_endpoint(judge_endpoint)
        is_azure_judge = "azure" in judge_endpoint.lower()
        judge_api_key = (
            os.getenv("JUDGE_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or ("" if is_azure_judge else adapter_api_key)
        )
        judge_model = os.getenv("JUDGE_MODEL") or os.getenv("CHAT_DEPLOYMENT", "c1")

        s_path = Path(suite_path or os.getenv("SUITE_PATH", "attacks/suite.yaml"))
        r_path = Path(results_path or os.getenv("RESULTS_BASELINE_PATH", "results/baseline.json"))
        m_path = Path(memory_db_path or os.getenv("MEMORY_DB_PATH", "/tmp/pyrit_data/suite_memory.db"))

        return cls(
            adapter_url=adapter_url,
            adapter_api_key=adapter_api_key,
            judge_endpoint=judge_endpoint,
            judge_api_key=judge_api_key,
            judge_model=judge_model,
            suite_path=s_path,
            results_path=r_path,
            memory_db_path=m_path,
        )
