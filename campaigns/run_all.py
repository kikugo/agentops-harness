"""Orchestrator to run PyRIT red-teaming campaigns against C1."""

import argparse
import asyncio
import logging

from campaigns.config import CampaignConfig
from campaigns.crescendo import run_crescendo_campaign
from campaigns.indirect_injection import run_indirect_injection_campaign
from campaigns.system_prompt import run_system_prompt_campaign
from campaigns.tool_abuse import run_tool_abuse_campaign

logger = logging.getLogger("campaigns")


async def run_all_campaigns(
    config: CampaignConfig,
    campaign_name: str = "all",
    limit: int | None = None,
) -> dict[str, list[dict]]:
    results = {}

    campaign_map = {
        "crescendo": run_crescendo_campaign,
        "system_prompt": run_system_prompt_campaign,
        "indirect_injection": run_indirect_injection_campaign,
        "tool_abuse": run_tool_abuse_campaign,
    }

    if campaign_name != "all" and campaign_name not in campaign_map:
        raise ValueError(
            f"Unknown campaign '{campaign_name}'. Allowed: all, {', '.join(campaign_map.keys())}"
        )

    targets = campaign_map.keys() if campaign_name == "all" else [campaign_name]

    for name in targets:
        logger.info(f"=== Starting Campaign: {name} ===")
        fn = campaign_map[name]
        try:
            res = await fn(config=config, limit=limit)
            results[name] = res
            success_count = sum(1 for r in res if r.get("achieved"))
            logger.info(f"=== Completed {name}: {len(res)} attempts, {success_count} breaches ===")
        except Exception as e:
            logger.exception(f"Error running campaign {name}")
            results[name] = [{"error": str(e)}]

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyRIT red-teaming campaigns against C1")
    parser.add_argument(
        "--campaign",
        choices=["all", "crescendo", "system_prompt", "indirect_injection", "tool_abuse"],
        default="all",
        help="Which campaign to run (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of attempts per campaign"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = CampaignConfig.from_env()
    results = asyncio.run(
        run_all_campaigns(config=config, campaign_name=args.campaign, limit=args.limit)
    )

    print("\n================ CAMPAIGN SUMMARY ================")
    for c_name, c_res in results.items():
        total = len(c_res)
        breaches = sum(1 for r in c_res if r.get("achieved"))
        rate = (breaches / total * 100) if total else 0.0
        print(f"Campaign: {c_name:<25} Total: {total:<4} Breaches: {breaches:<4} ({rate:.1f}%)")
    print("==================================================\n")


if __name__ == "__main__":
    main()
