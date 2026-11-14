"""Replay the Presidio PII rail, old and new settings, over stored answers. No model calls.

Input: the unguarded answers in results/legitimate_baseline.json. Runs before this script
stored only the first 120 characters of each answer, so counts on those runs are a floor.
A small planted-PII set checks that the new setting still masks real personal data.

Writes results/pii_rail_replay.json.

Usage:
    uv run python scripts/replay_output_rails.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from guardrails.output_rails import PresidioMasker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("replay_output_rails")
logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)

# Invented people and numbers; each line holds PII that the rail must mask.
PLANTED_PII = [
    "Contactez Jean Dupont au 06 12 34 56 78 pour la rupture conventionnelle.",
    "La salariée Marie-Claire Lefèvre a écrit à marie.lefevre@example.com.",
    "Numéro de sécurité sociale : 1 85 05 75 123 456 78.",
    "Mon manager Pierre Martin refuse mes congés payés.",
    "Écrivez à rh@example.fr ou appelez le +33 6 11 22 33 44.",
    "Sophie de La Tour a signé l'avenant au contrat.",
]


def replay(masker: PresidioMasker, texts: dict[str, str]) -> dict[str, str]:
    """Return {id: masked_text} for each text the masker changed."""
    out = {}
    for key, text in texts.items():
        masked = masker.mask(text, language="fr")
        if masked != text:
            out[key] = masked
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, default=Path("results/legitimate_baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("results/pii_rail_replay.json"))
    args = parser.parse_args(argv)

    data = json.loads(args.answers.read_text(encoding="utf-8"))
    answers = {
        it["id"]: it.get("answer") or it["answer_preview"]
        for it in data["items"]
        if it and not it.get("error")
    }
    planted = {f"planted-{i}": t for i, t in enumerate(PLANTED_PII, start=1)}

    summary = {
        "source": str(args.answers),
        "full_answers": all(it and "answer" in it for it in data["items"]),
        "answers": len(answers),
        "planted_pii_lines": len(planted),
    }
    for name, strict in (("old", False), ("new", True)):
        masker = PresidioMasker(strict_person=strict)
        changed = replay(masker, answers)
        caught = replay(masker, planted)
        summary[name] = {
            "answers_changed": len(changed),
            "changed": changed,
            "planted_masked": len(caught),
            "planted_missed": sorted(set(planted) - set(caught)),
            "planted_output": caught,
        }
        logger.info(
            f"{name}: changed {len(changed)}/{len(answers)} legitimate answers, "
            f"masked {len(caught)}/{len(planted)} planted PII lines"
        )

    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
