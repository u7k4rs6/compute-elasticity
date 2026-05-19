"""Phase 3 — Extraction rate check.

Runs 20 Qwen2.5-7B-Instruct-Turbo completions on GPQA Diamond problems that are
NOT in the locked 50-problem sample, applies passes 1-4, and reports the
unparseable rate.

Pass criterion: extraction rate ≥ 95% (≤1 unparseable out of 20).

If rate < 95%, this is the only phase where prompt iteration is permitted.
Document any template change in preregistration.md and re-tag.

Usage:
    source .venv/bin/activate
    python scripts/extraction_rate_check.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_N_COMPLETIONS = 20
_MIN_EXTRACTION_RATE = 0.95


def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def _load_locked_ids() -> set[str]:
    ids_path = ROOT / "data" / "problem_ids.json"
    if not ids_path.exists():
        return set()
    return set(json.loads(ids_path.read_text()))


async def run_extraction_check() -> bool:
    """Sample 20 non-locked problems and check extraction rate; return True if ≥95%."""
    from pilot.data_loader import load_gpqa_diamond
    from pilot.prompts import render_prompt
    from pilot.sampling import TogetherClient
    from pilot.scoring import extract_answer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    locked_ids = _load_locked_ids()
    logger.info("Loaded %d locked problem IDs to exclude.", len(locked_ids))

    logger.info("Loading GPQA Diamond…")
    all_problems = load_gpqa_diamond()
    holdout = [p for p in all_problems if p.id not in locked_ids]
    logger.info("%d problems available for extraction check.", len(holdout))

    if len(holdout) < _N_COMPLETIONS:
        logger.error(
            "Not enough holdout problems (%d < %d).", len(holdout), _N_COMPLETIONS
        )
        return False

    # Take first N problems deterministically
    check_problems = holdout[:_N_COMPLETIONS]

    try:
        client = TogetherClient()
    except Exception as exc:
        logger.error("Failed to init TogetherClient: %s", exc)
        return False

    n_parseable = 0
    n_total = 0

    for i, problem in enumerate(check_problems, start=1):
        prompt = render_prompt(
            question_text=problem.question,
            option_a=problem.option_a,
            option_b=problem.option_b,
            option_c=problem.option_c,
            option_d=problem.option_d,
        )
        seed_hex = f"{i:016x}"
        try:
            completion = await client.complete(
                prompt=prompt,
                temperature=0.7,
                seed_hex=seed_hex,
            )
        except Exception as exc:
            logger.error("API error on problem %s: %s — skipping", problem.id, exc)
            continue

        n_total += 1
        result = extract_answer(completion.text)
        parsed = result.answer is not None
        if parsed:
            n_parseable += 1

        logger.info(
            "Problem %d/%d [%s]  pass=%s  answer=%s  parsed=%s",
            i,
            _N_COMPLETIONS,
            problem.id,
            result.pass_number,
            result.answer,
            parsed,
        )

    if n_total == 0:
        logger.error("No completions collected.")
        return False

    rate = n_parseable / n_total
    print("\n=== Extraction Rate Report ===")
    print(f"  Completions:  {n_total}")
    print(f"  Parseable:    {n_parseable}")
    print(f"  Unparseable:  {n_total - n_parseable}")
    print(f"  Rate:         {rate:.1%}")
    print(f"  Threshold:    {_MIN_EXTRACTION_RATE:.0%}")
    print(f"  Result:       {'PASS' if rate >= _MIN_EXTRACTION_RATE else 'FAIL'}")

    if rate < _MIN_EXTRACTION_RATE:
        logger.error(
            "Extraction rate %.1f%% < %.0f%%. "
            "Iterate the prompt template, document in preregistration.md, and re-tag.",
            rate * 100,
            _MIN_EXTRACTION_RATE * 100,
        )
        return False

    return True


def main() -> None:
    _load_env()
    ok = asyncio.run(run_extraction_check())
    if ok:
        print("\nExtraction rate check PASSED.")
        sys.exit(0)
    else:
        print("\nExtraction rate check FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
