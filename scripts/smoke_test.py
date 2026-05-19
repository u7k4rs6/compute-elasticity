"""Phase 3 smoke test — verify Together AI API access end-to-end.

Sends 5 simple prompts to Together AI and prints latency + token counts.
Exits 0 on success, 1 on any failure.

Usage:
    source .venv/bin/activate
    python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


_SMOKE_PROMPTS = [
    "What is 2 + 2? Answer with just the number.",
    "Name the capital of France. One word only.",
    "What color is the sky on a clear day? One word.",
    "How many sides does a triangle have? Answer with just the number.",
    "What is the chemical symbol for water? Answer with just the formula.",
]


async def run_smoke_test() -> bool:
    """Run 5 prompts against Together AI; return True if all succeed."""
    from pilot.sampling import TogetherClient

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        client = TogetherClient()
    except Exception as exc:
        logger.error("Failed to initialise TogetherClient: %s", exc)
        return False

    all_ok = True
    for i, prompt in enumerate(_SMOKE_PROMPTS, start=1):
        try:
            completion = await client.complete(
                prompt=prompt,
                temperature=0.0,
                seed_hex="00000000",
            )
            logger.info(
                "Prompt %d/%d — latency=%.0fms  in=%d  out=%d  text=%r",
                i,
                len(_SMOKE_PROMPTS),
                completion.latency_ms,
                completion.input_tokens,
                completion.output_tokens,
                completion.text[:80],
            )
            if completion.latency_ms > 60_000:
                logger.warning(
                    "Suspiciously high latency: %.0fms", completion.latency_ms
                )
            if completion.output_tokens == 0:
                logger.error("Zero output tokens on prompt %d", i)
                all_ok = False
        except Exception as exc:
            logger.error("Prompt %d failed: %s", i, exc)
            all_ok = False

    return all_ok


def main() -> None:
    _load_env()
    ok = asyncio.run(run_smoke_test())
    if ok:
        print("\nSmoke test PASSED — Together AI is reachable and responding.")
        sys.exit(0)
    else:
        print("\nSmoke test FAILED — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
