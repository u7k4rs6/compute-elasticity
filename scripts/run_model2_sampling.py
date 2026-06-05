"""Phase 12 -- Model 2 sampling: Meta-Llama-3-8B-Instruct-Lite.

Mirrors the model-1 pilot exactly (47 main-pilot problems, T=0.7, N=64,
locked prompt template verified by SHA-256 hash), writing to
outputs/samples_model2/ (never touches outputs/samples/).

Differences from model-1 run:
  - Model: meta-llama/Meta-Llama-3-8B-Instruct-Lite  (Meta Llama 3 family)
  - Seeds: secrets.token_bytes(8).hex() per sample (random, not hash-based)
  - Idempotency: by sample count (not by seed) -- safe to re-run

Handles httpx.ReadTimeout explicitly (separate from connection timeout).
Cost guard: aborts if estimate > $8.

Usage:
    source .venv/bin/activate
    python scripts/run_model2_sampling.py
    python scripts/run_model2_sampling.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import random
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL2: str = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
TEMPERATURE: float = 0.7
N_PER_PROBLEM: int = 64
COST_ABORT_USD: float = 8.0
MAX_CONCURRENT: int = 16
MAX_RETRIES: int = 5
CONNECT_TIMEOUT: float = 30.0
READ_TIMEOUT: float = 120.0
MAX_OUTPUT_TOKENS: int = 2048

# Pricing for Meta-Llama-3-8B-Instruct-Lite ($0.14/M tokens both in/out)
INPUT_PRICE_PER_TOKEN: float = 0.14 / 1_000_000
OUTPUT_PRICE_PER_TOKEN: float = 0.14 / 1_000_000

SAMPLES2_DIR = ROOT / "outputs" / "samples_model2"
TOGETHER_API_BASE = "https://api.together.xyz/v1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _require_api_key() -> str:
    key = os.getenv("TOGETHER_API_KEY")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set. Add to .env.")
    return key


def _main_pilot_ids() -> list[str]:
    locked = json.loads((ROOT / "data" / "problem_ids.json").read_text())
    gate_set = set(
        json.loads((ROOT / "outputs" / "gate_minus_1_labels.json").read_text())[
            "gate_problems"
        ]
    )
    return sorted(pid for pid in locked if pid not in gate_set)


# ---------------------------------------------------------------------------
# Sample counting (for idempotency)
# ---------------------------------------------------------------------------


def _count_existing_samples(path: Path) -> int:
    """Count T=0.7 samples already in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if abs(obj.get("temperature", -1) - TEMPERATURE) < 1e-6:
                count += 1
        except json.JSONDecodeError:
            continue
    return count


# ---------------------------------------------------------------------------
# Async API call with explicit ReadTimeout handling
# ---------------------------------------------------------------------------


async def _call_api(api_key: str, prompt: str, seed_hex: str) -> dict[str, Any]:
    """Single chat completion call; raises RetryableError on transient failures."""
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=30.0, pool=5.0
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL2,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "seed": int(seed_hex[:8], 16),
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{TOGETHER_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
        )
    if resp.status_code in (429, 500, 502, 503, 504):
        raise _RetryableError(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


class _RetryableError(Exception):
    pass


async def _call_with_retry(
    api_key: str,
    prompt: str,
    seed_hex: str,
) -> dict[str, Any] | None:
    """Call API with exponential backoff. Returns None on persistent failure."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await _call_api(api_key, prompt, seed_hex)
        except _RetryableError as exc:
            if attempt == MAX_RETRIES:
                logger.error(
                    "Exhausted %d retries (%s) -- skipping sample", MAX_RETRIES, exc
                )
                return None
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning("Attempt %d: %s; retrying in %.1fs", attempt + 1, exc, delay)
            await asyncio.sleep(delay)
        except httpx.ReadTimeout:
            if attempt == MAX_RETRIES:
                logger.error(
                    "ReadTimeout after %d retries -- skipping sample", MAX_RETRIES
                )
                return None
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "ReadTimeout (attempt %d/%d); retrying in %.1fs",
                attempt + 1,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)
        except httpx.ConnectTimeout:
            if attempt == MAX_RETRIES:
                logger.error(
                    "ConnectTimeout after %d retries -- skipping sample", MAX_RETRIES
                )
                return None
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning("ConnectTimeout; retrying in %.1fs", delay)
            await asyncio.sleep(delay)
    return None  # unreachable


# ---------------------------------------------------------------------------
# Problem sampler
# ---------------------------------------------------------------------------


async def _sample_problem(
    problem: Any,
    api_key: str,
    prompt: str,
    out_path: Path,
    semaphore: asyncio.Semaphore,
    extract_fn: Any,
    prompt_hash: str,
) -> dict[str, Any]:
    """Sample N_PER_PROBLEM completions for one problem idempotently."""
    existing = _count_existing_samples(out_path)
    if existing >= N_PER_PROBLEM:
        return {"pid": problem.id, "new": 0, "existing": existing, "skipped": True}

    needed = N_PER_PROBLEM - existing
    new_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    with out_path.open("a") as fh:
        for _ in range(needed):
            seed_hex = secrets.token_bytes(8).hex()
            async with semaphore:
                t0 = time.monotonic()
                data = await _call_with_retry(api_key, prompt, seed_hex)
                latency_ms = (time.monotonic() - t0) * 1000

            if data is None:
                # Never fabricate -- log and continue to next slot
                logger.error(
                    "%s: sample skipped due to persistent API failure", problem.id
                )
                continue

            choice = data["choices"][0]
            text = choice["message"]["content"]
            usage = data.get("usage", {})
            in_tok = int(usage.get("prompt_tokens", 0))
            out_tok = int(usage.get("completion_tokens", 0))

            extraction = extract_fn(text)
            is_correct = (
                extraction.answer == problem.ground_truth
                if extraction.answer is not None
                else False
            )
            extracted = extraction.answer if extraction.answer else "UNPARSEABLE"

            record = {
                "schema_version": "v6.0-pilot-model2",
                "problem_id": problem.id,
                "subject": problem.subject,
                "ground_truth": problem.ground_truth,
                "model": MODEL2,
                "provider": "together_ai",
                "temperature": TEMPERATURE,
                "sample_idx": existing + new_count,
                "n_total_in_batch": N_PER_PROBLEM,
                "seed_hex": seed_hex,
                "prompt_template_hash": prompt_hash,
                "full_response": text,
                "extracted_answer": extracted,
                "extraction_pass": extraction.pass_number,
                "correct": is_correct,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": latency_ms,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()

            new_count += 1
            total_input_tokens += in_tok
            total_output_tokens += out_tok

    return {
        "pid": problem.id,
        "new": new_count,
        "existing": existing,
        "skipped": False,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _main(dry_run: bool) -> None:
    from pilot.data_loader import load_gpqa_diamond
    from pilot.prompts import _LOCKED_PROMPT_HASH, PROMPT_TEMPLATE_HASH, render_prompt
    from pilot.scoring import extract_answer

    if PROMPT_TEMPLATE_HASH != _LOCKED_PROMPT_HASH:
        logger.error(
            "Prompt hash mismatch: expected %s, got %s",
            _LOCKED_PROMPT_HASH,
            PROMPT_TEMPLATE_HASH,
        )
        sys.exit(1)
    logger.info("Prompt hash verified: %s", PROMPT_TEMPLATE_HASH[:16])

    main_ids = _main_pilot_ids()
    if len(main_ids) != 47:
        logger.error("Expected 47 main IDs, got %d", len(main_ids))
        sys.exit(1)

    all_problems = load_gpqa_diamond()
    problem_map = {p.id: p for p in all_problems}

    # Cost estimate
    # Use a realistic estimate: ~450 input + 900 output tokens per call
    n_calls_est = sum(
        max(0, N_PER_PROBLEM - _count_existing_samples(SAMPLES2_DIR / f"{pid}.jsonl"))
        for pid in main_ids
    )
    est_cost = (
        n_calls_est * 450 * INPUT_PRICE_PER_TOKEN
        + n_calls_est * 900 * OUTPUT_PRICE_PER_TOKEN
    )
    est_total_cost = (
        N_PER_PROBLEM * 47 * 450 * INPUT_PRICE_PER_TOKEN
        + N_PER_PROBLEM * 47 * 900 * OUTPUT_PRICE_PER_TOKEN
    )

    sep = "=" * 72
    print(f"\n{sep}")
    print("Model-2 Pilot Sampling")
    print(sep)
    print(f"  Model             : {MODEL2}")
    print(f"  Temperature       : {TEMPERATURE}")
    print(f"  Samples/problem   : {N_PER_PROBLEM}")
    print(f"  Problems          : {len(main_ids)}")
    print(f"  Calls needed now  : {n_calls_est}  (skip existing)")
    print(f"  Est cost (now)    : ${est_cost:.3f}")
    print(f"  Est cost (full)   : ${est_total_cost:.3f}")
    print(f"  Cost abort at     : ${COST_ABORT_USD}")
    print(f"  Prompt hash       : {PROMPT_TEMPLATE_HASH[:16]}... (verified)")
    if dry_run:
        print("\n  DRY RUN -- no API calls.")
    print(sep)

    if est_cost > COST_ABORT_USD:
        logger.error(
            "Estimated cost $%.3f exceeds abort threshold $%.1f -- halting",
            est_cost,
            COST_ABORT_USD,
        )
        sys.exit(1)

    if dry_run:
        return

    api_key = _require_api_key()
    SAMPLES2_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = []
    for pid in main_ids:
        p = problem_map[pid]
        prompt = render_prompt(
            question_text=p.question,
            option_a=p.option_a,
            option_b=p.option_b,
            option_c=p.option_c,
            option_d=p.option_d,
        )
        out_path = SAMPLES2_DIR / f"{pid}.jsonl"
        tasks.append(
            _sample_problem(
                p,
                api_key,
                prompt,
                out_path,
                semaphore,
                extract_answer,
                PROMPT_TEMPLATE_HASH,
            )
        )

    total_new = 0
    total_in_tok = 0
    total_out_tok = 0
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        if not result["skipped"]:
            total_new += result["new"]
            total_in_tok += result.get("total_input_tokens", 0)
            total_out_tok += result.get("total_output_tokens", 0)
        actual_cost = (
            total_in_tok * INPUT_PRICE_PER_TOKEN
            + total_out_tok * OUTPUT_PRICE_PER_TOKEN
        )
        logger.info(
            "[%d/%d] %s  new=%d  actual_cost=$%.4f",
            completed,
            len(main_ids),
            result["pid"],
            result.get("new", 0),
            actual_cost,
        )

    actual_cost = (
        total_in_tok * INPUT_PRICE_PER_TOKEN + total_out_tok * OUTPUT_PRICE_PER_TOKEN
    )

    print(f"\n{sep}")
    print("Model-2 Sampling Complete")
    print(sep)
    print(f"  New samples added : {total_new}")
    print(f"  Input tokens      : {total_in_tok:,}")
    print(f"  Output tokens     : {total_out_tok:,}")
    print(f"  Actual cost       : ${actual_cost:.4f}")
    print(sep)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Model-2 pilot sampling")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _load_env()

    asyncio.run(_main(args.dry_run))


if __name__ == "__main__":
    main()
