"""Phase 13 -- Full 198 GPQA Diamond sampling for both models.

Phase A: Verify problem set
  - Load all 198 GPQA Diamond problems (stable IDs gpqa_diamond_<idx>).
  - Confirm existing 47 main-pilot IDs are a subset; STOP if any are missing.
  - Verify locked prompt SHA-256 still matches pilot/prompts.py.

Phase B: Sampling (both models, idempotent, with logprobs)
  - Qwen2.5-7B-Instruct-Turbo  -> outputs/samples/
  - Meta-Llama-3-8B-Instruct-Lite -> outputs/samples_model2/
  - Skip problems already at >=64 valid T=0.7 samples.
  - Request token_logprobs; store mean per-token entropy in record.
  - T=0.7, unique secrets.token_bytes(8) seed per sample, up to 32 concurrent.
  - Cost guard: abort if estimate > $9.

Usage:
    source .venv/bin/activate
    python scripts/run_phase13_sampling.py [--dry-run] [--model {1,2,both}]
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
MODEL1: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"
MODEL2: str = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

TEMPERATURE: float = 0.7
N_PER_PROBLEM: int = 64
COST_ABORT_USD: float = 9.0
MAX_CONCURRENT: int = 32
MAX_RETRIES: int = 5
CONNECT_TIMEOUT: float = 30.0
READ_TIMEOUT: float = 120.0
MAX_OUTPUT_TOKENS: int = 2048
CHECKPOINT_INTERVAL: int = 10

# Pricing (USD per token, both input and output)
MODEL1_PRICE: float = 0.18 / 1_000_000
MODEL2_PRICE: float = 0.14 / 1_000_000

# Estimated tokens per sample (conservative)
EST_INPUT_TOKENS: int = 450
EST_OUTPUT_TOKENS: int = 900

SAMPLES1_DIR = ROOT / "outputs" / "samples"
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


# ---------------------------------------------------------------------------
# Phase A: problem set verification
# ---------------------------------------------------------------------------


def _load_and_verify_problem_set() -> tuple[list[Any], list[str], list[str]]:
    """Load 198 problems, verify existing 47 are a subset.

    Returns:
        (all_problems, exploratory_ids, confirmatory_ids)
    """
    from pilot.data_loader import load_gpqa_diamond
    from pilot.prompts import _LOCKED_PROMPT_HASH, PROMPT_TEMPLATE_HASH

    # Verify prompt hash
    if PROMPT_TEMPLATE_HASH != _LOCKED_PROMPT_HASH:
        raise RuntimeError(
            f"Prompt hash mismatch: {PROMPT_TEMPLATE_HASH!r} != {_LOCKED_PROMPT_HASH!r}"
        )
    logger.info("Prompt hash verified: %s...", PROMPT_TEMPLATE_HASH[:16])

    all_problems = load_gpqa_diamond()
    if len(all_problems) != 198:
        raise RuntimeError(f"Expected 198 GPQA Diamond problems, got {len(all_problems)}")

    all_ids = {p.id for p in all_problems}

    # Load existing 47 main-pilot IDs
    locked_ids = set(json.loads((ROOT / "data" / "problem_ids.json").read_text()))
    gate_ids = set(
        json.loads((ROOT / "outputs" / "gate_minus_1_labels.json").read_text())[
            "gate_problems"
        ]
    )
    exploratory_ids = sorted(locked_ids - gate_ids)  # 47 main pilot problems

    # STOP condition: any existing ID not in the 198
    missing = [pid for pid in exploratory_ids if pid not in all_ids]
    if missing:
        raise RuntimeError(
            f"STOP: {len(missing)} existing pilot IDs not found in 198-problem dataset: {missing}"
        )
    logger.info(
        "Problem set verified: 198 problems, %d exploratory, %d confirmatory",
        len(exploratory_ids),
        198 - len(exploratory_ids),
    )

    confirmatory_ids = sorted(pid for pid in all_ids if pid not in set(exploratory_ids))
    return all_problems, exploratory_ids, confirmatory_ids


# ---------------------------------------------------------------------------
# Sample counting
# ---------------------------------------------------------------------------


def _count_valid_samples(path: Path) -> int:
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
# Logprob extraction
# ---------------------------------------------------------------------------


def _extract_token_logprobs(logprobs_raw: Any) -> list[float] | None:
    """Extract token log probabilities from Together AI response (either format)."""
    if logprobs_raw is None:
        return None
    # OpenAI v1 format: {"content": [{"logprob": float, ...}]}
    if isinstance(logprobs_raw, dict) and "content" in logprobs_raw:
        content = logprobs_raw["content"]
        if content and isinstance(content, list):
            result = [tok.get("logprob") for tok in content if tok.get("logprob") is not None]
            return result if result else None
    # Legacy Together AI format: {"token_logprobs": [...]}
    if isinstance(logprobs_raw, dict) and "token_logprobs" in logprobs_raw:
        lp = logprobs_raw["token_logprobs"]
        if lp and isinstance(lp, list):
            return [x for x in lp if x is not None]
    return None


def _mean_entropy(token_logprobs: list[float] | None) -> float | None:
    """Mean per-token entropy (surprise = -logprob). Lower = more confident."""
    if not token_logprobs:
        return None
    return float(-sum(token_logprobs) / len(token_logprobs))


# ---------------------------------------------------------------------------
# Async API call
# ---------------------------------------------------------------------------


class _RetryableError(Exception):
    pass


async def _call_api(api_key: str, model: str, prompt: str, seed_hex: str) -> dict[str, Any]:
    """Single chat completion with logprobs requested."""
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=30.0, pool=5.0
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "seed": int(seed_hex[:8], 16),
        "logprobs": 1,  # request log probabilities for chosen tokens
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{TOGETHER_API_BASE}/chat/completions", headers=headers, json=payload
        )
    if resp.status_code in (429, 500, 502, 503, 504):
        raise _RetryableError(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


async def _call_with_retry(
    api_key: str, model: str, prompt: str, seed_hex: str
) -> dict[str, Any] | None:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await _call_api(api_key, model, prompt, seed_hex)
        except _RetryableError as exc:
            if attempt == MAX_RETRIES:
                logger.error("Exhausted %d retries (%s) -- skipping sample", MAX_RETRIES, exc)
                return None
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning("Attempt %d: %s; retrying in %.1fs", attempt + 1, exc, delay)
            await asyncio.sleep(delay)
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            if attempt == MAX_RETRIES:
                logger.error("%s after %d retries -- skipping sample", type(exc).__name__, MAX_RETRIES)
                return None
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning("%s (attempt %d/%d); retrying in %.1fs", type(exc).__name__, attempt + 1, MAX_RETRIES, delay)
            await asyncio.sleep(delay)
    return None


# ---------------------------------------------------------------------------
# Problem sampler
# ---------------------------------------------------------------------------


async def _sample_problem(
    problem: Any,
    api_key: str,
    model: str,
    out_dir: Path,
    semaphore: asyncio.Semaphore,
    extract_fn: Any,
    prompt_hash: str,
    schema_version: str,
) -> dict[str, Any]:
    """Sample up to N_PER_PROBLEM completions idempotently."""
    from pilot.prompts import render_prompt

    out_path = out_dir / f"{problem.id}.jsonl"
    existing = _count_valid_samples(out_path)
    if existing >= N_PER_PROBLEM:
        return {"pid": problem.id, "new": 0, "existing": existing, "skipped": True}

    prompt = render_prompt(
        question_text=problem.question,
        option_a=problem.option_a,
        option_b=problem.option_b,
        option_c=problem.option_c,
        option_d=problem.option_d,
    )

    needed = N_PER_PROBLEM - existing
    new_count = 0
    total_in_tok = 0
    total_out_tok = 0
    logprobs_available: bool | None = None

    with out_path.open("a") as fh:
        for _ in range(needed):
            seed_hex = secrets.token_bytes(8).hex()
            async with semaphore:
                t0 = time.monotonic()
                data = await _call_with_retry(api_key, model, prompt, seed_hex)
                latency_ms = (time.monotonic() - t0) * 1000

            if data is None:
                logger.error("%s: sample skipped (API failure)", problem.id)
                continue

            choice = data["choices"][0]
            text = choice["message"]["content"]
            usage = data.get("usage", {})
            in_tok = int(usage.get("prompt_tokens", 0))
            out_tok = int(usage.get("completion_tokens", 0))

            # Extract logprobs
            raw_lp = choice.get("logprobs")
            token_lps = _extract_token_logprobs(raw_lp)
            mean_ent = _mean_entropy(token_lps)
            if logprobs_available is None:
                logprobs_available = token_lps is not None

            # Extract answer
            extraction = extract_fn(text)
            is_correct = (
                extraction.answer == problem.ground_truth
                if extraction.answer is not None
                else False
            )
            extracted = extraction.answer if extraction.answer else "UNPARSEABLE"

            record = {
                "schema_version": schema_version,
                "problem_id": problem.id,
                "subject": problem.subject,
                "ground_truth": problem.ground_truth,
                "model": model,
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
                "mean_token_entropy": mean_ent,  # None if logprobs unavailable
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()

            new_count += 1
            total_in_tok += in_tok
            total_out_tok += out_tok

    return {
        "pid": problem.id,
        "new": new_count,
        "existing": existing,
        "skipped": False,
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "logprobs_available": logprobs_available,
    }


# ---------------------------------------------------------------------------
# Run one model over all 198 problems
# ---------------------------------------------------------------------------


async def _run_model(
    all_problems: list[Any],
    model: str,
    out_dir: Path,
    api_key: str,
    price_per_tok: float,
    schema_version: str,
    dry_run: bool,
) -> None:
    from pilot.prompts import PROMPT_TEMPLATE_HASH
    from pilot.scoring import extract_answer

    out_dir.mkdir(parents=True, exist_ok=True)

    # Cost estimate
    n_calls_est = sum(
        max(0, N_PER_PROBLEM - _count_valid_samples(out_dir / f"{p.id}.jsonl"))
        for p in all_problems
    )
    est_cost = n_calls_est * (EST_INPUT_TOKENS + EST_OUTPUT_TOKENS) * price_per_tok

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"Model: {model}")
    print(sep)
    print(f"  Problems total     : {len(all_problems)}")
    print(f"  Calls needed now   : {n_calls_est}")
    print(f"  Est cost (now)     : ${est_cost:.3f}")
    print(f"  Cost abort at      : ${COST_ABORT_USD}")
    if dry_run:
        print("\n  DRY RUN -- no API calls.")
        print(sep)
        return
    if est_cost > COST_ABORT_USD:
        raise RuntimeError(
            f"Estimated cost ${est_cost:.3f} exceeds abort threshold ${COST_ABORT_USD} -- halting"
        )
    print(sep)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [
        _sample_problem(p, api_key, model, out_dir, semaphore, extract_answer,
                        PROMPT_TEMPLATE_HASH, schema_version)
        for p in all_problems
    ]

    total_new = 0
    total_in_tok = 0
    total_out_tok = 0
    completed = 0
    n_with_logprobs = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        if not result["skipped"]:
            total_new += result["new"]
            total_in_tok += result.get("total_input_tokens", 0)
            total_out_tok += result.get("total_output_tokens", 0)
            if result.get("logprobs_available"):
                n_with_logprobs += 1
        actual_cost = (total_in_tok + total_out_tok) * price_per_tok
        if completed % CHECKPOINT_INTERVAL == 0:
            logger.info(
                "[%d/%d] %s new=%d cost=$%.4f",
                completed, len(all_problems), result["pid"],
                result.get("new", 0), actual_cost,
            )

    actual_cost = (total_in_tok + total_out_tok) * price_per_tok
    print(f"\n{sep}")
    print(f"Sampling complete: {model}")
    print(sep)
    print(f"  New samples added  : {total_new}")
    print(f"  Input tokens       : {total_in_tok:,}")
    print(f"  Output tokens      : {total_out_tok:,}")
    print(f"  Actual cost        : ${actual_cost:.4f}")
    print(f"  Problems w/ logprobs: {n_with_logprobs}")
    print(sep)


# ---------------------------------------------------------------------------
# Parse rate report
# ---------------------------------------------------------------------------


def _report_parse_rates(all_problems: list[Any], dirs: dict[str, Path]) -> None:
    """Print parse rate per model over the full 198."""
    for model_label, out_dir in dirs.items():
        total = 0
        parsed = 0
        for p in all_problems:
            path = out_dir / f"{p.id}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if abs(obj.get("temperature", -1) - TEMPERATURE) < 1e-6:
                        total += 1
                        ea = obj.get("extracted_answer", "")
                        if ea and ea != "UNPARSEABLE" and ea in "ABCD":
                            parsed += 1
                except json.JSONDecodeError:
                    continue
        rate = parsed / total if total else 0.0
        flag = " [FLAG: < 90%]" if rate < 0.90 else ""
        print(f"  {model_label}: parse rate = {rate:.4f} ({parsed}/{total}){flag}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Phase 13 sampling for both models")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", choices=["1", "2", "both"], default="both")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_env()

    # Phase A: verify problem set + prompt hash
    all_problems, exploratory_ids, confirmatory_ids = _load_and_verify_problem_set()
    print(f"\nProblem set: {len(all_problems)} total")
    print(f"  Exploratory (existing 47 main-pilot): {len(exploratory_ids)}")
    print(f"  Confirmatory (new 151): {len(confirmatory_ids)}")

    api_key = _require_api_key()

    run_m1 = args.model in ("1", "both")
    run_m2 = args.model in ("2", "both")

    if run_m1:
        asyncio.run(
            _run_model(
                all_problems, MODEL1, SAMPLES1_DIR, api_key,
                MODEL1_PRICE, "v6.0-pilot-phase13", args.dry_run,
            )
        )
    if run_m2:
        asyncio.run(
            _run_model(
                all_problems, MODEL2, SAMPLES2_DIR, api_key,
                MODEL2_PRICE, "v6.0-pilot-model2-phase13", args.dry_run,
            )
        )

    if not args.dry_run:
        print("\nParse rates (full 198):")
        dirs: dict[str, Path] = {}
        if run_m1:
            dirs["Qwen2.5-7B"] = SAMPLES1_DIR
        if run_m2:
            dirs["Llama-3-8B"] = SAMPLES2_DIR
        _report_parse_rates(all_problems, dirs)


if __name__ == "__main__":
    main()
