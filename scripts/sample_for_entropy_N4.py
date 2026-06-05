"""H3 ablation — entropy/NLL at N=4 data collection.

Sends 4 completions per main-pilot problem with logprobs=True (top_logprobs=5).
Each sample uses a deterministic seed derived from (problem_id, sample_idx, temperature)
for reproducibility. Output is append-only JSONL per problem, idempotent on re-run.

Together AI logprobs schema (differs from OpenAI):
  choices[0].logprobs = {
    "tokens": [...],
    "token_logprobs": [...],   # primary: log p of selected token
    "top_logprobs": {} or [{token: logprob, ...}, ...]
  }

Estimated cost: ~$0.04 for 47 problems × 4 samples at T=0.7.

Usage:
    source .venv/bin/activate
    python scripts/sample_for_entropy_N4.py
    python scripts/sample_for_entropy_N4.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TEMPERATURE: float = 0.7
_TOP_LOGPROBS: int = 5
_MAX_TOKENS: int = 2048
_MAX_RETRIES: int = 5
_TIMEOUT_S: float = 120.0
_N_SAMPLES: int = 4
_SAMPLES_DIR = ROOT / "outputs" / "samples_for_entropy_N4"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment / ID helpers
# ---------------------------------------------------------------------------


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _require_api_key() -> str:
    key = os.getenv("TOGETHER_API_KEY")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set. Add to .env (see .env.example).")
    return key


def _load_locked_ids() -> list[str]:
    path = ROOT / "data" / "problem_ids.json"
    if not path.exists():
        raise FileNotFoundError(f"Locked problem IDs not found: {path}")
    return json.loads(path.read_text())


def _load_gate_ids() -> list[str]:
    path = ROOT / "outputs" / "gate_minus_1_labels.json"
    return json.loads(path.read_text())["gate_problems"]


def _main_pilot_ids(locked: list[str], gate: list[str]) -> list[str]:
    gate_set = set(gate)
    return sorted(pid for pid in locked if pid not in gate_set)


def _make_seed_hex(problem_id: str, sample_idx: int, temperature: float) -> str:
    """Deterministic seed matching pilot/sampling.py convention."""
    return f"{hash((problem_id, sample_idx, temperature)) & 0xFFFFFFFFFFFFFFFF:016x}"


def _load_existing_seeds(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seeds: set[str] = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                seeds.add(json.loads(line)["seed_hex"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seeds


# ---------------------------------------------------------------------------
# API call (mirrors sample_for_entropy.py; adds seed param)
# ---------------------------------------------------------------------------


def _chat_with_logprobs(
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
    seed_hex: str,
) -> dict:
    """POST a chat completion with logprobs and seed; return raw response dict.

    Retries up to _MAX_RETRIES times on transient HTTP errors and timeouts
    with exponential backoff + jitter.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": _MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": _TOP_LOGPROBS,
        "seed": int(seed_hex[:8], 16),
    }

    for attempt in range(_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=_TIMEOUT_S) as client:
                t0 = time.monotonic()
                resp = client.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                latency_ms = (time.monotonic() - t0) * 1000

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"Exhausted {_MAX_RETRIES} retries on HTTP {resp.status_code}"
                    )
                delay = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "HTTP %d (attempt %d/%d); retrying in %.1fs",
                    resp.status_code, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Non-retryable HTTP {resp.status_code}: {resp.text[:300]}"
                )

            data = resp.json()
            data["_latency_ms"] = latency_ms
            return data

        except httpx.TimeoutException as exc:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Request timed out after {_MAX_RETRIES} retries"
                ) from exc
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "Timeout (attempt %d/%d); retrying in %.1fs",
                attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Logprobs parsing (Together AI schema — identical to sample_for_entropy.py)
# ---------------------------------------------------------------------------


def _parse_together_logprobs(
    choice: dict,
) -> tuple[list[float], list[list[tuple[str, float]]], bool]:
    lp_field = choice.get("logprobs") or {}
    token_lps: list[float] = [
        float(v) for v in (lp_field.get("token_logprobs") or []) if v is not None
    ]
    top_raw = lp_field.get("top_logprobs")
    top_lps_per_token: list[list[tuple[str, float]]] = []
    top_lps_available = False
    if isinstance(top_raw, list) and top_raw:
        top_lps_available = True
        for tok_dict in top_raw:
            if isinstance(tok_dict, dict) and tok_dict:
                top_lps_per_token.append(
                    [(tok, float(lp)) for tok, lp in tok_dict.items()]
                )
            else:
                top_lps_per_token.append([])
    return token_lps, top_lps_per_token, top_lps_available


def _token_entropy(top_lps: list[tuple[str, float]]) -> float:
    if not top_lps:
        return 0.0
    raw_probs = [math.exp(lp) for _, lp in top_lps]
    total = sum(raw_probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in raw_probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------


def _process_sample(
    problem_id: str,
    subject: str,
    sample_idx: int,
    prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    out_path: Path,
    seed_hex: str,
) -> dict:
    """Call API for one sample, compute NLL/entropy, append to JSONL."""
    data = _chat_with_logprobs(api_key, api_base, model, prompt, _TEMPERATURE, seed_hex)
    choice = data["choices"][0]
    full_response: str = choice["message"]["content"]
    usage = data.get("usage", {})

    token_lps, top_lps_per_token, top_lps_available = _parse_together_logprobs(choice)
    nlls = [-lp for lp in token_lps]
    n_tokens = len(nlls)
    mean_nll = float(sum(nlls) / n_tokens) if n_tokens else 0.0

    record: dict = {
        "schema_version": "v6.0-pilot",
        "problem_id": problem_id,
        "subject": subject,
        "model": model,
        "temperature": _TEMPERATURE,
        "sample_idx": sample_idx,
        "seed_hex": seed_hex,
        "n_tokens": n_tokens,
        "mean_per_token_nll": mean_nll,
        "top_logprobs_available": top_lps_available,
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
        "latency_ms": data.get("_latency_ms", 0.0),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if top_lps_available and top_lps_per_token:
        entropies = [_token_entropy(lps) for lps in top_lps_per_token]
        n_e = len(entropies)
        record["mean_per_token_entropy"] = float(sum(entropies) / n_e) if n_e else 0.0

    with out_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 ablation: entropy/NLL at N=4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_env()

    from pilot.config import (
        MODEL,
        TOGETHER_API_BASE,
        TOGETHER_INPUT_PRICE_PER_TOKEN,
        TOGETHER_OUTPUT_PRICE_PER_TOKEN,
    )
    from pilot.data_loader import load_gpqa_diamond
    from pilot.prompts import PROMPT_TEMPLATE_HASH, _LOCKED_PROMPT_HASH, render_prompt

    # Verify locked prompt hash
    if PROMPT_TEMPLATE_HASH != _LOCKED_PROMPT_HASH:
        logger.error(
            "Prompt template hash mismatch: expected %s, got %s",
            _LOCKED_PROMPT_HASH, PROMPT_TEMPLATE_HASH,
        )
        sys.exit(1)
    logger.info("Prompt hash verified: %s", PROMPT_TEMPLATE_HASH[:16])

    locked_ids = _load_locked_ids()
    gate_ids = _load_gate_ids()
    main_ids = _main_pilot_ids(locked_ids, gate_ids)

    if len(main_ids) != 47:
        logger.error("Expected 47 main-pilot IDs, got %d.", len(main_ids))
        sys.exit(1)

    all_problems = load_gpqa_diamond()
    problem_map = {p.id: p for p in all_problems}

    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Cost estimate
    n_calls_est = _N_SAMPLES * len(main_ids)
    est_cost = (
        n_calls_est * 800 * TOGETHER_OUTPUT_PRICE_PER_TOKEN
        + n_calls_est * 400 * TOGETHER_INPUT_PRICE_PER_TOKEN
    )
    if est_cost > 1.00:
        logger.error("Estimated cost $%.4f exceeds $1.00 abort threshold. Halting.", est_cost)
        sys.exit(1)

    sep = "=" * 68
    print(f"\n{sep}")
    print("H3 Ablation — entropy/NLL at N=4 — sampling plan")
    print(sep)
    print(f"  Model              : {MODEL}")
    print(f"  Temperature        : {_TEMPERATURE}")
    print(f"  Samples per problem: {_N_SAMPLES}")
    print(f"  Problems           : {len(main_ids)}")
    print(f"  Total API calls    : {n_calls_est}")
    print(f"  Estimated cost     : ${est_cost:.4f}")
    print(f"  Output dir         : outputs/samples_for_entropy_N4/")
    print(f"  Prompt hash        : {PROMPT_TEMPLATE_HASH[:16]}... (verified)")
    if args.dry_run:
        print("\n  DRY RUN — no API calls will be made.")
    print(sep)

    if args.dry_run:
        return

    api_key = _require_api_key()
    total_cost = 0.0
    total_calls = 0

    for num, pid in enumerate(main_ids, start=1):
        p = problem_map[pid]
        prompt = render_prompt(
            question_text=p.question,
            option_a=p.option_a,
            option_b=p.option_b,
            option_c=p.option_c,
            option_d=p.option_d,
        )
        out_path = _SAMPLES_DIR / f"{pid}.jsonl"
        existing_seeds = _load_existing_seeds(out_path)
        needed = [
            (i, _make_seed_hex(pid, i, _TEMPERATURE))
            for i in range(_N_SAMPLES)
            if _make_seed_hex(pid, i, _TEMPERATURE) not in existing_seeds
        ]

        if not needed:
            logger.info("[%d/%d] %s — all %d samples present, skipping.", num, len(main_ids), pid, _N_SAMPLES)
            continue

        logger.info("[%d/%d] %s (%s) — %d sample(s) to collect…", num, len(main_ids), pid, p.subject, len(needed))
        for sample_idx, seed_hex in needed:
            rec = _process_sample(
                problem_id=pid,
                subject=p.subject,
                sample_idx=sample_idx,
                prompt=prompt,
                api_key=api_key,
                api_base=TOGETHER_API_BASE,
                model=MODEL,
                out_path=out_path,
                seed_hex=seed_hex,
            )
            call_cost = (
                rec["input_tokens"] * TOGETHER_INPUT_PRICE_PER_TOKEN
                + rec["output_tokens"] * TOGETHER_OUTPUT_PRICE_PER_TOKEN
            )
            total_cost += call_cost
            total_calls += 1
            logger.info(
                "  [%d/%d] idx=%d  n_tok=%d  nll=%.4f  entropy=%s  cost=$%.5f",
                total_calls, n_calls_est, sample_idx, rec["n_tokens"],
                rec["mean_per_token_nll"],
                f"{rec['mean_per_token_entropy']:.4f}" if "mean_per_token_entropy" in rec else "n/a",
                call_cost,
            )

    print(f"\n{sep}")
    print("H3 Ablation — sampling complete")
    print(sep)
    print(f"  API calls made     : {total_calls}")
    print(f"  Actual cost        : ${total_cost:.4f}")
    print(sep)


if __name__ == "__main__":
    main()
