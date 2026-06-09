"""Phase 14b -- Reasoning-model probe on 47 exploratory problems.

EXPLORATORY (post-hoc). Pre-registered PH1-PH4 on confirmatory set unchanged.

Originally planned for Qwen/QwQ-32B, which requires a dedicated (non-serverless)
endpoint on Together AI and was therefore inaccessible. Substituted with
MiniMaxAI/MiniMax-M2.7 (serverless, reasoning-native: smoke test showed 1573
billed output tokens vs 670 visible chars, indicating hidden CoT tokens).

Smoke test: one API call to verify model ID and answer extraction.
Sampling: N=16 per problem (or less if budget demands), T=0.7, locked prompt.
Output: outputs/samples_qwq/ (JSONL per problem), outputs/gate_qwq/ (analysis).

Hard cap: $6 total (abort before exceeding).
Budget: MiniMax-M2.7 = $0.30/M input + $1.20/M output.

Usage:
    python scripts/run_phase14b_qwq_probe.py --smoke-test   # B1: one call
    python scripts/run_phase14b_qwq_probe.py --sample       # B3: full sampling
    python scripts/run_phase14b_qwq_probe.py --analyze      # B4-B5: analysis+figure
    python scripts/run_phase14b_qwq_probe.py --all          # smoke+sample+analyze
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
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# QwQ-32B requires a dedicated endpoint (not serverless) on Together AI.
# MiniMax-M2.7 is the substitute: serverless, reasoning-native (hidden CoT
# tokens; smoke test showed 1573 billed output tokens vs 670 visible chars).
MODEL_QWQ: str = "MiniMaxAI/MiniMax-M2.7"
MODEL_QWQ_ORIGINAL: str = "Qwen/QwQ-32B"  # intended model, unavailable serverless

TEMPERATURE: float = 0.7
# N=8: MiniMax M2.7 uses ~5k-16k output tokens (hidden CoT); N=16 would exceed
# $6 at worst case. Hard abort at $6 protects against overrun.
N_PER_PROBLEM: int = 8
COST_ABORT_USD: float = 6.0
MAX_CONCURRENT: int = 4  # M2.7 is slow; fewer concurrent avoids 429s
MAX_RETRIES: int = 5
CONNECT_TIMEOUT: float = 30.0
READ_TIMEOUT: float = (
    600.0  # M2.7 reasoning can take ~85s; 600s avoids spurious retries
)
# 16384 gives room for both thinking tokens and the visible final answer.
# 8192 was too small: model maxed out mid-think, leaving empty content.
MAX_OUTPUT_TOKENS: int = 16384
CHECKPOINT_INTERVAL: int = 5

# Pricing USD/token (input != output)
QWQ_PRICE_INPUT: float = 0.30 / 1_000_000
QWQ_PRICE_OUTPUT: float = 1.20 / 1_000_000
QWQ_PRICE_PER_TOKEN: float = (0.30 + 1.20) / 2 / 1_000_000  # kept for backward compat

TOGETHER_API_BASE: str = "https://api.together.xyz/v1"
SCHEMA_VERSION: str = "1.0-minimax-m27-probe"

OUT_DIR_QWQ: Path = ROOT / "outputs" / "samples_qwq"
OUT_DIR_GATE: Path = ROOT / "outputs" / "gate_qwq"

_N_VALUES: list[int] = [1, 2, 4, 8]
_AGREEMENT_K_VALUES: list[int] = [4]
_AGREEMENT_TAUS: list[float] = [
    round(0.50 + i * 0.05, 2) for i in range(11)
]  # 0.50 .. 1.00
_ENTROPY_K: int = 4
_N_ENTROPY_THRESHOLDS: int = 40
_N_MC: int = 2000
_ENTROPY_SEED: int = 44
_SEED_MV: int = 42
_BOOTSTRAP_SEED: int = 42
_BOOTSTRAP_N: int = 1000

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
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


def _load_exploratory_problems() -> list[Any]:
    """Load the 47 exploratory problems."""
    from pilot.data_loader import load_gpqa_diamond

    all_problems = load_gpqa_diamond()
    all_map = {p.id: p for p in all_problems}

    locked_ids: set[str] = set(
        json.loads((ROOT / "data" / "problem_ids.json").read_text())
    )
    gate_ids: set[str] = set(
        json.loads((ROOT / "outputs" / "gate_minus_1_labels.json").read_text())[
            "gate_problems"
        ]
    )
    exploratory_ids = sorted(locked_ids - gate_ids)
    return [all_map[pid] for pid in exploratory_ids if pid in all_map]


def _count_valid_samples(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if r.get("extracted_answer") and r["extracted_answer"] != "UNPARSEABLE":
                    count += 1
            except json.JSONDecodeError:
                pass
    return count


def _extract_token_logprobs(logprobs_raw: Any) -> list[float] | None:
    if logprobs_raw is None:
        return None
    if isinstance(logprobs_raw, dict) and "content" in logprobs_raw:
        content = logprobs_raw["content"]
        if isinstance(content, list):
            return [t.get("logprob", 0.0) for t in content if isinstance(t, dict)]
    if isinstance(logprobs_raw, dict) and "token_logprobs" in logprobs_raw:
        lp = logprobs_raw["token_logprobs"]
        if isinstance(lp, list):
            return [x for x in lp if isinstance(x, (int, float))]
    return None


def _mean_entropy(token_logprobs: list[float] | None) -> float | None:
    if not token_logprobs:
        return None
    return float(-sum(token_logprobs) / len(token_logprobs))


class _RetryableError(Exception):
    pass


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


async def _call_api(api_key: str, prompt: str, seed_hex: str) -> dict[str, Any]:
    """Single chat completion for QwQ-32B with logprobs."""
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=30.0, pool=5.0
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_QWQ,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "seed": int(seed_hex[:8], 16),
        "logprobs": 1,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{TOGETHER_API_BASE}/chat/completions", headers=headers, json=payload
        )
    if resp.status_code in (429, 500, 502, 503, 504):
        raise _RetryableError(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


async def _call_with_retry(
    api_key: str, prompt: str, seed_hex: str
) -> dict[str, Any] | None:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await _call_api(api_key, prompt, seed_hex)
        except _RetryableError as exc:
            if attempt == MAX_RETRIES:
                logger.error("Exhausted %d retries (%s) -- skipping", MAX_RETRIES, exc)
                return None
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning("Attempt %d: %s; retrying in %.1fs", attempt + 1, exc, delay)
            await asyncio.sleep(delay)
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            if attempt == MAX_RETRIES:
                logger.error(
                    "%s after %d retries -- skipping", type(exc).__name__, MAX_RETRIES
                )
                return None
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "%s (attempt %d/%d); retrying in %.1fs",
                type(exc).__name__,
                attempt + 1,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)
    return None


# ---------------------------------------------------------------------------
# B1: Smoke test
# ---------------------------------------------------------------------------


async def smoke_test(api_key: str) -> None:
    """One call to verify model ID and answer extraction."""
    from pilot.prompts import _LOCKED_PROMPT_HASH, PROMPT_TEMPLATE_HASH, render_prompt
    from pilot.scoring import extract_answer

    if PROMPT_TEMPLATE_HASH != _LOCKED_PROMPT_HASH:
        raise RuntimeError("Prompt hash mismatch.")
    logger.info("Prompt hash verified: %s...", PROMPT_TEMPLATE_HASH[:16])

    problems = _load_exploratory_problems()
    prob = problems[0]

    prompt = render_prompt(
        question_text=prob.question,
        option_a=prob.option_a,
        option_b=prob.option_b,
        option_c=prob.option_c,
        option_d=prob.option_d,
    )

    seed_hex = "deadbeef12345678"
    logger.info("B1: Smoke test -- calling %s...", MODEL_QWQ)
    t0 = time.monotonic()
    data = await _call_with_retry(api_key, prompt, seed_hex)
    elapsed = time.monotonic() - t0

    if data is None:
        raise RuntimeError("Smoke test: API call returned None")

    choice = data["choices"][0]
    text = choice["message"]["content"]
    usage = data.get("usage", {})
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    cost_usd = in_tok * QWQ_PRICE_INPUT + out_tok * QWQ_PRICE_OUTPUT

    extraction = extract_answer(text)
    raw_lp = choice.get("logprobs")
    token_lps = _extract_token_logprobs(raw_lp)
    mean_ent = _mean_entropy(token_lps)

    print("\n" + "=" * 70)
    print("B1: QwQ-32B SMOKE TEST")
    print("=" * 70)
    print(f"Model ID:          {MODEL_QWQ}")
    print(f"Problem ID:        {prob.id}")
    print(f"Input tokens:      {in_tok}")
    print(f"Output tokens:     {out_tok}")
    print(f"Cost (this call):  ${cost_usd:.4f}")
    print(f"Latency:           {elapsed:.1f}s")
    print(f"Logprobs:          {'YES' if token_lps is not None else 'NO'}")
    print(
        f"Mean entropy:      {mean_ent:.4f}" if mean_ent else "Mean entropy:      N/A"
    )
    print(f"Extracted answer:  {extraction.answer} (pass {extraction.pass_number})")
    print(f"Ground truth:      {prob.ground_truth}")
    print(f"Correct:           {extraction.answer == prob.ground_truth}")
    print("\nResponse snippet (first 500 chars):")
    print(text[:500])
    print("..." if len(text) > 500 else "")
    print("=" * 70)

    # B2: estimate cost for N_PER_PROBLEM=16 on 47 problems
    # Use this single sample as the token estimate
    avg_in = in_tok
    avg_out = out_tok
    total_samples = 47 * N_PER_PROBLEM
    est_cost = total_samples * (avg_in * QWQ_PRICE_INPUT + avg_out * QWQ_PRICE_OUTPUT)
    print(f"\nB2: Cost estimate for 47 problems x N={N_PER_PROBLEM}:")
    print(f"  Avg input tokens:   {avg_in}")
    print(f"  Avg output tokens:  {avg_out}")
    print(f"  Total samples:      {total_samples}")
    print(f"  Estimated cost:     ${est_cost:.2f}")
    print(f"  Hard cap:           ${COST_ABORT_USD:.2f}")
    if est_cost > COST_ABORT_USD:
        safe_n = int(
            COST_ABORT_USD
            / (47 * (avg_in * QWQ_PRICE_INPUT + avg_out * QWQ_PRICE_OUTPUT))
            * N_PER_PROBLEM
        )
        print(f"  WARNING: estimate exceeds cap. Reduce N to {safe_n}.")
    else:
        print(f"  STATUS: within budget. Proceeding with N={N_PER_PROBLEM}.")


# ---------------------------------------------------------------------------
# B3: Sampling
# ---------------------------------------------------------------------------


async def _sample_problem_qwq(
    problem: Any,
    api_key: str,
    semaphore: asyncio.Semaphore,
    extract_fn: Any,
    prompt_hash: str,
    cost_tracker: dict[str, float],
) -> dict[str, Any]:
    """Sample up to N_PER_PROBLEM completions idempotently."""
    from pilot.prompts import render_prompt

    out_path = OUT_DIR_QWQ / f"{problem.id}.jsonl"
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

    with out_path.open("a") as fh:
        for _ in range(needed):
            # Cost guard before each sample
            if cost_tracker["total_usd"] >= COST_ABORT_USD:
                logger.error(
                    "ABORT: cost $%.4f >= hard cap $%.2f",
                    cost_tracker["total_usd"],
                    COST_ABORT_USD,
                )
                break

            seed_hex = secrets.token_bytes(8).hex()
            async with semaphore:
                t0 = time.monotonic()
                data = await _call_with_retry(api_key, prompt, seed_hex)
                latency_ms = (time.monotonic() - t0) * 1000

            if data is None:
                logger.error("%s: sample skipped (API failure)", problem.id)
                continue

            choice = data["choices"][0]
            text = choice["message"]["content"]
            usage = data.get("usage", {})
            in_tok = int(usage.get("prompt_tokens", 0))
            out_tok = int(usage.get("completion_tokens", 0))

            sample_cost = in_tok * QWQ_PRICE_INPUT + out_tok * QWQ_PRICE_OUTPUT
            cost_tracker["total_usd"] += sample_cost

            raw_lp = choice.get("logprobs")
            token_lps = _extract_token_logprobs(raw_lp)
            mean_ent = _mean_entropy(token_lps)

            extraction = extract_fn(text)
            is_correct = (
                extraction.answer == problem.ground_truth
                if extraction.answer is not None
                else False
            )
            extracted = extraction.answer if extraction.answer else "UNPARSEABLE"

            record = {
                "schema_version": SCHEMA_VERSION,
                "problem_id": problem.id,
                "subject": problem.subject,
                "ground_truth": problem.ground_truth,
                "model": MODEL_QWQ,
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
                "mean_token_entropy": mean_ent,
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
        "skipped": existing >= N_PER_PROBLEM,
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
    }


async def run_sampling(api_key: str) -> None:
    """B3: Sample all 47 exploratory problems."""
    from pilot.prompts import _LOCKED_PROMPT_HASH, PROMPT_TEMPLATE_HASH
    from pilot.scoring import extract_answer

    if PROMPT_TEMPLATE_HASH != _LOCKED_PROMPT_HASH:
        raise RuntimeError("Prompt hash mismatch.")

    problems = _load_exploratory_problems()
    logger.info(
        "B3: Sampling %d exploratory problems x N=%d with %s",
        len(problems),
        N_PER_PROBLEM,
        MODEL_QWQ,
    )

    OUT_DIR_QWQ.mkdir(parents=True, exist_ok=True)

    cost_tracker: dict[str, float] = {"total_usd": 0.0}
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        _sample_problem_qwq(
            p, api_key, semaphore, extract_answer, PROMPT_TEMPLATE_HASH, cost_tracker
        )
        for p in problems
    ]

    completed = 0
    total_new = 0
    total_in_tok = 0
    total_out_tok = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        total_new += result.get("new", 0)
        total_in_tok += result.get("total_input_tokens", 0)
        total_out_tok += result.get("total_output_tokens", 0)

        if completed % CHECKPOINT_INTERVAL == 0 or completed == len(problems):
            logger.info(
                "Progress: %d/%d problems done, %d new samples, cost=$%.4f",
                completed,
                len(problems),
                total_new,
                cost_tracker["total_usd"],
            )

    logger.info(
        "B3 complete: %d new samples, %d input tokens, %d output tokens, total cost=$%.4f",
        total_new,
        total_in_tok,
        total_out_tok,
        cost_tracker["total_usd"],
    )


# ---------------------------------------------------------------------------
# B4: Analysis helpers
# ---------------------------------------------------------------------------


def _plurality(answers: list[str]) -> str:
    """Return the plurality answer; ties broken by first-encountered order."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for a in answers:
        if a not in counts:
            order.append(a)
        counts[a] = counts.get(a, 0) + 1
    return max(order, key=lambda x: counts[x])


def _mv_curve(
    answers: list[str], ground_truth: str, seed: int = _SEED_MV
) -> dict[str, float]:
    """Compute expected majority-vote accuracy for each N in _N_VALUES."""
    from math import comb

    curve: dict[str, float] = {}
    n_total = len(answers)
    for n in _N_VALUES:
        if n > n_total:
            break
        n_combos = comb(n_total, n)
        if n_combos <= 5000:
            # Exact enumeration
            from itertools import combinations

            total = 0
            correct = 0
            for subset in combinations(range(n_total), n):
                sub = [answers[i] for i in subset]
                if _plurality(sub) == ground_truth:
                    correct += 1
                total += 1
            curve[str(n)] = correct / total if total else 0.0
        else:
            rng = np.random.default_rng(seed)
            draws = [
                rng.choice(n_total, size=n, replace=False).tolist()
                for _ in range(_N_MC)
            ]
            correct = sum(
                1 for d in draws if _plurality([answers[i] for i in d]) == ground_truth
            )
            curve[str(n)] = correct / _N_MC
    return curve


def _bootstrap_rate(
    values: list[float], n_boot: int = _BOOTSTRAP_N, seed: int = _BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Return (lower_95, upper_95) CI for mean of values."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boot_means = [
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)
    ]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _oracle_acc(
    curves: list[dict[str, float]], n1: int = 1, n2: int = 16
) -> tuple[float, float]:
    """Binary oracle: max(MV_acc(n1), MV_acc(n2)) per problem, averaged."""
    accs = []
    computes = []
    for c in curves:
        a1 = c.get(str(n1), 0.0)
        a2 = c.get(str(n2), 0.0)
        if a1 >= a2:
            accs.append(a1)
            computes.append(n1)
        else:
            accs.append(a2)
            computes.append(n2)
    return float(np.mean(accs)), float(np.mean(computes))


def _agreement_gate_sweep(
    answers_list: list[list[str]],
    ground_truths: list[str],
    mv64_accs: list[float],
    k_values: list[int],
    taus: list[float],
    rng_seed: int = 44,
) -> list[dict[str, Any]]:
    """Sweep (k, tau) for agreement gate. Returns list of result dicts."""
    n_problems = len(answers_list)
    rng = np.random.default_rng(rng_seed)
    results = []
    for k in k_values:
        n_draws = _N_MC
        # Precompute probe draws per problem
        probe_draws: list[np.ndarray] = []
        probe_plurality_correct: list[np.ndarray] = []
        probe_fractions: list[np.ndarray] = []
        for i, (answers, gt) in enumerate(zip(answers_list, ground_truths)):
            n_total = len(answers)
            if n_total < k:
                probe_draws.append(None)
                probe_plurality_correct.append(None)
                probe_fractions.append(None)
                continue
            idxs = rng.choice(n_total, size=(n_draws, k), replace=True)
            ans_arr = np.array(answers)
            fracs = np.array(
                [
                    max(np.sum(ans_arr[row] == a) for a in set(ans_arr[row])) / k
                    for row in idxs
                ]
            )
            correct = np.array(
                [int(_plurality(list(ans_arr[row])) == gt) for row in idxs]
            )
            probe_draws.append(idxs)
            probe_plurality_correct.append(correct)
            probe_fractions.append(fracs)

        for tau in taus:
            gate_accs = []
            gate_cmps = []
            for i in range(n_problems):
                if probe_fractions[i] is None:
                    gate_accs.append(mv64_accs[i])
                    gate_cmps.append(N_PER_PROBLEM)
                    continue
                fracs = probe_fractions[i]
                correct = probe_plurality_correct[i]
                low_mask = fracs >= tau
                prob_low = low_mask.mean()
                acc_low = correct[low_mask].mean() if low_mask.any() else 0.0
                gate_acc = prob_low * acc_low + (1 - prob_low) * mv64_accs[i]
                gate_acc_cmp = prob_low * k + (1 - prob_low) * N_PER_PROBLEM
                gate_accs.append(gate_acc)
                gate_cmps.append(gate_acc_cmp)

            best_oracle, _ = _oracle_acc(
                [
                    {"1": a, str(N_PER_PROBLEM): b}
                    for a, b in zip([0.0] * n_problems, mv64_accs)
                ]
            )
            fixed64_acc = float(np.mean(mv64_accs))
            oracle_acc, _ = _oracle_acc(
                [
                    {"1": mv64_accs[i], str(N_PER_PROBLEM): mv64_accs[i]}
                    for i in range(n_problems)
                ]
            )
            gate_acc_mean = float(np.mean(gate_accs))
            gate_cmp_mean = float(np.mean(gate_cmps))
            results.append(
                {
                    "k": k,
                    "tau": tau,
                    "gate_acc": gate_acc_mean,
                    "gate_compute": gate_cmp_mean,
                    "fixed64_acc": fixed64_acc,
                }
            )
    return results


# ---------------------------------------------------------------------------
# B4-B5: Full analysis and figure
# ---------------------------------------------------------------------------


def run_analysis() -> None:
    """B4: Compute probe metrics. B5: Save figure."""
    import matplotlib

    matplotlib.use("Agg")

    from pilot.prompts import _LOCKED_PROMPT_HASH, PROMPT_TEMPLATE_HASH

    if PROMPT_TEMPLATE_HASH != _LOCKED_PROMPT_HASH:
        raise RuntimeError("Prompt hash mismatch.")

    logger.info("B4: Loading QwQ probe samples...")
    problems = _load_exploratory_problems()
    pid_to_problem = {p.id: p for p in problems}

    samples_by_pid: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(OUT_DIR_QWQ.glob("*.jsonl")):
        pid = path.stem
        records = []
        with path.open() as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        samples_by_pid[pid] = records

    if not samples_by_pid:
        logger.error("No QwQ samples found in %s", OUT_DIR_QWQ)
        return

    pids = sorted(samples_by_pid.keys())
    logger.info("Loaded %d problems with QwQ samples", len(pids))

    # Build answer lists and ground truths
    answers_list: list[list[str]] = []
    ground_truths: list[str] = []
    parse_rates: list[float] = []
    entropy_means: list[float | None] = []

    for pid in pids:
        records = samples_by_pid[pid]
        prob = pid_to_problem[pid]
        valid = [
            r["extracted_answer"]
            for r in records
            if r.get("extracted_answer") and r["extracted_answer"] != "UNPARSEABLE"
        ]
        answers_list.append(valid)
        ground_truths.append(prob.ground_truth)
        parse_rates.append(len(valid) / len(records) if records else 0.0)
        ents = [
            r["mean_token_entropy"]
            for r in records
            if r.get("mean_token_entropy") is not None
        ]
        entropy_means.append(float(np.mean(ents)) if ents else None)

    n_problems = len(pids)
    logger.info(
        "Parse rate: %.3f (mean across problems)",
        float(np.mean(parse_rates)),
    )

    # MV curves — skip problems with <2 valid answers (can't compute MV)
    logger.info("Computing MV curves...")
    curves: list[dict[str, float]] = []
    valid_mask: list[bool] = []
    for answers, gt in zip(answers_list, ground_truths):
        if len(answers) >= 2:
            curves.append(_mv_curve(answers, gt))
            valid_mask.append(True)
        else:
            curves.append({})
            valid_mask.append(False)

    n_valid = sum(valid_mask)
    n_skipped = n_problems - n_valid
    if n_skipped:
        logger.warning(
            "%d/%d problems skipped (< 2 valid answers); analysis on %d problems",
            n_skipped,
            n_problems,
            n_valid,
        )

    valid_curves = [c for c, ok in zip(curves, valid_mask) if ok]
    valid_gts = [gt for gt, ok in zip(ground_truths, valid_mask) if ok]
    valid_answers = [a for a, ok in zip(answers_list, valid_mask) if ok]
    valid_mv16 = [
        c.get(str(N_PER_PROBLEM), c.get(str(max(_N_VALUES)), 0.0)) for c in valid_curves
    ]

    # Backfire rate
    mv_gains = [
        c.get(str(N_PER_PROBLEM), c.get(str(max(_N_VALUES)), 0.0)) - c.get("1", 0.0)
        for c in valid_curves
    ]
    backfire_flags = [1 if g < 0 else 0 for g in mv_gains]
    backfire_rate = float(np.mean(backfire_flags)) if backfire_flags else 0.0
    bf_ci_lo, bf_ci_hi = (
        _bootstrap_rate(backfire_flags) if len(backfire_flags) > 1 else (0.0, 0.0)
    )

    # MV acc at N=8 and N=1
    mv1_accs = [c.get("1", 0.0) for c in valid_curves]
    mv16_accs = valid_mv16
    mv1_mean = float(np.mean(mv1_accs)) if mv1_accs else 0.0
    mv16_mean = float(np.mean(mv16_accs)) if mv16_accs else 0.0
    mv_gain_mean = mv16_mean - mv1_mean

    # Grid oracle (over available N values)
    grid_oracle_per_problem = [
        max((c.get(str(n), 0.0) for n in _N_VALUES), default=0.0) for c in valid_curves
    ]
    grid_oracle_acc = (
        float(np.mean(grid_oracle_per_problem)) if grid_oracle_per_problem else 0.0
    )
    grid_oracle_best_n = [
        max(
            (n for n in _N_VALUES if str(n) in c),
            key=lambda n: c.get(str(n), 0.0),
            default=1,
        )
        for c in valid_curves
    ]
    grid_oracle_compute = (
        float(np.mean(grid_oracle_best_n)) if grid_oracle_best_n else 1.0
    )

    binary_oracle_acc, binary_oracle_compute = _oracle_acc(valid_curves)

    # Agreement gate sweep (k=4 only since N_PER_PROBLEM=8)
    logger.info("Running agreement gate sweep...")
    gate_results = _agreement_gate_sweep(
        valid_answers, valid_gts, mv16_accs, _AGREEMENT_K_VALUES, _AGREEMENT_TAUS
    )

    # Best gate result (k=4, highest k available)
    k4_results = [r for r in gate_results if r["k"] == _AGREEMENT_K_VALUES[-1]]
    best_k8 = (
        max(k4_results, key=lambda r: r["gate_acc"])
        if k4_results
        else {"gate_acc": mv16_mean, "k": 4, "tau": 0.5}
    )
    agree_ceiling_captured = (
        (best_k8["gate_acc"] - mv16_mean) / (binary_oracle_acc - mv16_mean)
        if binary_oracle_acc > mv16_mean
        else 0.0
    )

    # Calibration
    conf_bins = [(0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    calibration: list[dict[str, Any]] = []
    for lo, hi in conf_bins:
        bin_correct = []
        for answers, gt in zip(valid_answers, valid_gts):
            if not answers:
                continue
            counts: dict[str, int] = {}
            for a in answers:
                counts[a] = counts.get(a, 0) + 1
            top = max(counts, key=lambda x: counts[x])
            frac = counts[top] / len(answers)
            if lo <= frac < hi:
                bin_correct.append(1 if top == gt else 0)
        calibration.append(
            {
                "bin": f"[{lo:.2f}, {hi:.2f})",
                "n": len(bin_correct),
                "frac_correct": float(np.mean(bin_correct)) if bin_correct else None,
            }
        )

    # Token usage summary
    all_in_tok = sum(
        r.get("input_tokens", 0) for records in samples_by_pid.values() for r in records
    )
    all_out_tok = sum(
        r.get("output_tokens", 0)
        for records in samples_by_pid.values()
        for r in records
    )
    total_cost = all_in_tok * QWQ_PRICE_INPUT + all_out_tok * QWQ_PRICE_OUTPUT

    # Save analysis results
    OUT_DIR_GATE.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR_GATE / "qwq_probe_results.json"
    results = {
        "EXPLORATORY": True,
        "NOTE": (
            "All Phase 14b QwQ-32B results are post-hoc EXPLORATORY. "
            "Pre-registered PH1-PH4 on confirmatory set are unchanged."
        ),
        "model": MODEL_QWQ,
        "n_problems_total": n_problems,
        "n_problems_analyzed": n_valid,
        "n_problems_skipped_low_parse": n_skipped,
        "n_per_problem": N_PER_PROBLEM,
        "temperature": TEMPERATURE,
        "parse_rate_mean": float(np.mean(parse_rates)),
        "mv1_acc": mv1_mean,
        "mv16_acc": mv16_mean,
        "mv_gain": mv_gain_mean,
        "backfire_rate": backfire_rate,
        "backfire_ci_95": [bf_ci_lo, bf_ci_hi],
        "binary_oracle_acc": binary_oracle_acc,
        "binary_oracle_compute": binary_oracle_compute,
        "grid_oracle_acc": grid_oracle_acc,
        "grid_oracle_compute": grid_oracle_compute,
        "agree_gate_best_k8": best_k8,
        "agree_gate_ceiling_captured": agree_ceiling_captured,
        "calibration": calibration,
        "token_usage": {
            "total_input_tokens": all_in_tok,
            "total_output_tokens": all_out_tok,
            "total_cost_usd": total_cost,
        },
        "gate_sweep": gate_results,
    }
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved %s", results_path)

    # B5: Figure
    _make_figure(
        valid_curves,
        mv_gains,
        backfire_rate,
        bf_ci_lo,
        bf_ci_hi,
        calibration,
        gate_results,
        binary_oracle_acc,
        mv16_mean,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("[EXPLORATORY] Phase 14b: QwQ-32B Probe Summary")
    print("=" * 70)
    print(
        f"Problems: {n_valid}/{n_problems} analyzable, N={N_PER_PROBLEM}, T={TEMPERATURE}"
    )
    print(f"Parse rate: {np.mean(parse_rates):.3f}")
    print(f"MV acc(1):  {mv1_mean:.4f}")
    print(f"MV acc(16): {mv16_mean:.4f}")
    print(f"MV gain:    {mv_gain_mean:+.4f}")
    print(f"Backfire:   {backfire_rate:.3f} [{bf_ci_lo:.3f}, {bf_ci_hi:.3f}] 95% CI")
    print(
        f"Binary oracle acc: {binary_oracle_acc:.4f} (compute={binary_oracle_compute:.1f})"
    )
    print(
        f"Grid oracle acc:   {grid_oracle_acc:.4f} (compute={grid_oracle_compute:.1f})"
    )
    print(f"Agree gate k=8 best ceil: {agree_ceiling_captured:.4f}")
    print("\nCalibration:")
    for b in calibration:
        fc = b["frac_correct"]
        print(
            f"  {b['bin']}  n={b['n']}  frac_correct={fc:.3f}"
            if fc is not None
            else f"  {b['bin']}  n={b['n']}  frac_correct=N/A"
        )
    print(f"\nToken usage: in={all_in_tok}, out={all_out_tok}, cost=${total_cost:.4f}")
    print("=" * 70)


def _make_figure(
    curves: list[dict[str, float]],
    mv_gains: list[float],
    backfire_rate: float,
    bf_ci_lo: float,
    bf_ci_hi: float,
    calibration: list[dict[str, Any]],
    gate_results: list[dict[str, Any]],
    binary_oracle_acc: float,
    mv16_mean: float,
) -> None:
    """B5: Save qwq_probe.png with backfire distribution and calibration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: backfire distribution (mv_gain histogram)
    ax = axes[0]
    gains = np.array(mv_gains)
    ax.bar(
        range(len(gains)),
        sorted(gains),
        color=["#c0392b" if g < 0 else "#2980b9" for g in sorted(gains)],
        width=1.0,
        edgecolor="none",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Problem (sorted by mv_gain)")
    ax.set_ylabel("mv_gain = MV_acc(16) - MV_acc(1)")
    ax.set_title(
        f"[EXPLORATORY] QwQ-32B Backfire\n"
        f"Backfire rate: {backfire_rate:.1%} [{bf_ci_lo:.1%}, {bf_ci_hi:.1%}]"
    )
    ax.tick_params(labelbottom=False)

    # Right: calibration
    ax = axes[1]
    bin_labels = [b["bin"] for b in calibration]
    frac_corrects = [
        b["frac_correct"] if b["frac_correct"] is not None else 0.0 for b in calibration
    ]
    ns = [b["n"] for b in calibration]
    ax.bar(range(len(bin_labels)), frac_corrects, color="#2980b9", width=0.6)
    ax.plot(
        [0, len(bin_labels) - 1],
        [0.25, 1.0],
        "k--",
        linewidth=0.8,
        label="Perfect cal.",
    )
    for i, (fc, n) in enumerate(zip(frac_corrects, ns)):
        ax.text(i, fc + 0.02, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Fraction correct (plurality answer)")
    ax.set_ylim(0, 1.1)
    ax.set_title("[EXPLORATORY] QwQ-32B Calibration\n(Confidence bin vs accuracy)")

    fig.tight_layout()
    out_path = OUT_DIR_GATE / "qwq_probe.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="Phase 14b: QwQ-32B probe")
    parser.add_argument("--smoke-test", action="store_true", help="B1: one API call")
    parser.add_argument("--sample", action="store_true", help="B3: full sampling")
    parser.add_argument("--analyze", action="store_true", help="B4-B5: analysis+figure")
    parser.add_argument("--all", action="store_true", help="smoke+sample+analyze")
    args = parser.parse_args()

    _load_env()

    do_smoke = args.smoke_test or args.all
    do_sample = args.sample or args.all
    do_analyze = args.analyze or args.all

    if not any([do_smoke, do_sample, do_analyze]):
        parser.print_help()
        sys.exit(1)

    if do_smoke or do_sample:
        api_key = _require_api_key()
        if do_smoke:
            asyncio.run(smoke_test(api_key))
        if do_sample:
            asyncio.run(run_sampling(api_key))

    if do_analyze:
        run_analysis()


if __name__ == "__main__":
    main()
