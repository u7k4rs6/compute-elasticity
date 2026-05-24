"""Phase 9 prep — H3 entropy baseline data collection.

Sends 1 completion per main-pilot problem with logprobs=True (top_logprobs=5),
then computes per-token NLL (negative log-likelihood) as the H3 baseline
feature.  Shannon entropy from top_logprobs is also recorded when Together AI
returns that field non-empty.

Together AI's logprobs schema (differs from OpenAI):
  choices[0].logprobs = {
    "tokens": [...],
    "token_logprobs": [...],   # primary signal: log p of selected token
    "top_logprobs": {} or [{token: logprob, ...}, ...]  # may be empty
  }

Idempotent: skips problems whose output file already exists unless --force.
Use --force to regenerate existing files (e.g. after a schema fix).
Estimated cost: ~$0.01 for 47 problems at T=0.7.

Usage:
    source .venv/bin/activate
    python scripts/sample_for_entropy.py
    python scripts/sample_for_entropy.py --force
    python scripts/sample_for_entropy.py --dry-run
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
        raise RuntimeError(
            "TOGETHER_API_KEY env var is not set. "
            "Add it to your .env file (see .env.example)."
        )
    return key


def _load_locked_ids() -> list[str]:
    path = ROOT / "data" / "problem_ids.json"
    if not path.exists():
        raise FileNotFoundError(f"Locked problem IDs not found: {path}")
    return json.loads(path.read_text())


def _load_gate_ids() -> list[str]:
    path = ROOT / "outputs" / "gate_minus_1_labels.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Gate -1 labels not found: {path}\n"
            "Run scripts/run_gate_minus_1.py first."
        )
    return json.loads(path.read_text())["gate_problems"]


def _main_pilot_ids(locked: list[str], gate: list[str]) -> list[str]:
    gate_set = set(gate)
    return sorted(pid for pid in locked if pid not in gate_set)


# ---------------------------------------------------------------------------
# API call with logprobs
# ---------------------------------------------------------------------------


def _chat_with_logprobs(
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
) -> dict:
    """POST a chat completion with logprobs; return the raw response dict.

    Retries up to _MAX_RETRIES times on transient HTTP errors (429/5xx)
    with exponential backoff + jitter.  Non-retryable errors raise RuntimeError.
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
                    resp.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
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
                attempt + 1,
                _MAX_RETRIES,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Logprobs parsing  (Together AI schema)
# ---------------------------------------------------------------------------
# Together AI response shape:
#   choices[0].logprobs = {
#     "tokens":        ["tok1", "tok2", ...],
#     "token_logprobs": [-0.12, -3.4, ...],   # log p of selected token
#     "top_logprobs":  {} or [{"tok_a": -0.1, "tok_b": -2.3}, ...]
#   }
# top_logprobs is a list of dicts (one dict per token position), where each
# dict maps candidate tokens to their log-probabilities.  Together may return
# an empty dict {} when the field is unsupported for the requested model.


def _parse_together_logprobs(
    choice: dict,
) -> tuple[list[float], list[list[tuple[str, float]]], bool]:
    """Parse Together AI's logprobs field from a chat choice.

    Returns:
        token_lps: per-token log-probabilities of the selected token (primary).
        top_lps_per_token: per-token list of (token, logprob) pairs from
            top_logprobs; empty list if that field was absent or empty.
        top_lps_available: True if top_logprobs contained usable data.
    """
    lp_field = choice.get("logprobs") or {}

    # Primary signal: selected-token log-probabilities
    token_lps: list[float] = [
        float(v) for v in (lp_field.get("token_logprobs") or []) if v is not None
    ]

    # Secondary signal: top-K distribution (may be empty dict or empty list)
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
    """Shannon entropy (nats) of the renormalised top-K distribution."""
    if not top_lps:
        return 0.0
    raw_probs = [math.exp(lp) for _, lp in top_lps]
    total = sum(raw_probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in raw_probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


# ---------------------------------------------------------------------------
# Per-problem processing
# ---------------------------------------------------------------------------


def _median(xs: list[float]) -> float:
    """Return median of a non-empty list."""
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _process_problem(
    problem_id: str,
    subject: str,
    prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    out_dir: Path,
    force: bool = False,
    log_schema: bool = False,
) -> dict | None:
    """Sample one completion for a problem, compute NLL metrics, write output.

    Returns the written result dict, or None if skipped (exists and not --force).
    """
    out_path = out_dir / f"{problem_id}.json"
    if out_path.exists() and not force:
        logger.info("  [skip] %s — output already exists.", problem_id)
        return None

    data = _chat_with_logprobs(api_key, api_base, model, prompt, _TEMPERATURE)
    choice = data["choices"][0]

    if log_schema:
        raw_lp = choice.get("logprobs", "NOT PRESENT")
        logger.info("  [schema] %s logprobs field: %s", problem_id, raw_lp)

    full_response: str = choice["message"]["content"]
    usage = data.get("usage", {})
    input_tokens: int = int(usage.get("prompt_tokens", 0))
    output_tokens: int = int(usage.get("completion_tokens", 0))

    token_lps, top_lps_per_token, top_lps_available = _parse_together_logprobs(choice)

    if not token_lps:
        logger.warning(
            "  %s: token_logprobs absent or empty — NLL will be 0.", problem_id
        )

    # Primary metric: NLL = -log p(selected token)
    nlls = [-lp for lp in token_lps]
    n_tokens = len(nlls)
    mean_nll = float(sum(nlls) / n_tokens) if n_tokens else 0.0
    median_nll = _median(nlls) if n_tokens else 0.0
    max_nll = float(max(nlls)) if nlls else 0.0

    result: dict = {
        "schema_version": "v6.0-pilot",
        "problem_id": problem_id,
        "subject": subject,
        "model": model,
        "temperature": _TEMPERATURE,
        "n_tokens": n_tokens,
        "mean_per_token_nll": mean_nll,
        "median_per_token_nll": median_nll,
        "max_per_token_nll": max_nll,
        "top_logprobs_available": top_lps_available,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "full_response": full_response,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Secondary metric: entropy from top-K distribution (when available)
    if top_lps_available and top_lps_per_token:
        entropies = [_token_entropy(lps) for lps in top_lps_per_token]
        n_e = len(entropies)
        result["mean_per_token_entropy"] = float(sum(entropies) / n_e) if n_e else 0.0
        result["median_per_token_entropy"] = _median(entropies) if n_e else 0.0
        result["max_per_token_entropy"] = float(max(entropies)) if entropies else 0.0

    out_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "  [done] %s  n_tokens=%d  mean_NLL=%.4f  top_lps=%s",
        problem_id,
        n_tokens,
        mean_nll,
        top_lps_available,
    )
    return result


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_summary(results: list[dict], outputs_dir: Path) -> dict:
    """Write entropy_baseline_summary.json; return the summary dict."""
    means_nll = [r["mean_per_token_nll"] for r in results]
    n_tokens_list = [r["n_tokens"] for r in results]
    n_top_lps = sum(1 for r in results if r.get("top_logprobs_available"))

    def _q(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        idx = q * (len(s) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        return s[lo] + (idx - lo) * (s[hi] - s[lo])

    summary: dict = {
        "schema_version": "v6.0-pilot",
        "n_problems": len(results),
        "mean_nll": {
            "min": min(means_nll),
            "p25": _q(means_nll, 0.25),
            "median": _q(means_nll, 0.50),
            "p75": _q(means_nll, 0.75),
            "max": max(means_nll),
        },
        "n_tokens": {
            "min": min(n_tokens_list),
            "median": _q([float(x) for x in n_tokens_list], 0.50),
            "max": max(n_tokens_list),
        },
        "n_problems_with_top_logprobs": n_top_lps,
        "problem_ids": [r["problem_id"] for r in results],
    }

    # Include entropy stats only if top_logprobs were available for any problem
    if n_top_lps:
        means_h = [
            r["mean_per_token_entropy"]
            for r in results
            if "mean_per_token_entropy" in r
        ]
        summary["mean_entropy"] = {
            "min": min(means_h),
            "p25": _q(means_h, 0.25),
            "median": _q(means_h, 0.50),
            "p75": _q(means_h, 0.75),
            "max": max(means_h),
        }

    path = outputs_dir / "entropy_baseline_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", path)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 entropy baseline sampling")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making any API calls",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-sample even when output file already exists (overwrites)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _load_env()

    from pilot.config import (
        MODEL,
        OUTPUTS_DIR,
        TOGETHER_API_BASE,
        TOGETHER_INPUT_PRICE_PER_TOKEN,
        TOGETHER_OUTPUT_PRICE_PER_TOKEN,
    )
    from pilot.data_loader import load_gpqa_diamond
    from pilot.prompts import render_prompt

    if not args.dry_run:
        api_key = _require_api_key()
    else:
        api_key = "dry-run-placeholder"

    locked_ids = _load_locked_ids()
    gate_ids = _load_gate_ids()
    main_ids = _main_pilot_ids(locked_ids, gate_ids)

    if len(main_ids) != 47:
        logger.error("Expected 47 main-pilot IDs, got %d.", len(main_ids))
        sys.exit(1)

    all_problems = load_gpqa_diamond()
    problem_map = {p.id: p for p in all_problems}

    missing = [pid for pid in main_ids if pid not in problem_map]
    if missing:
        logger.error("Problems not found in dataset: %s", missing)
        sys.exit(1)

    out_dir = OUTPUTS_DIR / "entropy_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    already_done = sum(1 for pid in main_ids if (out_dir / f"{pid}.json").exists())
    to_do = len(main_ids) if args.force else len(main_ids) - already_done

    # Rough cost estimate: ~800 output tokens @ Together rate per problem
    est_cost = (
        to_do * 800 * TOGETHER_OUTPUT_PRICE_PER_TOKEN
        + to_do * 400 * TOGETHER_INPUT_PRICE_PER_TOKEN
    )

    sep = "=" * 68
    print(f"\n{sep}")
    print("H3 Entropy Baseline — sampling plan")
    print(sep)
    print(f"  Model              : {MODEL}")
    print(f"  Temperature        : {_TEMPERATURE}")
    print(f"  top_logprobs       : {_TOP_LOGPROBS}")
    print(f"  Problems total     : {len(main_ids)}")
    print(f"  Already done       : {already_done}")
    print(
        f"  To sample          : {to_do}  {'(--force: will overwrite)' if args.force else ''}"
    )
    print(f"  Estimated cost     : ${est_cost:.4f}")
    print(f"  Output dir         : {out_dir.relative_to(ROOT)}")
    if args.dry_run:
        print("\n  DRY RUN — no API calls will be made.")
    print(sep)

    if args.dry_run:
        for pid in main_ids:
            p = problem_map[pid]
            exists = (out_dir / f"{pid}.json").exists()
            if args.force:
                status = "overwrite" if exists else "sample"
            else:
                status = "skip" if exists else "sample"
            print(f"  [{status}] {pid}  ({p.subject})")
        print(sep)
        return

    results: list[dict] = []
    total_cost = 0.0

    for num, pid in enumerate(main_ids, start=1):
        p = problem_map[pid]
        prompt = render_prompt(
            question_text=p.question,
            option_a=p.option_a,
            option_b=p.option_b,
            option_c=p.option_c,
            option_d=p.option_d,
        )
        logger.info("[%d/%d] %s (%s)…", num, len(main_ids), pid, p.subject)

        result = _process_problem(
            problem_id=pid,
            subject=p.subject,
            prompt=prompt,
            api_key=api_key,
            api_base=TOGETHER_API_BASE,
            model=MODEL,
            out_dir=out_dir,
            force=args.force,
            log_schema=(num == 1),  # log raw logprobs structure for first problem only
        )

        if result is not None:
            problem_cost = (
                result["input_tokens"] * TOGETHER_INPUT_PRICE_PER_TOKEN
                + result["output_tokens"] * TOGETHER_OUTPUT_PRICE_PER_TOKEN
            )
            total_cost += problem_cost
            results.append(result)
        else:
            # Skipped (exists, no --force) — load existing for summary
            results.append(json.loads((out_dir / f"{pid}.json").read_text()))

    summary = _write_summary(results, OUTPUTS_DIR)

    print(f"\n{sep}")
    print("H3 Entropy Baseline — Complete")
    print(sep)
    print(f"  Problems in summary : {len(results)}")
    print(f"  Cost this run       : ${total_cost:.4f}")
    print()
    m = summary["mean_nll"]
    print("  Mean per-token NLL across problems:")
    print(f"    min    : {m['min']:.4f} nats")
    print(f"    p25    : {m['p25']:.4f} nats")
    print(f"    median : {m['median']:.4f} nats")
    print(f"    p75    : {m['p75']:.4f} nats")
    print(f"    max    : {m['max']:.4f} nats")
    if "mean_entropy" in summary:
        e = summary["mean_entropy"]
        print()
        print("  Mean per-token entropy (top_logprobs available):")
        print(f"    median : {e['median']:.4f} nats")
    print()
    print(
        f"  Written to: {(OUTPUTS_DIR / 'entropy_baseline_summary.json').relative_to(ROOT)}"
    )
    print(sep)


if __name__ == "__main__":
    main()
