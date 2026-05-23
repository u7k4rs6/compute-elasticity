"""Phase 6 — Main pilot sampling.

Samples N=64 completions per problem on all 47 main-pilot problems at T=0.7.
The N ∈ {1, 2, 4, 8, 16, 32, 64} analysis grid is derived by subsampling
from the 64 collected here.

Idempotent: reuses existing samples from extended recon (1 per problem already
exists). Progress is checkpointed to outputs/phase_6_progress.json and committed
to git every 5 problems. Ctrl+C saves state and allows clean resumption.

Usage:
    source .venv/bin/activate
    python scripts/run_main_pilot.py             # live run
    python scripts/run_main_pilot.py --dry-run   # plan without API calls
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_N_MAX: int = 64
_TEMPERATURE: float = 0.7
_INITIAL_COST_ESTIMATE: float = 0.30  # conservative per-problem estimate (USD)
_PROGRESS_FILENAME: str = "phase_6_progress.json"

# Tokens-per-completion assumed for dry-run estimate when no recon data available
_DRY_RUN_INPUT_TOKENS: int = 400
_DRY_RUN_OUTPUT_TOKENS: int = 800

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment / ID helpers (same pattern as run_recon_extended.py)
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


def _load_locked_ids() -> list[str]:
    ids_path = ROOT / "data" / "problem_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(f"Locked problem IDs not found: {ids_path}")
    return json.loads(ids_path.read_text())


def _load_gate_problem_ids() -> list[str]:
    labels_path = ROOT / "outputs" / "gate_minus_1_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Gate -1 labels not found: {labels_path}\n"
            "Run scripts/run_gate_minus_1.py first."
        )
    return json.loads(labels_path.read_text())["gate_problems"]


def _select_main_pilot_problems(
    all_problems: list[Any],
    gate_ids: list[str],
    locked_ids: list[str],
) -> list[Any]:
    """Return the 47 main-pilot problems: locked 50 minus 3 Gate-1, sorted by ID."""
    locked = set(locked_ids)
    gate = set(gate_ids)
    problems = [p for p in all_problems if p.id in locked and p.id not in gate]
    problems.sort(key=lambda p: p.id)
    return problems


# ---------------------------------------------------------------------------
# Sample counting (temperature-aware, same as run_recon_extended.py)
# ---------------------------------------------------------------------------


def _count_samples_at_temperature(
    samples_dir: Path, problem_id: str, temperature: float
) -> int:
    """Count JSONL lines whose temperature field matches."""
    path = samples_dir / f"{problem_id}.jsonl"
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            if abs(json.loads(line).get("temperature", -1) - temperature) < 1e-6:
                count += 1
        except json.JSONDecodeError:
            continue
    return count


# ---------------------------------------------------------------------------
# Progress file
# ---------------------------------------------------------------------------


def _init_progress(
    problems: list[Any],
    existing_counts: dict[str, int],
    schema_version: str,
) -> dict[str, Any]:
    """Build a fresh progress structure."""
    return {
        "schema_version": schema_version,
        "phase": "main_pilot",
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_problems_total": len(problems),
        "n_problems_complete": sum(
            1 for p in problems if existing_counts.get(p.id, 0) >= _N_MAX
        ),
        "per_problem": [
            {
                "problem_id": p.id,
                "subject": p.subject,
                "n_samples": existing_counts.get(p.id, 0),
                "complete": existing_counts.get(p.id, 0) >= _N_MAX,
                "cost_usd": 0.0,
            }
            for p in problems
        ],
        "cumulative_cost_usd": 0.0,
        "halted": False,
        "halt_reason": None,
    }


def _load_or_init_progress(
    outputs_dir: Path,
    problems: list[Any],
    existing_counts: dict[str, int],
    schema_version: str,
) -> dict[str, Any]:
    """Load existing progress file if valid; otherwise initialise fresh."""
    path = outputs_dir / _PROGRESS_FILENAME
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("phase") == "main_pilot" and data.get(
                "n_problems_total"
            ) == len(problems):
                logger.info(
                    "Resuming from progress file: %d/%d complete, $%.4f spent.",
                    data.get("n_problems_complete", 0),
                    len(problems),
                    data.get("cumulative_cost_usd", 0.0),
                )
                return data
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Progress file malformed — starting fresh.")
    return _init_progress(problems, existing_counts, schema_version)


def _save_progress(outputs_dir: Path, progress: dict[str, Any]) -> None:
    """Atomically update last_updated and write progress to disk."""
    progress["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path = outputs_dir / _PROGRESS_FILENAME
    path.write_text(json.dumps(progress, indent=2))


# ---------------------------------------------------------------------------
# Git checkpoint
# ---------------------------------------------------------------------------


def _git_checkpoint(
    outputs_dir: Path,
    n_complete: int,
    n_total: int,
    cost: float,
) -> None:
    """Stage progress.json and commit a checkpoint (no push)."""
    progress_path = outputs_dir / _PROGRESS_FILENAME
    try:
        # Check if file has unstaged changes or is untracked
        diff = subprocess.run(
            ["git", "status", "--porcelain", str(progress_path)],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if not diff.stdout.strip():
            return  # nothing to commit
        subprocess.run(
            ["git", "add", str(progress_path)],
            check=True,
            capture_output=True,
            cwd=ROOT,
        )
        msg = (
            f"phase-6-progress: {n_complete}/{n_total} problems sampled"
            f" (cost: ${cost:.2f})"
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            check=True,
            capture_output=True,
            cwd=ROOT,
        )
        logger.info("Git checkpoint: %s", msg)
    except subprocess.CalledProcessError as exc:
        logger.warning("Git checkpoint failed: %s", exc.stderr.decode().strip())


# ---------------------------------------------------------------------------
# Cost estimation helpers
# ---------------------------------------------------------------------------


def _estimate_cost_per_token(
    input_price: float, output_price: float
) -> tuple[float, float]:
    """Return (input_usd_per_token, output_usd_per_token) — trivial pass-through."""
    return input_price, output_price


def _try_load_recon_cost_per_sample(outputs_dir: Path) -> float | None:
    """Read cost-per-new-sample from recon_extended_results.json if available."""
    path = outputs_dir / "recon_extended_results.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        n_new = data.get("n_newly_sampled", 0)
        cost = data.get("new_sampling_cost_usd", 0.0)
        if n_new > 0 and cost > 0:
            return cost / n_new
    except (json.JSONDecodeError, KeyError, TypeError, ZeroDivisionError):
        pass
    return None


# ---------------------------------------------------------------------------
# Dry-run plan printer
# ---------------------------------------------------------------------------


def _print_dry_run(
    problems: list[Any],
    existing_counts: dict[str, int],
    input_price: float,
    output_price: float,
    outputs_dir: Path,
) -> None:
    """Print sampling plan and estimated cost without making any API calls."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("DRY RUN — Phase 6 Main Pilot Sampling Plan")
    print(sep)

    recon_cost_per_sample = _try_load_recon_cost_per_sample(outputs_dir)
    if recon_cost_per_sample is not None:
        cost_label = f"recon-derived (${recon_cost_per_sample:.5f}/sample)"
    else:
        tok_cost = (
            _DRY_RUN_INPUT_TOKENS * input_price + _DRY_RUN_OUTPUT_TOKENS * output_price
        )
        recon_cost_per_sample = tok_cost
        cost_label = (
            f"token-estimated {_DRY_RUN_INPUT_TOKENS}in"
            f"+{_DRY_RUN_OUTPUT_TOKENS}out (${recon_cost_per_sample:.5f}/sample)"
        )

    total_new = 0
    estimated_total_cost = 0.0
    subject_dist: dict[str, int] = {}

    print(
        f"\n  {'#':<4}  {'Problem':<28}  {'Subject':<12}  "
        f"{'Existing':<9}  {'New':<5}  Est. Cost"
    )
    print(f"  {'-'*4}  {'-'*28}  {'-'*12}  {'-'*9}  {'-'*5}  ---------")

    for num, p in enumerate(problems, start=1):
        existing = existing_counts.get(p.id, 0)
        new = max(0, _N_MAX - existing)
        est = new * recon_cost_per_sample
        total_new += new
        estimated_total_cost += est
        subject_dist[p.subject] = subject_dist.get(p.subject, 0) + 1
        marker = " " if new > 0 else "*"
        print(
            f"  {marker}{num:<3}  {p.id:<28}  {p.subject:<12}  "
            f"{existing:<9}  {new:<5}  ${est:.4f}"
        )

    print("\n  (* = already at N=64, will be skipped)")
    print(f"\n  Cost estimate basis : {cost_label}")
    print(f"  Problems            : {len(problems)}")
    print(
        "  Subject distribution: "
        + "  ".join(
            f"{s}={subject_dist.get(s, 0)}" for s in ("physics", "chemistry", "biology")
        )
    )
    print(f"  Total new samples   : {total_new:,}")
    print(f"  Estimated cost      : ${estimated_total_cost:.4f}")
    print(f"  Cost halt threshold : ${12.0 * 0.80:.2f}  (80% of $12.00 cap)")
    print(f"  Margin remaining    : ${12.0 * 0.80 - estimated_total_cost:.4f}")
    print(sep)

    if estimated_total_cost > 12.0 * 0.80:
        print(
            "  WARNING: estimated cost exceeds halt threshold — review before running."
        )
    else:
        print("  Looks safe. Run without --dry-run to start sampling.")
    print(sep)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------


def _print_final_summary(
    problems: list[Any],
    this_run_costs: dict[str, float],
    this_run_new_samples: dict[str, int],
    cumulative_cost: float,
    elapsed_s: float,
) -> None:
    """Print end-of-run summary."""
    sep = "=" * 72
    total_new_this_run = sum(this_run_new_samples.values())
    run_cost = sum(this_run_costs.values())

    by_subject: dict[str, dict[str, int]] = {}
    for p in problems:
        s = p.subject
        if s not in by_subject:
            by_subject[s] = {"problems": 0, "new_samples": 0}
        by_subject[s]["problems"] += 1
        by_subject[s]["new_samples"] += this_run_new_samples.get(p.id, 0)

    top3 = sorted(this_run_costs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    mins, secs = divmod(int(elapsed_s), 60)
    hrs, mins = divmod(mins, 60)
    elapsed_str = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"

    print(f"\n{sep}")
    print("Phase 6 — Complete")
    print(sep)
    print(f"  New samples this run : {total_new_this_run:,}")
    print(f"  Cost this run        : ${run_cost:.4f}")
    print(f"  Cumulative cost      : ${cumulative_cost:.4f}")
    print(f"  Time elapsed         : {elapsed_str}")
    print()
    print("  Per-subject (new samples this run):")
    for subj in ("physics", "chemistry", "biology"):
        d = by_subject.get(subj, {"problems": 0, "new_samples": 0})
        print(
            f"    {subj:<12}: {d['new_samples']:>4} new samples across {d['problems']} problems"
        )
    if top3:
        print()
        print("  Top 3 most expensive problems (this run):")
        for pid, cost in top3:
            print(f"    {pid}  ${cost:.4f}")
    print()
    print("  Next step: python scripts/run_fitting.py  (Phase 8)")
    print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_main_pilot(dry_run: bool = False) -> bool:
    """Run Phase 6 main pilot sampling; return True on clean completion."""
    from pilot.config import (
        CHECKPOINT_INTERVAL,
        COST_HARD_CAP,
        COST_WARN_FRACTION,
        OUTPUTS_DIR,
        SAMPLES_DIR,
        SCHEMA_VERSION,
        TOGETHER_INPUT_PRICE_PER_TOKEN,
        TOGETHER_OUTPUT_PRICE_PER_TOKEN,
    )
    from pilot.data_loader import load_gpqa_diamond
    from pilot.sampling import TogetherClient, sample_problem

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # --- Problem selection ---
    locked_ids = _load_locked_ids()
    gate_ids = _load_gate_problem_ids()
    logger.info("Loading GPQA Diamond…")
    all_problems = load_gpqa_diamond()
    problems = _select_main_pilot_problems(all_problems, gate_ids, locked_ids)
    if len(problems) != 47:
        logger.error("Expected 47 problems, got %d.", len(problems))
        return False

    # --- Count existing samples ---
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    existing_counts = {
        p.id: _count_samples_at_temperature(SAMPLES_DIR, p.id, _TEMPERATURE)
        for p in problems
    }
    n_already_complete = sum(1 for n in existing_counts.values() if n >= _N_MAX)
    logger.info(
        "%d/47 problems already have %d samples. %d need new sampling.",
        n_already_complete,
        _N_MAX,
        47 - n_already_complete,
    )

    # --- Dry-run path ---
    if dry_run:
        _print_dry_run(
            problems,
            existing_counts,
            TOGETHER_INPUT_PRICE_PER_TOKEN,
            TOGETHER_OUTPUT_PRICE_PER_TOKEN,
            OUTPUTS_DIR,
        )
        return True

    # --- Live run ---
    try:
        client = TogetherClient()
    except Exception as exc:
        logger.error("Failed to initialise TogetherClient: %s", exc)
        return False

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_or_init_progress(
        OUTPUTS_DIR, problems, existing_counts, SCHEMA_VERSION
    )
    progress_map = {pp["problem_id"]: pp for pp in progress["per_problem"]}

    cumulative_cost: float = progress.get("cumulative_cost_usd", 0.0)
    halt_threshold: float = COST_HARD_CAP * COST_WARN_FRACTION
    cost_per_problem_estimate: float = _INITIAL_COST_ESTIMATE
    completed_problem_costs: list[float] = []

    this_run_costs: dict[str, float] = {}
    this_run_new_samples: dict[str, int] = {}

    start_time = time.monotonic()
    interrupted = False

    try:
        for problem_num, problem in enumerate(problems, start=1):
            pp_entry = progress_map[problem.id]

            if pp_entry["complete"]:
                logger.info(
                    "[%d/47] %s — already complete, skipping.",
                    problem_num,
                    problem.id,
                )
                this_run_new_samples[problem.id] = 0
                this_run_costs[problem.id] = 0.0
                continue

            # --- Cost guard ---
            if cumulative_cost + cost_per_problem_estimate > halt_threshold:
                halt_reason = (
                    f"cost guard: ${cumulative_cost:.4f} + "
                    f"est ${cost_per_problem_estimate:.4f} "
                    f"> halt threshold ${halt_threshold:.2f}"
                )
                logger.error("HALTING — %s", halt_reason)
                progress["halted"] = True
                progress["halt_reason"] = halt_reason
                _save_progress(OUTPUTS_DIR, progress)
                return False

            # --- Sample ---
            output_path = SAMPLES_DIR / f"{problem.id}.jsonl"
            new_samples = await sample_problem(
                problem=problem,
                n_total=_N_MAX,
                client=client,
                output_path=output_path,
                temperature=_TEMPERATURE,
            )

            # --- Cost accounting ---
            problem_cost = sum(
                s.input_tokens * TOGETHER_INPUT_PRICE_PER_TOKEN
                + s.output_tokens * TOGETHER_OUTPUT_PRICE_PER_TOKEN
                for s in new_samples
            )
            cumulative_cost += problem_cost
            completed_problem_costs.append(problem_cost)
            if completed_problem_costs:
                cost_per_problem_estimate = sum(completed_problem_costs) / len(
                    completed_problem_costs
                )

            this_run_costs[problem.id] = problem_cost
            this_run_new_samples[problem.id] = len(new_samples)

            # --- Update progress ---
            n_now = _count_samples_at_temperature(SAMPLES_DIR, problem.id, _TEMPERATURE)
            pp_entry["n_samples"] = n_now
            pp_entry["complete"] = n_now >= _N_MAX
            pp_entry["cost_usd"] = pp_entry.get("cost_usd", 0.0) + problem_cost
            progress["n_problems_complete"] = sum(
                1 for pp in progress["per_problem"] if pp["complete"]
            )
            progress["cumulative_cost_usd"] = cumulative_cost
            _save_progress(OUTPUTS_DIR, progress)

            logger.info(
                "[%d/47] %s (%s)  +%d samples  $%.4f (cumulative $%.4f)",
                problem_num,
                problem.id,
                problem.subject,
                len(new_samples),
                problem_cost,
                cumulative_cost,
            )

            # --- Git checkpoint every CHECKPOINT_INTERVAL problems ---
            if progress["n_problems_complete"] % CHECKPOINT_INTERVAL == 0:
                _git_checkpoint(
                    OUTPUTS_DIR,
                    progress["n_problems_complete"],
                    47,
                    cumulative_cost,
                )

    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        logger.info(
            "Interrupted after %d problems. Progress saved.",
            progress.get("n_problems_complete", 0),
        )
        logger.info("Resume with: python scripts/run_main_pilot.py")
        progress["halted"] = True
        progress["halt_reason"] = "KeyboardInterrupt"
        _save_progress(OUTPUTS_DIR, progress)

    # --- Final checkpoint and summary ---
    _git_checkpoint(OUTPUTS_DIR, progress["n_problems_complete"], 47, cumulative_cost)

    elapsed = time.monotonic() - start_time
    _print_final_summary(
        problems, this_run_costs, this_run_new_samples, cumulative_cost, elapsed
    )

    all_complete = progress["n_problems_complete"] == 47
    if all_complete and not interrupted:
        logger.info(
            "Phase 6 complete. All 47 problems have %d samples at T=%.1f.",
            _N_MAX,
            _TEMPERATURE,
        )
    elif not interrupted:
        logger.warning(
            "%d/47 problems complete. Re-run to resume.",
            progress["n_problems_complete"],
        )

    return all_complete and not interrupted


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 6: sample N=64 completions for all 47 main-pilot problems."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sampling plan and cost estimate without making API calls.",
    )
    args = parser.parse_args()
    _load_env()
    ok = asyncio.run(run_main_pilot(dry_run=args.dry_run))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
