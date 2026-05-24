"""Phase 7 — Temperature side test.

Samples 16 completions per (problem, temperature) across 3 temperatures
{0.3, 0.7, 1.0} on the 10 recon problems (Gate-1 + Day-0 recon set).

T=0.7 samples reused from Phase 5/6 where they already exist (≥16 needed).
Progress checkpointed to outputs/side_test_progress.json and committed every
5 fully-processed problems (all 3 temperatures done). Ctrl+C saves state and
allows clean resumption.

Usage:
    source .venv/bin/activate
    python scripts/run_side_test.py             # live run
    python scripts/run_side_test.py --dry-run   # plan without API calls
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

_N_SIDE: int = 16  # = config.N_SIDE_TEST
_TEMPERATURES: tuple[float, ...] = (0.3, 0.7, 1.0)  # = config.TEMPERATURES_SIDE
_N_PROBLEMS: int = 10
_PROGRESS_FILENAME: str = "side_test_progress.json"
# Fallback if phase_6_progress.json is absent
_PRIOR_PHASE_COST_FALLBACK: float = 0.51

# Tokens-per-completion assumed for dry-run / cost-guard estimate
_DRY_RUN_INPUT_TOKENS: int = 400
_DRY_RUN_OUTPUT_TOKENS: int = 800

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
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


# ---------------------------------------------------------------------------
# Problem ID loading
# ---------------------------------------------------------------------------


def _load_side_test_problem_ids() -> list[str]:
    """Read 10 side-test problem IDs from recon_results.json."""
    path = ROOT / "outputs" / "recon_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Recon results not found: {path}\n" "Run scripts/run_recon.py first."
        )
    data = json.loads(path.read_text())
    gate_ids: list[str] = data.get("gate_1_problems_reused", [])
    new_ids: list[str] = data.get("new_recon_problems", [])
    combined = gate_ids + new_ids
    if len(combined) != _N_PROBLEMS:
        raise ValueError(
            f"Expected {_N_PROBLEMS} side-test problems, "
            f"got {len(combined)} from recon_results.json."
        )
    return combined


def _load_prior_phase_cost(outputs_dir: Path) -> float:
    """Read cumulative cost from phase_6_progress.json; fall back to hardcoded."""
    path = outputs_dir / "phase_6_progress.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            cost = data.get("cumulative_cost_usd", None)
            if isinstance(cost, (int, float)) and cost > 0:
                return float(cost)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    logger.warning(
        "Could not read Phase 6 cost from phase_6_progress.json; "
        "using fallback $%.2f.",
        _PRIOR_PHASE_COST_FALLBACK,
    )
    return _PRIOR_PHASE_COST_FALLBACK


# ---------------------------------------------------------------------------
# Sample counting (temperature-aware)
# ---------------------------------------------------------------------------


def _temp_key(t: float) -> str:
    """Stable string key for a temperature float (e.g. 0.3 → '0.3')."""
    return f"{t:.1f}"


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
    problem_ids: list[str],
    problem_subjects: dict[str, str],
    existing_counts: dict[str, dict[str, int]],
    schema_version: str,
    prior_cost: float,
) -> dict[str, Any]:
    """Build a fresh progress structure."""
    per_problem: list[dict[str, Any]] = []
    for pid in problem_ids:
        temps: dict[str, Any] = {}
        all_complete = True
        for t in _TEMPERATURES:
            tk = _temp_key(t)
            n = existing_counts.get(pid, {}).get(tk, 0)
            done = n >= _N_SIDE
            if not done:
                all_complete = False
            temps[tk] = {"n_samples": n, "complete": done, "cost_usd": 0.0}
        per_problem.append(
            {
                "problem_id": pid,
                "subject": problem_subjects.get(pid, "unknown"),
                "temperatures": temps,
                "all_complete": all_complete,
            }
        )

    n_cells_complete = sum(
        1
        for pp in per_problem
        for tk in pp["temperatures"]
        if pp["temperatures"][tk]["complete"]
    )

    return {
        "schema_version": schema_version,
        "phase": "side_test",
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_problems_total": len(problem_ids),
        "n_cells_total": len(problem_ids) * len(_TEMPERATURES),
        "n_cells_complete": n_cells_complete,
        "n_problems_complete": sum(1 for pp in per_problem if pp["all_complete"]),
        "per_problem": per_problem,
        "prior_phases_cost_usd": prior_cost,
        "this_phase_cost_usd": 0.0,
        "halted": False,
        "halt_reason": None,
    }


def _load_or_init_progress(
    outputs_dir: Path,
    problem_ids: list[str],
    problem_subjects: dict[str, str],
    existing_counts: dict[str, dict[str, int]],
    schema_version: str,
    prior_cost: float,
) -> dict[str, Any]:
    """Load existing progress file if valid; otherwise initialise fresh."""
    path = outputs_dir / _PROGRESS_FILENAME
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("phase") == "side_test" and data.get("n_problems_total") == len(
                problem_ids
            ):
                logger.info(
                    "Resuming from progress file: %d/%d cells complete, "
                    "$%.4f this phase.",
                    data.get("n_cells_complete", 0),
                    data.get("n_cells_total", 0),
                    data.get("this_phase_cost_usd", 0.0),
                )
                return data
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Progress file malformed — starting fresh.")
    return _init_progress(
        problem_ids, problem_subjects, existing_counts, schema_version, prior_cost
    )


def _save_progress(outputs_dir: Path, progress: dict[str, Any]) -> None:
    progress["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (outputs_dir / _PROGRESS_FILENAME).write_text(json.dumps(progress, indent=2))


# ---------------------------------------------------------------------------
# Git checkpoint
# ---------------------------------------------------------------------------


def _git_checkpoint(
    outputs_dir: Path,
    n_complete: int,
    n_total: int,
    this_phase_cost: float,
) -> None:
    """Stage progress file and commit a checkpoint (no push)."""
    progress_path = outputs_dir / _PROGRESS_FILENAME
    try:
        diff = subprocess.run(
            ["git", "status", "--porcelain", str(progress_path)],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if not diff.stdout.strip():
            return
        subprocess.run(
            ["git", "add", str(progress_path)],
            check=True,
            capture_output=True,
            cwd=ROOT,
        )
        msg = (
            f"phase-7-progress: {n_complete}/{n_total} problems sampled"
            f" (cost: ${this_phase_cost:.2f})"
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
# Dry-run plan printer
# ---------------------------------------------------------------------------


def _print_dry_run(
    problem_ids: list[str],
    problem_subjects: dict[str, str],
    existing_counts: dict[str, dict[str, int]],
    input_price: float,
    output_price: float,
    prior_cost: float,
    halt_threshold: float,
) -> None:
    """Print sampling plan and estimated cost without making any API calls."""
    sep = "=" * 76
    print(f"\n{sep}")
    print("DRY RUN — Phase 7 Temperature Side Test Sampling Plan")
    print(sep)

    cost_per_sample = (
        _DRY_RUN_INPUT_TOKENS * input_price + _DRY_RUN_OUTPUT_TOKENS * output_price
    )

    temp_headers = "  ".join(f"T={t:.1f}(+new)" for t in _TEMPERATURES)
    print(f"\n  {'#':<3}  {'Problem':<28}  {'Subj':<9}  {temp_headers}  Est.Cost")
    print(
        f"  {'-'*3}  {'-'*28}  {'-'*9}  "
        + "  ".join(["-" * 11] * len(_TEMPERATURES))
        + "  --------"
    )

    total_new = 0
    estimated_total_cost = 0.0

    for num, pid in enumerate(problem_ids, start=1):
        cell_strs = []
        problem_new = 0
        for t in _TEMPERATURES:
            tk = _temp_key(t)
            existing = existing_counts.get(pid, {}).get(tk, 0)
            new = max(0, _N_SIDE - existing)
            problem_new += new
            cell_strs.append(f"{existing:>2}+{new:<8}")
        problem_cost = problem_new * cost_per_sample
        total_new += problem_new
        estimated_total_cost += problem_cost
        subj = problem_subjects.get(pid, "?")[:9]
        print(
            f"  {num:<3}  {pid:<28}  {subj:<9}  "
            + "  ".join(cell_strs)
            + f"  ${problem_cost:.4f}"
        )

    print()
    print(
        f"  Cost estimate basis  : {_DRY_RUN_INPUT_TOKENS}in"
        f"+{_DRY_RUN_OUTPUT_TOKENS}out tokens (${cost_per_sample:.5f}/sample)"
    )
    print(f"  Prior phases cost    : ${prior_cost:.4f}")
    print(f"  Total new samples    : {total_new:,}")
    print(f"  Estimated this phase : ${estimated_total_cost:.4f}")
    cumulative_est = prior_cost + estimated_total_cost
    print(f"  Estimated cumulative : ${cumulative_est:.4f}")
    print(f"  Cost halt threshold  : ${halt_threshold:.2f}  (80% of $12.00 cap)")
    print(f"  Remaining margin     : ${halt_threshold - cumulative_est:.4f}")
    print(sep)

    if cumulative_est > halt_threshold:
        print("  WARNING: estimated cumulative cost exceeds halt threshold.")
    else:
        print("  Looks safe. Run without --dry-run to start sampling.")
    print(sep)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------


def _print_final_summary(
    problem_ids: list[str],
    problem_subjects: dict[str, str],
    this_run_new_samples: dict[str, int],
    this_phase_cost: float,
    prior_cost: float,
    elapsed_s: float,
) -> None:
    """Print end-of-run summary."""
    sep = "=" * 72
    mins, secs = divmod(int(elapsed_s), 60)
    hrs, mins = divmod(mins, 60)
    elapsed_str = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"

    total_new = sum(this_run_new_samples.values())
    by_subject: dict[str, int] = {}
    for pid in problem_ids:
        subj = problem_subjects.get(pid, "unknown")
        by_subject[subj] = by_subject.get(subj, 0) + this_run_new_samples.get(pid, 0)

    print(f"\n{sep}")
    print("Phase 7 — Temperature Side Test")
    print(sep)
    print(f"  New samples this run : {total_new:,}")
    print(f"  Cost this phase      : ${this_phase_cost:.4f}")
    print(f"  Cumulative cost      : ${prior_cost + this_phase_cost:.4f}")
    print(f"  Time elapsed         : {elapsed_str}")
    if by_subject:
        print()
        print("  New samples by subject (this run):")
        for subj in ("physics", "chemistry", "biology"):
            n = by_subject.get(subj, 0)
            if n or subj in by_subject:
                print(f"    {subj:<12}: {n:>4}")
    print()
    print("  Next step: python scripts/run_fitting.py  (Phase 8)")
    print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_side_test(dry_run: bool = False) -> bool:
    """Run Phase 7 temperature side test; return True on clean completion."""
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # --- Load problem IDs ---
    side_test_ids = _load_side_test_problem_ids()
    logger.info(
        "Side-test problem IDs (%d): %s", len(side_test_ids), ", ".join(side_test_ids)
    )

    # --- Build subject map from full dataset ---
    logger.info("Loading GPQA Diamond…")
    all_problems = load_gpqa_diamond()
    problem_map = {p.id: p for p in all_problems}

    missing = [pid for pid in side_test_ids if pid not in problem_map]
    if missing:
        logger.error("Problem IDs not found in dataset: %s", missing)
        return False

    problems = [problem_map[pid] for pid in side_test_ids]
    problem_subjects = {p.id: p.subject for p in problems}

    # --- Count existing samples for all (problem, temperature) cells ---
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    existing_counts: dict[str, dict[str, int]] = {}
    for p in problems:
        existing_counts[p.id] = {
            _temp_key(t): _count_samples_at_temperature(SAMPLES_DIR, p.id, t)
            for t in _TEMPERATURES
        }

    total_cells = len(problems) * len(_TEMPERATURES)
    cells_done = sum(
        1
        for p in problems
        for t in _TEMPERATURES
        if existing_counts[p.id][_temp_key(t)] >= _N_SIDE
    )
    logger.info(
        "%d/%d (problem, temperature) cells already have ≥%d samples.",
        cells_done,
        total_cells,
        _N_SIDE,
    )

    # --- Prior phases cost and halt threshold ---
    prior_cost = _load_prior_phase_cost(OUTPUTS_DIR)
    halt_threshold = COST_HARD_CAP * COST_WARN_FRACTION

    # --- Dry-run path ---
    if dry_run:
        _print_dry_run(
            side_test_ids,
            problem_subjects,
            existing_counts,
            TOGETHER_INPUT_PRICE_PER_TOKEN,
            TOGETHER_OUTPUT_PRICE_PER_TOKEN,
            prior_cost,
            halt_threshold,
        )
        return True

    # --- Live run ---
    from pilot.sampling import TogetherClient, sample_problem

    try:
        client = TogetherClient()
    except Exception as exc:
        logger.error("Failed to initialise TogetherClient: %s", exc)
        return False

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_or_init_progress(
        OUTPUTS_DIR,
        side_test_ids,
        problem_subjects,
        existing_counts,
        SCHEMA_VERSION,
        prior_cost,
    )
    progress_map = {pp["problem_id"]: pp for pp in progress["per_problem"]}

    this_phase_cost: float = progress.get("this_phase_cost_usd", 0.0)
    this_run_new_samples: dict[str, int] = {pid: 0 for pid in side_test_ids}
    cell_cost_estimate = _N_SIDE * (
        _DRY_RUN_INPUT_TOKENS * TOGETHER_INPUT_PRICE_PER_TOKEN
        + _DRY_RUN_OUTPUT_TOKENS * TOGETHER_OUTPUT_PRICE_PER_TOKEN
    )

    start_time = time.monotonic()
    interrupted = False

    try:
        for problem_num, problem in enumerate(problems, start=1):
            pp_entry = progress_map[problem.id]

            if pp_entry["all_complete"]:
                logger.info(
                    "[%d/%d] %s — all temperatures complete, skipping.",
                    problem_num,
                    _N_PROBLEMS,
                    problem.id,
                )
                continue

            for t in _TEMPERATURES:
                tk = _temp_key(t)
                cell = pp_entry["temperatures"][tk]

                if cell["complete"]:
                    logger.info(
                        "  [T=%s] %s — already complete (%d samples), skipping.",
                        tk,
                        problem.id,
                        cell["n_samples"],
                    )
                    continue

                # --- Cost guard ---
                cumulative = prior_cost + this_phase_cost
                if cumulative + cell_cost_estimate > halt_threshold:
                    halt_reason = (
                        f"cost guard: cumulative ${cumulative:.4f} + "
                        f"est ${cell_cost_estimate:.4f} "
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
                    n_total=_N_SIDE,
                    client=client,
                    output_path=output_path,
                    temperature=t,
                )

                # --- Cost accounting ---
                cell_cost = sum(
                    s.input_tokens * TOGETHER_INPUT_PRICE_PER_TOKEN
                    + s.output_tokens * TOGETHER_OUTPUT_PRICE_PER_TOKEN
                    for s in new_samples
                )
                this_phase_cost += cell_cost
                this_run_new_samples[problem.id] += len(new_samples)

                # --- Update cell progress ---
                n_now = _count_samples_at_temperature(SAMPLES_DIR, problem.id, t)
                cell["n_samples"] = n_now
                cell["complete"] = n_now >= _N_SIDE
                cell["cost_usd"] = cell.get("cost_usd", 0.0) + cell_cost

                progress["n_cells_complete"] = sum(
                    1
                    for pp in progress["per_problem"]
                    for tk2 in pp["temperatures"]
                    if pp["temperatures"][tk2]["complete"]
                )
                progress["this_phase_cost_usd"] = this_phase_cost
                _save_progress(OUTPUTS_DIR, progress)

                logger.info(
                    "  [%d/%d | T=%s] %s (%s)  +%d samples  $%.4f (phase $%.4f)",
                    problem_num,
                    _N_PROBLEMS,
                    tk,
                    problem.id,
                    problem.subject,
                    len(new_samples),
                    cell_cost,
                    this_phase_cost,
                )

            # --- Mark problem complete if all temperatures done ---
            pp_entry["all_complete"] = all(
                pp_entry["temperatures"][_temp_key(t)]["complete"]
                for t in _TEMPERATURES
            )
            progress["n_problems_complete"] = sum(
                1 for pp in progress["per_problem"] if pp["all_complete"]
            )
            _save_progress(OUTPUTS_DIR, progress)

            # --- Git checkpoint every CHECKPOINT_INTERVAL fully-processed problems ---
            if (
                pp_entry["all_complete"]
                and progress["n_problems_complete"] % CHECKPOINT_INTERVAL == 0
            ):
                _git_checkpoint(
                    OUTPUTS_DIR,
                    progress["n_problems_complete"],
                    _N_PROBLEMS,
                    this_phase_cost,
                )

    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        logger.info(
            "Interrupted after %d problems fully complete. Progress saved.",
            progress.get("n_problems_complete", 0),
        )
        logger.info("Resume with: python scripts/run_side_test.py")
        progress["halted"] = True
        progress["halt_reason"] = "KeyboardInterrupt"
        _save_progress(OUTPUTS_DIR, progress)

    # --- Final checkpoint and summary ---
    _git_checkpoint(
        OUTPUTS_DIR,
        progress.get("n_problems_complete", 0),
        _N_PROBLEMS,
        this_phase_cost,
    )

    elapsed = time.monotonic() - start_time
    _print_final_summary(
        side_test_ids,
        problem_subjects,
        this_run_new_samples,
        this_phase_cost,
        prior_cost,
        elapsed,
    )

    all_complete = progress.get("n_problems_complete", 0) == _N_PROBLEMS
    if all_complete and not interrupted:
        logger.info(
            "Phase 7 complete. All %d problems have %d samples at each of %s.",
            _N_PROBLEMS,
            _N_SIDE,
            _TEMPERATURES,
        )
    elif not interrupted:
        logger.warning(
            "%d/%d problems fully complete. Re-run to resume.",
            progress.get("n_problems_complete", 0),
            _N_PROBLEMS,
        )

    return all_complete and not interrupted


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7: temperature side test — "
            f"{_N_SIDE} samples × {len(_TEMPERATURES)} temperatures × {_N_PROBLEMS} problems."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sampling plan and cost estimate without making API calls.",
    )
    args = parser.parse_args()
    _load_env()
    ok = asyncio.run(run_side_test(dry_run=args.dry_run))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
