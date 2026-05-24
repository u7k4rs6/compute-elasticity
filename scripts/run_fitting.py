"""Phase 8 — Curve fitting.

Applies 5 curve families (constant, logistic, Gompertz, shifted-logistic,
unimodal) to per-problem R(N) accuracy curves using existing samples.
Produces per-problem fit JSON files and an aggregate summary.

No API calls — pure local computation.

Usage:
    source .venv/bin/activate
    python scripts/run_fitting.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_N_MAIN_PROBLEMS: int = 47
_T_MAIN: float = 0.7
_N_SAMPLES_MAIN: int = 64
_N_SAMPLES_SIDE: int = 16
_TEMPERATURES_SIDE: tuple[float, ...] = (0.3, 0.7, 1.0)
_N_VALUES_MAIN: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
_N_VALUES_SIDE: tuple[int, ...] = (1, 2, 4, 8, 16)

# Subsampling parameters
_N_SUBSAMPLES_POINT: int = 1000  # for point-estimate R(N)
_N_BOOTSTRAP: int = 1000  # bootstrap resamples for CI
_N_BOOT_INNER: int = 100  # inner subsamples within each bootstrap resample
_N_SUBSAMPLES_SIDE: int = 500  # for side-test point estimates
_N_BOOTSTRAP_SIDE: int = 500  # side-test bootstrap resamples

# Exact enumeration threshold: use all C(n,N) if ≤ this (for N=2: C(64,2)=2016)
_EXACT_THRESHOLD: int = 5000

# BIC margin for "competing families" (H2b)
_BIC_MARGIN: float = 2.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment / path helpers
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


def _temp_key(t: float) -> str:
    return f"{t:.1f}"


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _load_samples_at_temperature(
    samples_dir: Path, problem_id: str, temperature: float
) -> list[dict[str, Any]]:
    """Return all sample dicts matching temperature, sorted by sample_idx."""
    path = samples_dir / f"{problem_id}.jsonl"
    if not path.exists():
        return []
    samples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if abs(obj.get("temperature", -1) - temperature) < 1e-6:
                samples.append(obj)
        except json.JSONDecodeError:
            continue
    samples.sort(key=lambda s: s.get("sample_idx", 0))
    return samples


def _read_metadata(samples_dir: Path, problem_id: str) -> tuple[str, str]:
    """Return (subject, ground_truth) from the first line of a sample file."""
    path = samples_dir / f"{problem_id}.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                return obj.get("subject", "unknown"), obj.get("ground_truth", "")
            except json.JSONDecodeError:
                continue
    return "unknown", ""


# ---------------------------------------------------------------------------
# Problem ID selection
# ---------------------------------------------------------------------------


def _load_gate_ids() -> list[str]:
    path = ROOT / "outputs" / "gate_minus_1_labels.json"
    return json.loads(path.read_text())["gate_problems"]


def _load_locked_ids() -> list[str]:
    return json.loads((ROOT / "data" / "problem_ids.json").read_text())


def _load_side_test_ids() -> list[str]:
    data = json.loads((ROOT / "outputs" / "recon_results.json").read_text())
    return data["gate_1_problems_reused"] + data["new_recon_problems"]


def _select_main_pilot_ids(locked_ids: list[str], gate_ids: list[str]) -> list[str]:
    gate = set(gate_ids)
    ids = [pid for pid in locked_ids if pid not in gate]
    ids.sort()
    return ids


# ---------------------------------------------------------------------------
# Majority-vote and R(N) estimation
# ---------------------------------------------------------------------------


def _majority_vote_correct(
    answers: list[str | None], ground_truth: str, rng: np.random.Generator
) -> float:
    """Return 1.0 if majority-vote answer == ground_truth, else 0.0.

    Ties broken uniformly at random with a deterministic RNG. Ground truth
    is not consulted during tie-breaking — only for the final correctness
    check. This is the standard unbiased best-of-N majority-vote estimator.
    """
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return 0.0
    max_count = max(counts.values())
    tied = sorted(a for a, c in counts.items() if c == max_count)
    winner = tied[0] if len(tied) == 1 else str(rng.choice(tied))
    return 1.0 if winner == ground_truth else 0.0


def _compute_R_at_N(
    answers: list[str | None],
    ground_truth: str,
    N: int,
    rng: np.random.Generator,
    n_subsamples: int,
    force_random: bool = False,
) -> float:
    """Estimate R(N) = E[majority_vote_correct] over N-sample subsets."""
    n_total = len(answers)
    if N >= n_total:
        return _majority_vote_correct(answers, ground_truth, rng)
    if N == 1:
        return sum(1.0 for a in answers if a == ground_truth) / n_total

    n_combos = comb(n_total, N)
    if not force_random and n_combos <= _EXACT_THRESHOLD:
        scores = [
            _majority_vote_correct([answers[i] for i in idx], ground_truth, rng)
            for idx in combinations(range(n_total), N)
        ]
        return float(np.mean(scores))

    scores = [
        _majority_vote_correct(
            [answers[i] for i in rng.choice(n_total, size=N, replace=False)],
            ground_truth,
            rng,
        )
        for _ in range(n_subsamples)
    ]
    return float(np.mean(scores))


def _compute_R_curve(
    answers: list[str | None],
    ground_truth: str,
    n_values: tuple[int, ...],
    rng: np.random.Generator,
    n_subsamples: int,
    force_random: bool = False,
) -> dict[int, float]:
    return {
        N: _compute_R_at_N(answers, ground_truth, N, rng, n_subsamples, force_random)
        for N in n_values
    }


def _bootstrap_ci(
    answers: list[str | None],
    ground_truth: str,
    n_values: tuple[int, ...],
    rng: np.random.Generator,
    n_bootstrap: int,
    n_inner: int,
) -> tuple[dict[int, float], dict[int, float]]:
    """Percentile bootstrap 90% CI for R(N) at each N."""
    n_total = len(answers)
    boot_curves: list[dict[int, float]] = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_total, size=n_total, replace=True)
        boot_answers = [answers[i] for i in indices]
        curve = _compute_R_curve(
            boot_answers, ground_truth, n_values, rng, n_inner, force_random=True
        )
        boot_curves.append(curve)

    ci_lower: dict[int, float] = {}
    ci_upper: dict[int, float] = {}
    for N in n_values:
        vals = [c[N] for c in boot_curves]
        ci_lower[N] = float(np.percentile(vals, 5))
        ci_upper[N] = float(np.percentile(vals, 95))
    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Core fit function
# ---------------------------------------------------------------------------


def _fit_problem(
    problem_id: str,
    subject: str,
    ground_truth: str,
    samples: list[dict[str, Any]],
    n_values: tuple[int, ...],
    temperature: float,
    rng: np.random.Generator,
    schema_version: str,
    n_subsamples: int = _N_SUBSAMPLES_POINT,
    n_bootstrap: int = _N_BOOTSTRAP,
    n_boot_inner: int = _N_BOOT_INNER,
) -> dict[str, Any]:
    """Fit all 5 families to R(N) for one (problem, temperature) cell."""
    from pilot.analysis import fitted_elasticity_summary
    from pilot.fitting import bic_weights, fit_all_families

    answers: list[str | None] = [s.get("extracted_answer") for s in samples]
    n_used = len(answers)

    R_curve = _compute_R_curve(answers, ground_truth, n_values, rng, n_subsamples)
    ci_lower, ci_upper = _bootstrap_ci(
        answers, ground_truth, n_values, rng, n_bootstrap, n_boot_inner
    )

    c_arr = np.array(n_values, dtype=float)
    R_arr = np.array([R_curve[N] for N in n_values], dtype=float)
    fits = fit_all_families(c_arr, R_arr)
    weights = bic_weights(fits)
    best_family = max(weights, key=lambda f: weights[f])
    mean_elasticity = fitted_elasticity_summary(fits[best_family], c_range=(8.0, 64.0))

    return {
        "schema_version": schema_version,
        "problem_id": problem_id,
        "subject": subject,
        "temperature": temperature,
        "n_samples_used": n_used,
        "R_at_N": {str(N): R_curve[N] for N in n_values},
        "R_at_N_ci_lower": {str(N): ci_lower[N] for N in n_values},
        "R_at_N_ci_upper": {str(N): ci_upper[N] for N in n_values},
        "fits": {
            name: {
                "params": list(fit.params),
                "bic": fit.bic,
                "residual_se": fit.residual_se,
                "converged": fit.converged,
                "weight": weights[name],
            }
            for name, fit in fits.items()
        },
        "best_fit_family": best_family,
        "mean_elasticity_8_64": mean_elasticity,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_fitting() -> bool:
    """Run Phase 8 curve fitting; return True on success."""
    from pilot.config import (
        CURVE_FAMILIES,
        FITS_DIR,
        OUTPUTS_DIR,
        SAMPLES_DIR,
        SCHEMA_VERSION,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rng = np.random.default_rng(42)

    locked_ids = _load_locked_ids()
    gate_ids = _load_gate_ids()
    side_test_ids = _load_side_test_ids()
    main_pilot_ids = _select_main_pilot_ids(locked_ids, gate_ids)

    if len(main_pilot_ids) != _N_MAIN_PROBLEMS:
        logger.error(
            "Expected %d main-pilot IDs, got %d.", _N_MAIN_PROBLEMS, len(main_pilot_ids)
        )
        return False

    FITS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Fits directory: %s", FITS_DIR)

    # -------------------------------------------------------------------------
    # Main pilot (47 problems, T=0.7, N ∈ {1,2,4,8,16,32,64})
    # -------------------------------------------------------------------------
    main_results: list[dict[str, Any]] = []
    n_nonconverged_total = 0

    for num, pid in enumerate(main_pilot_ids, start=1):
        samples = _load_samples_at_temperature(SAMPLES_DIR, pid, _T_MAIN)
        if len(samples) < max(_N_VALUES_MAIN):
            logger.warning(
                "[%d/%d] %s: only %d samples, need %d — skipping.",
                num,
                _N_MAIN_PROBLEMS,
                pid,
                len(samples),
                _N_SAMPLES_MAIN,
            )
            continue

        subject, ground_truth = _read_metadata(SAMPLES_DIR, pid)
        logger.info("[%d/%d] Fitting %s (%s)…", num, _N_MAIN_PROBLEMS, pid, subject)

        result = _fit_problem(
            pid,
            subject,
            ground_truth,
            samples,
            _N_VALUES_MAIN,
            _T_MAIN,
            rng,
            SCHEMA_VERSION,
            n_subsamples=_N_SUBSAMPLES_POINT,
            n_bootstrap=_N_BOOTSTRAP,
            n_boot_inner=_N_BOOT_INNER,
        )
        (FITS_DIR / f"{pid}.json").write_text(json.dumps(result, indent=2))

        n_nonconv = sum(1 for f in CURVE_FAMILIES if not result["fits"][f]["converged"])
        n_nonconverged_total += n_nonconv
        if n_nonconv:
            logger.warning("  %s: %d non-converged families.", pid, n_nonconv)

        main_results.append(result)

    # -------------------------------------------------------------------------
    # Side test (10 problems × 3 temperatures, N ∈ {1,2,4,8,16})
    # -------------------------------------------------------------------------
    side_results: list[dict[str, Any]] = []
    side_subsample_rng = np.random.default_rng(42)  # fixed for reproducibility

    for pid in side_test_ids:
        subject, ground_truth = _read_metadata(SAMPLES_DIR, pid)
        for t in _TEMPERATURES_SIDE:
            all_samples = _load_samples_at_temperature(SAMPLES_DIR, pid, t)
            if len(all_samples) < _N_SAMPLES_SIDE:
                logger.warning(
                    "Side test %s T=%s: only %d samples, need %d — skipping.",
                    pid,
                    _temp_key(t),
                    len(all_samples),
                    _N_SAMPLES_SIDE,
                )
                continue

            # Subsample exactly 16 with fixed seed for fair cross-T comparison
            indices = side_subsample_rng.choice(
                len(all_samples), size=_N_SAMPLES_SIDE, replace=False
            )
            samples_16 = [all_samples[i] for i in sorted(indices)]

            logger.info(
                "  Side test %s T=%s (%d avail)…", pid, _temp_key(t), len(all_samples)
            )
            result = _fit_problem(
                pid,
                subject,
                ground_truth,
                samples_16,
                _N_VALUES_SIDE,
                t,
                rng,
                SCHEMA_VERSION,
                n_subsamples=_N_SUBSAMPLES_SIDE,
                n_bootstrap=_N_BOOTSTRAP_SIDE,
                n_boot_inner=_N_BOOT_INNER,
            )

            tk = _temp_key(t)
            out_name = f"side_test_{pid}_t{tk}.json"
            (FITS_DIR / out_name).write_text(json.dumps(result, indent=2))
            side_results.append(result)

    # -------------------------------------------------------------------------
    # Aggregate summary
    # -------------------------------------------------------------------------
    if not main_results:
        logger.error("No main-pilot fit results produced.")
        return False

    family_wins: dict[str, int] = {f: 0 for f in CURVE_FAMILIES}
    family_bic_sums: dict[str, float] = {f: 0.0 for f in CURVE_FAMILIES}
    residual_ses: list[float] = []
    n_close_families = 0
    n_unimodal_wins = 0
    elasticities: list[float] = []

    for r in main_results:
        best = r["best_fit_family"]
        family_wins[best] = family_wins.get(best, 0) + 1
        for f in CURVE_FAMILIES:
            family_bic_sums[f] += r["fits"][f]["weight"]
        best_bic = r["fits"][best]["bic"]
        close = sum(
            1 for f in CURVE_FAMILIES if r["fits"][f]["bic"] - best_bic < _BIC_MARGIN
        )
        if close >= 2:
            n_close_families += 1
        residual_ses.append(r["fits"][best]["residual_se"])
        if best == "unimodal":
            n_unimodal_wins += 1
        elasticities.append(r["mean_elasticity_8_64"])

    n_fit = len(main_results)
    family_mean_bic_weights = {f: family_bic_sums[f] / n_fit for f in CURVE_FAMILIES}

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "n_main_pilot_problems": n_fit,
        "n_side_test_cells": len(side_results),
        "family_win_counts": family_wins,
        "family_mean_bic_weights": family_mean_bic_weights,
        "median_residual_se_main": float(np.median(residual_ses)),
        "n_problems_with_2plus_close_families": n_close_families,
        "n_problems_unimodal_wins": n_unimodal_wins,
        "fitted_elasticities_distribution": {
            "min": float(np.min(elasticities)),
            "median": float(np.median(elasticities)),
            "max": float(np.max(elasticities)),
        },
        "n_nonconverged_fits": n_nonconverged_total,
    }
    (OUTPUTS_DIR / "fits_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Fits summary written to %s", OUTPUTS_DIR / "fits_summary.json")

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print("Phase 8 — Fitting Summary")
    print(sep)
    print(f"  Main pilot problems fit : {n_fit}")
    print(f"  Side test cells fit     : {len(side_results)}")
    print()
    print("  Family win counts (main pilot):")
    for f in CURVE_FAMILIES:
        print(
            f"    {f:<20}: {family_wins.get(f, 0):>3} wins"
            f"  mean BIC weight {family_mean_bic_weights[f]:.4f}"
        )
    median_se = summary["median_residual_se_main"]
    print()
    print(
        f"  Median residual SE      : {median_se:.4f}"
        f"  (H1: pass ≤0.10 / fail ≥0.15)"
    )
    print(
        f"  Problems ≥2 close fams  : {n_close_families}/{n_fit}" f"  (H2b: pass ≥30%)"
    )
    print(f"  Unimodal wins           : {n_unimodal_wins}/{n_fit}" f"  (H5: pass >5%)")
    print(f"  Elasticity median [8,64]: {np.median(elasticities):.5f}")
    if n_nonconverged_total:
        print(f"\n  WARNING: {n_nonconverged_total} non-converged fits.")
    print(sep)

    return True


def main() -> None:
    """Entry point."""
    _load_env()
    ok = run_fitting()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
