"""Constrained c* re-fit for H5 workshop paper validation.

Re-fits all 5 curve families on the 47 main-pilot problems with c* ∈ [1, 64]
for the unimodal family. All other families and bounds are unchanged. R(N)
data is loaded from outputs/fits/ to avoid recomputing expensive bootstrap
estimates.

Produces outputs/fits_constrained/<problem_id>.json (per-problem) and
outputs/fits_constrained_summary.json (aggregate).

Usage:
    python scripts/run_fitting_constrained.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_C_STAR_LOWER: float = 1.0
_C_STAR_UPPER: float = 64.0
_N_VALUES_MAIN: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
_N_MAIN_PROBLEMS: int = 47

_FITS_DIR = ROOT / "outputs" / "fits"
_FITS_CONSTRAINED_DIR = ROOT / "outputs" / "fits_constrained"
_OUTPUTS_DIR = ROOT / "outputs"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constrained unimodal fit
# ---------------------------------------------------------------------------


def _fit_unimodal_constrained(c: np.ndarray, R: np.ndarray):
    """Fit unimodal R(c) = A·exp(-((c-c*)²/2σ²)) + b with c* ∈ [1, 64]."""
    from pilot.fitting import _make_fit_result, _unimodal

    peak_idx = int(np.argmax(R))
    c_star_init = float(np.clip(c[peak_idx], _C_STAR_LOWER, _C_STAR_UPPER))
    p0 = [
        max(float(R[peak_idx] - np.min(R)), 0.1),
        c_star_init,
        max(float(np.std(c)), 1.0),
        float(np.min(R)),
    ]
    bounds = ([0.0, _C_STAR_LOWER, 1e-4, 0.0], [1.0, _C_STAR_UPPER, 1e4, 1.0])
    return _make_fit_result("unimodal", _unimodal, c, R, bounds, p0)


def _fit_all_constrained(c: np.ndarray, R: np.ndarray):
    """Fit all 5 families; unimodal uses constrained c*."""
    from pilot.fitting import (
        bic_weights,
        fit_constant,
        fit_gompertz,
        fit_logistic,
        fit_shifted_logistic,
    )

    fits = {
        "constant": fit_constant(c, R),
        "logistic": fit_logistic(c, R),
        "gompertz": fit_gompertz(c, R),
        "shifted_logistic": fit_shifted_logistic(c, R),
        "unimodal": _fit_unimodal_constrained(c, R),
    }
    weights = bic_weights(fits)
    return fits, weights


# ---------------------------------------------------------------------------
# ID helpers (mirrors run_fitting.py)
# ---------------------------------------------------------------------------


def _load_gate_ids() -> list[str]:
    path = _OUTPUTS_DIR / "gate_minus_1_labels.json"
    return json.loads(path.read_text())["gate_problems"]


def _load_locked_ids() -> list[str]:
    return json.loads((ROOT / "data" / "problem_ids.json").read_text())


def _main_pilot_ids() -> list[str]:
    locked = _load_locked_ids()
    gate = set(_load_gate_ids())
    ids = sorted(pid for pid in locked if pid not in gate)
    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Re-fit all 47 main-pilot problems with constrained unimodal c*."""
    from pilot.config import CURVE_FAMILIES, SCHEMA_VERSION

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    main_ids = _main_pilot_ids()
    if len(main_ids) != _N_MAIN_PROBLEMS:
        logger.error("Expected %d main-pilot IDs, got %d.", _N_MAIN_PROBLEMS, len(main_ids))
        sys.exit(1)

    _FITS_CONSTRAINED_DIR.mkdir(parents=True, exist_ok=True)

    c_arr = np.array(_N_VALUES_MAIN, dtype=float)

    family_wins: dict[str, int] = {f: 0 for f in CURVE_FAMILIES}
    flip_problems: list[dict[str, str]] = []
    n_unimodal_wins = 0

    for num, pid in enumerate(main_ids, start=1):
        fit_path = _FITS_DIR / f"{pid}.json"
        if not fit_path.exists():
            logger.warning("[%d/%d] %s: fits file missing — skipping.", num, _N_MAIN_PROBLEMS, pid)
            continue

        existing = json.loads(fit_path.read_text())
        R_arr = np.array([existing["R_at_N"][str(N)] for N in _N_VALUES_MAIN], dtype=float)
        best_unconstrained = existing["best_fit_family"]
        subject = existing.get("subject", "unknown")

        logger.info("[%d/%d] %s (%s) — prev best: %s", num, _N_MAIN_PROBLEMS, pid, subject, best_unconstrained)

        fits, weights = _fit_all_constrained(c_arr, R_arr)
        best_constrained = max(weights, key=lambda f: weights[f])

        if best_unconstrained == "unimodal" and best_constrained != "unimodal":
            logger.info(
                "  FLIP: %s unimodal -> %s (c* constrained to [%.0f, %.0f])",
                pid,
                best_constrained,
                _C_STAR_LOWER,
                _C_STAR_UPPER,
            )
            flip_problems.append(
                {"problem_id": pid, "subject": subject, "flips_to": best_constrained}
            )

        family_wins[best_constrained] = family_wins.get(best_constrained, 0) + 1
        if best_constrained == "unimodal":
            n_unimodal_wins += 1

        unimodal_fit = fits["unimodal"]
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "problem_id": pid,
            "subject": subject,
            "c_star_bounds": [_C_STAR_LOWER, _C_STAR_UPPER],
            "R_at_N": existing["R_at_N"],
            "best_fit_unconstrained": best_unconstrained,
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
            "best_fit_constrained": best_constrained,
            "unimodal_c_star_constrained": float(unimodal_fit.params[1]),
            "winner_flips": best_unconstrained == "unimodal" and best_constrained != "unimodal",
        }
        (_FITS_CONSTRAINED_DIR / f"{pid}.json").write_text(json.dumps(result, indent=2))

    n_fit = sum(family_wins.values())
    unimodal_fraction = n_unimodal_wins / n_fit if n_fit else 0.0

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "c_star_bounds": [_C_STAR_LOWER, _C_STAR_UPPER],
        "n_problems": n_fit,
        "family_win_counts_constrained": family_wins,
        "n_unimodal_wins_constrained": n_unimodal_wins,
        "unimodal_fraction_constrained": round(unimodal_fraction, 4),
        "n_winner_flips": len(flip_problems),
        "flip_problems": flip_problems,
        "h5_verdict_constrained": "PASS" if unimodal_fraction >= 0.05 else "FAIL",
    }
    (_OUTPUTS_DIR / "fits_constrained_summary.json").write_text(json.dumps(summary, indent=2))

    sep = "=" * 72
    print(f"\n{sep}")
    print("Constrained c* re-fit summary (c* in [1, 64])")
    print(sep)
    print(f"  Problems re-fit         : {n_fit}")
    print(f"  Unimodal wins           : {n_unimodal_wins}/{n_fit}  ({unimodal_fraction:.1%})")
    print(f"  Winner flips            : {len(flip_problems)}")
    for f in flip_problems:
        print(f"    {f['problem_id']} ({f['subject']}) -> {f['flips_to']}")
    print(f"  H5 verdict (constrained): {summary['h5_verdict_constrained']}")
    print(sep)


if __name__ == "__main__":
    main()
