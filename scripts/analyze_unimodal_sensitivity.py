"""Unimodal c* sensitivity analysis (post-multi-start fits).

For each problem where unimodal wins BIC selection, this script:
  1. Reports c* distribution and regime breakdown (decaying / interior / saturating).
  2. Re-fits unimodal with a tightened upper bound c* ≤ 64 (the data grid ceiling)
     using the same multi-start logic as pilot/fitting.py.
  3. Checks whether this constrained fit flips the BIC winner to another family.

Writes outputs/unimodal_sensitivity.json.  No API calls, no test changes.

Usage:
    python scripts/analyze_unimodal_sensitivity.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Regime boundaries (same as analyze_unimodal_peaks.py)
_DECAYING: float = 4.0
_INTERIOR_LOW: float = 8.0
_INTERIOR_HIGH: float = 32.0
_SATURATING: float = 56.0

# Multi-start settings (mirror pilot/fitting.py)
_N_STARTS: int = 10
_SEED: int = 42

# Sensitivity test: constrain c* to the data grid ceiling
_CSTAR_CONSTRAINED_UB: float = 64.0

FAMILIES = ["constant", "logistic", "gompertz", "shifted_logistic", "unimodal"]


def _regime(c_star: float) -> str:
    if c_star <= _DECAYING:
        return "decaying"
    if c_star < _INTERIOR_LOW:
        return "borderline_low"
    if c_star <= _INTERIOR_HIGH:
        return "interior"
    if c_star < _SATURATING:
        return "borderline_high"
    return "saturating"


def _unimodal(
    c: np.ndarray, A: float, c_star: float, sigma: float, b: float
) -> np.ndarray:
    return A * np.exp(-((c - c_star) ** 2) / (2.0 * sigma**2)) + b


def _compute_bic(k: int, n: int, rss: float) -> float:
    rss_safe = max(rss, 1e-12)
    return k * np.log(n) + n * np.log(2 * np.pi * rss_safe / n) + n


def _fit_unimodal_constrained(
    c: np.ndarray,
    R: np.ndarray,
    c_star_ub: float = _CSTAR_CONSTRAINED_UB,
) -> dict:
    """Re-fit unimodal with c* ≤ c_star_ub using multi-start.

    Returns dict with keys: params, bic, residual_se, converged.
    """
    n = len(c)
    k = 4
    lower = np.array([0.0, 0.0, 1e-4, 0.0])
    upper = np.array([1.0, c_star_ub, 1e4, 1.0])
    sample_upper = np.minimum(upper, np.maximum(1e2, c.max() * 2.0))

    peak_idx = int(np.argmax(R))
    p0_heuristic = [
        max(float(R[peak_idx] - np.min(R)), 0.1),
        min(float(c[peak_idx]), c_star_ub),
        max(float(np.std(c)), 1.0),
        float(np.min(R)),
    ]

    rng = np.random.default_rng(_SEED)
    random_starts = [
        list(rng.uniform(lower, sample_upper)) for _ in range(_N_STARTS - 1)
    ]
    all_starts = [p0_heuristic] + random_starts

    best_rss = float("inf")
    best_popt: np.ndarray | None = None

    for p0_i in all_starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _unimodal,
                    c,
                    R,
                    p0=p0_i,
                    bounds=(lower, upper),
                    maxfev=10_000,
                    method="trf",
                )
            fitted = _unimodal(c, *popt)
            rss = float(np.sum((R - fitted) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_popt = popt
        except (RuntimeError, ValueError):
            continue

    converged = best_popt is not None
    if not converged:
        popt_arr = np.array(p0_heuristic, dtype=float)
        fitted = _unimodal(c, *popt_arr)
        best_rss = float(np.sum((R - fitted) ** 2))
    else:
        popt_arr = best_popt

    bic = _compute_bic(k=k, n=n, rss=best_rss)
    residual_se = float(np.sqrt(best_rss / n))
    return {
        "params": [float(x) for x in popt_arr],
        "bic": bic,
        "residual_se": residual_se,
        "converged": converged,
    }


def _bic_weights(bics: dict[str, float]) -> dict[str, float]:
    vals = np.array([bics[f] for f in FAMILIES])
    delta = vals - vals.min()
    raw = np.exp(-delta / 2.0)
    total = raw.sum()
    weights = raw / total if total > 0 else np.ones(len(FAMILIES)) / len(FAMILIES)
    return {f: float(w) for f, w in zip(FAMILIES, weights)}


def main() -> None:
    fits_dir = ROOT / "outputs" / "fits"
    out_path = ROOT / "outputs" / "unimodal_sensitivity.json"

    fit_files = sorted(
        f
        for f in fits_dir.glob("gpqa_diamond_*.json")
        if not f.name.startswith("side_test_")
    )
    if not fit_files:
        print("No main-pilot fit files found in", fits_dir)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Pass 1: collect unimodal winners, c* distribution
    # -------------------------------------------------------------------------
    unimodal_winners: list[dict] = []
    for f in fit_files:
        d = json.loads(f.read_text())
        if d["best_fit_family"] != "unimodal":
            continue
        params = d["fits"]["unimodal"]["params"]
        c_star = float(params[1])
        unimodal_winners.append(
            {
                "problem_id": d["problem_id"],
                "subject": d["subject"],
                "c_star_unconstrained": c_star,
                "regime": _regime(c_star),
                "unimodal_weight_unconstrained": d["fits"]["unimodal"]["weight"],
                "bics_unconstrained": {fam: d["fits"][fam]["bic"] for fam in FAMILIES},
                "R_at_N": d["R_at_N"],
            }
        )

    n = len(unimodal_winners)
    if n == 0:
        print("No unimodal-winning problems found.")
        sys.exit(1)

    c_stars = np.array([p["c_star_unconstrained"] for p in unimodal_winners])
    regime_counts: dict[str, int] = {
        "interior": 0,
        "borderline_low": 0,
        "borderline_high": 0,
        "decaying": 0,
        "saturating": 0,
    }
    for p in unimodal_winners:
        regime_counts[p["regime"]] += 1

    # -------------------------------------------------------------------------
    # Pass 2: re-fit unimodal with c* ≤ 64 for each winner
    # -------------------------------------------------------------------------
    for entry in unimodal_winners:
        R_dict = entry["R_at_N"]
        c_arr = np.array([float(k) for k in R_dict.keys()])
        R_arr = np.array([float(v) for v in R_dict.values()])

        constrained = _fit_unimodal_constrained(c_arr, R_arr)
        entry["c_star_constrained"] = constrained["params"][1]
        entry["unimodal_bic_constrained"] = constrained["bic"]
        entry["unimodal_converged_constrained"] = constrained["converged"]
        entry["unimodal_residual_se_constrained"] = constrained["residual_se"]

        # Recompute BIC weights with constrained unimodal BIC
        bics_constrained = dict(entry["bics_unconstrained"])
        bics_constrained["unimodal"] = constrained["bic"]
        weights_constrained = _bic_weights(bics_constrained)
        entry["weights_constrained"] = weights_constrained

        winner_constrained = max(
            weights_constrained, key=lambda f: weights_constrained[f]
        )
        entry["best_fit_constrained"] = winner_constrained
        entry["winner_flips"] = winner_constrained != "unimodal"
        entry["winner_flips_to"] = winner_constrained if entry["winner_flips"] else None

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------
    n_flips = sum(1 for p in unimodal_winners if p["winner_flips"])
    flip_targets: dict[str, int] = {}
    for p in unimodal_winners:
        if p["winner_flips"]:
            flip_targets[p["winner_flips_to"]] = (
                flip_targets.get(p["winner_flips_to"], 0) + 1
            )

    aggregate = {
        "n_unimodal_winners": n,
        "c_star_stats_unconstrained": {
            "min": float(np.min(c_stars)),
            "p25": float(np.percentile(c_stars, 25)),
            "median": float(np.median(c_stars)),
            "p75": float(np.percentile(c_stars, 75)),
            "max": float(np.max(c_stars)),
        },
        "regime_counts": regime_counts,
        "regime_fractions": {k: round(v / n, 4) for k, v in regime_counts.items()},
        "fraction_interior": round(regime_counts["interior"] / n, 4),
        "fraction_saturating": round(regime_counts["saturating"] / n, 4),
        "fraction_decaying": round(regime_counts["decaying"] / n, 4),
        "fraction_borderline": round(
            (regime_counts["borderline_low"] + regime_counts["borderline_high"]) / n, 4
        ),
        "sensitivity_c_star_ub": _CSTAR_CONSTRAINED_UB,
        "n_winner_flips": n_flips,
        "fraction_winner_flips": round(n_flips / n, 4),
        "flip_targets": flip_targets,
    }

    result = {
        "regime_boundaries": {
            "decaying": f"c* <= {_DECAYING}",
            "borderline_low": f"{_DECAYING} < c* < {_INTERIOR_LOW}",
            "interior": f"{_INTERIOR_LOW} <= c* <= {_INTERIOR_HIGH}",
            "borderline_high": f"{_INTERIOR_HIGH} < c* < {_SATURATING}",
            "saturating": f"c* >= {_SATURATING}",
        },
        "aggregate": aggregate,
        "per_problem": sorted(
            unimodal_winners, key=lambda p: p["c_star_unconstrained"]
        ),
    }
    out_path.write_text(json.dumps(result, indent=2))

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print("Unimodal Sensitivity Analysis  (c* ≤ 64 constraint test)")
    print(sep)

    s = aggregate["c_star_stats_unconstrained"]
    print(f"\n  Unimodal winners (unconstrained) : {n} / 47")
    print()
    print("  c* distribution (unconstrained):")
    print(f"    min    : {s['min']:>8.2f}")
    print(f"    p25    : {s['p25']:>8.2f}")
    print(f"    median : {s['median']:>8.2f}")
    print(f"    p75    : {s['p75']:>8.2f}")
    print(f"    max    : {s['max']:>8.2f}")
    print()
    print("  Regime breakdown (unconstrained c*):")
    print(
        f"    decaying    c* ≤ 4          : "
        f"{regime_counts['decaying']:>2}/{n}  ({aggregate['fraction_decaying']:.1%})"
        f"  ← peak below grid"
    )
    print(
        f"    borderline  (4,8)∪(32,56)   : "
        f"{regime_counts['borderline_low']+regime_counts['borderline_high']:>2}/{n}"
        f"  ({aggregate['fraction_borderline']:.1%})"
    )
    print(
        f"    interior    [8, 32]          : "
        f"{regime_counts['interior']:>2}/{n}  ({aggregate['fraction_interior']:.1%})"
        f"  ← genuine degradation"
    )
    print(
        f"    saturating  c* ≥ 56          : "
        f"{regime_counts['saturating']:>2}/{n}  ({aggregate['fraction_saturating']:.1%})"
        f"  ← peak beyond grid"
    )

    print()
    print(f"  Sensitivity: constrain c* ≤ {_CSTAR_CONSTRAINED_UB}")
    print(f"    Winner flips away from unimodal : {n_flips}/{n}  ({n_flips/n:.1%})")
    if flip_targets:
        for fam, cnt in sorted(flip_targets.items(), key=lambda x: -x[1]):
            print(f"      → {fam:<22} : {cnt}")
    else:
        print("      (no flips)")

    print()
    print("  Per-problem (sorted by unconstrained c*):")
    hdr = f"  {'Problem':<28}  {'c* free':>8}  {'c* ≤64':>8}  {'Regime':<16}  {'Flip?'}"
    print(hdr)
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*16}  {'-'*16}")
    for p in sorted(unimodal_winners, key=lambda p: p["c_star_unconstrained"]):
        flip_str = (
            f"→ {p['winner_flips_to']}" if p["winner_flips"] else "stays unimodal"
        )
        print(
            f"  {p['problem_id']:<28}  {p['c_star_unconstrained']:>8.2f}"
            f"  {p['c_star_constrained']:>8.2f}  {p['regime']:<16}  {flip_str}"
        )

    print()
    print(f"  Written to: {out_path.relative_to(ROOT)}")
    print(sep)


if __name__ == "__main__":
    main()
