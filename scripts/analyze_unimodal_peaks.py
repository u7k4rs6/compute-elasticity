"""Diagnostic: unimodal-peak location analysis.

For each main-pilot problem where unimodal wins BIC selection, extract the
fitted c* (peak location) parameter and characterise whether it falls in the
interior degradation regime or at an effectively saturating / decaying
boundary.

Writes outputs/unimodal_peak_analysis.json.  No API calls, no test changes.

Usage:
    python scripts/analyze_unimodal_peaks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# c* regime boundaries
_INTERIOR_LOW: float = 8.0
_INTERIOR_HIGH: float = 32.0
_SATURATING_CUTOFF: float = 56.0  # c* ≥ this → peak beyond grid, effectively saturating
_DECAYING_CUTOFF: float = 4.0  # c* ≤ this → peak below grid, effectively decaying


def _regime(c_star: float) -> str:
    if c_star <= _DECAYING_CUTOFF:
        return "decaying"
    if c_star < _INTERIOR_LOW:
        return "borderline_low"
    if c_star <= _INTERIOR_HIGH:
        return "interior"
    if c_star < _SATURATING_CUTOFF:
        return "borderline_high"
    return "saturating"


def main() -> None:
    fits_dir = ROOT / "outputs" / "fits"
    out_path = ROOT / "outputs" / "unimodal_peak_analysis.json"

    # Only main-pilot files (exclude side_test_* prefix)
    fit_files = sorted(
        f
        for f in fits_dir.glob("gpqa_diamond_*.json")
        if not f.name.startswith("side_test_")
    )

    if not fit_files:
        print("No main-pilot fit files found in", fits_dir)
        sys.exit(1)

    per_problem: list[dict] = []
    for f in fit_files:
        d = json.loads(f.read_text())
        if d["best_fit_family"] != "unimodal":
            continue
        params = d["fits"]["unimodal"]["params"]
        c_star = float(params[1])
        A = float(params[0])
        sigma = float(params[2])
        per_problem.append(
            {
                "problem_id": d["problem_id"],
                "subject": d["subject"],
                "c_star": c_star,
                "A": A,
                "sigma": sigma,
                "regime": _regime(c_star),
                "unimodal_weight": d["fits"]["unimodal"]["weight"],
                "unimodal_converged": d["fits"]["unimodal"]["converged"],
                "R_at_N": d["R_at_N"],
            }
        )

    n = len(per_problem)
    if n == 0:
        print("No unimodal-winning problems found.")
        sys.exit(1)

    c_stars = np.array([p["c_star"] for p in per_problem])

    regime_counts: dict[str, int] = {
        "interior": 0,
        "borderline_low": 0,
        "borderline_high": 0,
        "decaying": 0,
        "saturating": 0,
    }
    for p in per_problem:
        regime_counts[p["regime"]] += 1

    aggregate = {
        "n_unimodal_winners": n,
        "c_star_stats": {
            "min": float(np.min(c_stars)),
            "p25": float(np.percentile(c_stars, 25)),
            "median": float(np.median(c_stars)),
            "p75": float(np.percentile(c_stars, 75)),
            "max": float(np.max(c_stars)),
        },
        "regime_counts": regime_counts,
        "regime_fractions": {k: v / n for k, v in regime_counts.items()},
        "fraction_interior": regime_counts["interior"] / n,
        "fraction_saturating": regime_counts["saturating"] / n,
        "fraction_decaying": regime_counts["decaying"] / n,
        "fraction_borderline": (
            regime_counts["borderline_low"] + regime_counts["borderline_high"]
        )
        / n,
    }

    result = {
        "regime_boundaries": {
            "decaying": f"c* <= {_DECAYING_CUTOFF}",
            "borderline_low": f"{_DECAYING_CUTOFF} < c* < {_INTERIOR_LOW}",
            "interior": f"{_INTERIOR_LOW} <= c* <= {_INTERIOR_HIGH}",
            "borderline_high": f"{_INTERIOR_HIGH} < c* < {_SATURATING_CUTOFF}",
            "saturating": f"c* >= {_SATURATING_CUTOFF}",
        },
        "aggregate": aggregate,
        "per_problem": sorted(per_problem, key=lambda p: p["c_star"]),
    }
    out_path.write_text(json.dumps(result, indent=2))

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    sep = "=" * 68
    print(f"\n{sep}")
    print("Unimodal Peak Location Analysis")
    print(sep)
    print(f"  Unimodal-winning problems : {n} / 47")
    print()
    s = aggregate["c_star_stats"]
    print("  c* distribution:")
    print(f"    min    : {s['min']:>7.2f}")
    print(f"    p25    : {s['p25']:>7.2f}")
    print(f"    median : {s['median']:>7.2f}")
    print(f"    p75    : {s['p75']:>7.2f}")
    print(f"    max    : {s['max']:>7.2f}")
    print()
    print("  Regime breakdown:")
    print(
        f"    interior   [8, 32]     : "
        f"{regime_counts['interior']:>2} / {n}  "
        f"({aggregate['fraction_interior']:.1%})  ← real degradation"
    )
    print(
        f"    borderline (4,8)∪(32,56): "
        f"{regime_counts['borderline_low'] + regime_counts['borderline_high']:>2} / {n}  "
        f"({aggregate['fraction_borderline']:.1%})"
    )
    print(
        f"    saturating [56, ∞)     : "
        f"{regime_counts['saturating']:>2} / {n}  "
        f"({aggregate['fraction_saturating']:.1%})  ← peak beyond grid"
    )
    print(
        f"    decaying   (-∞, 4]     : "
        f"{regime_counts['decaying']:>2} / {n}  "
        f"({aggregate['fraction_decaying']:.1%})  ← peak below grid"
    )
    print()
    print("  Per-problem (sorted by c*):")
    print(f"  {'Problem':<28}  {'Subj':<8}  {'c*':>7}  {'Regime'}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*7}  {'-'*16}")
    for p in sorted(per_problem, key=lambda p: p["c_star"]):
        print(
            f"  {p['problem_id']:<28}  {p['subject']:<8}  "
            f"{p['c_star']:>7.2f}  {p['regime']}"
        )
    print()
    print(f"  Written to: {out_path.relative_to(ROOT)}")
    print(sep)


if __name__ == "__main__":
    main()
