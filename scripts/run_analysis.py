"""Phase 9 — H1–H6 confirmatory analysis.

Reads pre-computed outputs from Phases 6–8 and evaluates all six
pre-registered hypotheses against locked thresholds from pilot/config.py.

No API calls. Pure local computation (H6 bootstrap takes ~1–2 minutes).

Usage:
    python scripts/run_analysis.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom as sp_binom

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

_N_BOOTSTRAP_H6: int = 1000
_N_SAMPLES_SIDE: int = 16
_TEMPERATURES_SIDE: tuple[float, ...] = (0.3, 0.7, 1.0)
_N_VALUES_SIDE: tuple[int, ...] = (1, 2, 4, 8, 16)

_OUTPUTS_DIR: Path = ROOT / "outputs"
_FITS_DIR: Path = _OUTPUTS_DIR / "fits"
_DIVERSITY_DIR: Path = _OUTPUTS_DIR / "diversity"
_ENTROPY_DIR: Path = _OUTPUTS_DIR / "entropy_baseline"
_SAMPLES_DIR: Path = _OUTPUTS_DIR / "samples"


# ---------------------------------------------------------------------------
# Problem-ID helpers
# ---------------------------------------------------------------------------


def _load_gate_ids() -> list[str]:
    """Load Gate-1 problem IDs from pre-registration evidence file."""
    path = _OUTPUTS_DIR / "gate_minus_1_labels.json"
    return json.loads(path.read_text())["gate_problems"]


def _load_locked_ids() -> list[str]:
    """Load the 50 locked problem IDs from data/problem_ids.json."""
    return json.loads((ROOT / "data" / "problem_ids.json").read_text())


def _load_side_test_ids() -> list[str]:
    """Load the 10 side-test problem IDs (gate + recon) in Phase 7 order."""
    data = json.loads((_OUTPUTS_DIR / "recon_results.json").read_text())
    return data["gate_1_problems_reused"] + data["new_recon_problems"]


def _select_main_pilot_ids(locked_ids: list[str], gate_ids: list[str]) -> list[str]:
    """Return the 47 main-pilot IDs (locked minus gate-1), sorted."""
    gate = set(gate_ids)
    return sorted(pid for pid in locked_ids if pid not in gate)


# ---------------------------------------------------------------------------
# H1 — Median residual SE
# ---------------------------------------------------------------------------


def _evaluate_h1(fits_summary: dict) -> dict[str, Any]:
    """H1: Median residual SE < 0.10 across 47 main-pilot problems."""
    from pilot.config import HYPOTHESIS_THRESHOLDS

    th = HYPOTHESIS_THRESHOLDS["H1"]
    measured = fits_summary["median_residual_se_main"]
    if measured < th["pass_threshold"]:
        verdict = "PASS"
    elif measured >= th["fail_threshold"]:
        verdict = "FAIL"
    else:
        verdict = "AMBIGUOUS"
    return {
        "measured": measured,
        "threshold_pass": th["pass_threshold"],
        "threshold_fail": th["fail_threshold"],
        "deciding_metric": th["metric"],
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# H2 — Mode diversity
# ---------------------------------------------------------------------------


def _evaluate_h2(fits_summary: dict) -> dict[str, Any]:
    """H2: PASS if criterion (a) OR (b) holds."""
    from pilot.config import HYPOTHESIS_THRESHOLDS

    th_a = HYPOTHESIS_THRESHOLDS["H2a"]
    th_b = HYPOTHESIS_THRESHOLDS["H2b"]

    # Criterion (a): ≥2 families with mean BIC weight ≥ 0.10
    weights = fits_summary["family_mean_bic_weights"]
    families_above = [f for f, w in weights.items() if w >= th_a["min_weight"]]
    verdict_a = "PASS" if len(families_above) >= th_a["min_families"] else "FAIL"

    # Criterion (b): ≥30% of problems have ≥2 close families (BIC margin 2.0)
    n_total = fits_summary["n_main_pilot_problems"]
    n_close = fits_summary["n_problems_with_2plus_close_families"]
    fraction_b = n_close / n_total
    verdict_b = "PASS" if fraction_b >= th_b["min_fraction"] else "FAIL"

    overall = "PASS" if "PASS" in (verdict_a, verdict_b) else "FAIL"

    return {
        "criterion_a": {
            "families_above_0.10": families_above,
            "n_families": len(families_above),
            "min_required": th_a["min_families"],
            "verdict": verdict_a,
        },
        "criterion_b": {
            "n_close_problems": n_close,
            "n_total": n_total,
            "fraction_2plus_close": round(fraction_b, 4),
            "min_fraction": th_b["min_fraction"],
            "verdict": verdict_b,
        },
        "overall_verdict": overall,
    }


# ---------------------------------------------------------------------------
# H3 — Embedding diversity vs entropy as elasticity predictor
# ---------------------------------------------------------------------------


def _evaluate_h3(main_pilot_ids: list[str]) -> dict[str, Any]:
    """H3: AUC(diversity) >= 0.60 AND delta-AUC(diversity - entropy) >= 0.03."""
    from pilot.analysis import compute_auc
    from pilot.config import HYPOTHESIS_THRESHOLDS

    th = HYPOTHESIS_THRESHOLDS["H3"]

    diversity_vals: list[float] = []
    entropy_vals: list[float] = []
    nll_vals: list[float] = []
    elasticity_vals: list[float] = []

    for pid in main_pilot_ids:
        div_path = _DIVERSITY_DIR / f"{pid}.json"
        ent_path = _ENTROPY_DIR / f"{pid}.json"
        fit_path = _FITS_DIR / f"{pid}.json"

        if not all(p.exists() for p in (div_path, ent_path, fit_path)):
            logger.warning("Missing data for %s — skipping from H3", pid)
            continue

        div_d = json.loads(div_path.read_text())
        ent_d = json.loads(ent_path.read_text())
        fit_d = json.loads(fit_path.read_text())

        diversity_vals.append(float(div_d["diversity"]))
        entropy_vals.append(float(ent_d.get("mean_per_token_entropy", float("nan"))))
        nll_vals.append(float(ent_d["mean_per_token_nll"]))
        elasticity_vals.append(float(fit_d["mean_elasticity_8_64"]))

    diversity_arr = np.array(diversity_vals)
    entropy_arr = np.array(entropy_vals)
    nll_arr = np.array(nll_vals)
    elasticity_arr = np.array(elasticity_vals)

    median_elasticity = float(np.median(elasticity_arr))
    labels = (elasticity_arr > median_elasticity).astype(int)

    # Replace NaN entropy values with per-column median before AUC.
    entropy_clean = np.where(
        np.isnan(entropy_arr), np.nanmedian(entropy_arr), entropy_arr
    )

    auc_div = compute_auc(diversity_arr, labels)
    auc_ent = compute_auc(entropy_clean, labels)
    auc_nll = compute_auc(nll_arr, labels)

    delta_ent = auc_div - auc_ent
    delta_nll = auc_div - auc_nll

    pass_auc = th["pass_auc"]
    pass_delta = th["pass_delta_auc"]
    fail_auc = th["fail_auc"]

    if auc_div >= pass_auc and delta_ent >= pass_delta:
        verdict = "PASS"
    elif auc_div < fail_auc or delta_ent <= 0:
        verdict = "FAIL"
    else:
        verdict = "AMBIGUOUS"

    return {
        "n_problems": len(elasticity_vals),
        "median_elasticity_threshold": median_elasticity,
        "n_above_median": int(np.sum(labels)),
        "auc_diversity": round(auc_div, 4),
        "auc_entropy": round(auc_ent, 4),
        "auc_nll": round(auc_nll, 4),
        "delta_auc_entropy": round(delta_ent, 4),
        "delta_auc_nll": round(delta_nll, 4),
        "threshold_pass_auc": pass_auc,
        "threshold_pass_delta": pass_delta,
        "threshold_fail_auc": fail_auc,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# H4 — Deferred
# ---------------------------------------------------------------------------


def _evaluate_h4() -> dict[str, Any]:
    """H4: Domain distribution shift — deferred to full multi-model study."""
    return {"verdict": "DEFERRED", "note": "deferred to full multi-model study"}


# ---------------------------------------------------------------------------
# H5 — Unimodal wins >= 5%
# ---------------------------------------------------------------------------


def _evaluate_h5(fits_summary: dict) -> dict[str, Any]:
    """H5: fraction of unimodal BIC wins >= 0.05."""
    from pilot.config import HYPOTHESIS_THRESHOLDS

    th = HYPOTHESIS_THRESHOLDS["H5"]
    n_total = fits_summary["n_main_pilot_problems"]
    n_wins = fits_summary["n_problems_unimodal_wins"]
    fraction = n_wins / n_total

    if fraction >= th["pass_threshold"]:
        verdict = "PASS"
    elif fraction == th["fail_threshold"]:
        verdict = "FAIL"
    else:
        verdict = "AMBIGUOUS"

    interior_count = 0
    peak_path = _OUTPUTS_DIR / "unimodal_sensitivity.json"
    if peak_path.exists():
        peak_d = json.loads(peak_path.read_text())
        interior_count = (
            peak_d.get("aggregate", {}).get("regime_counts", {}).get("interior", 0)
        )

    return {
        "unimodal_win_count": n_wins,
        "n_total": n_total,
        "unimodal_fraction": round(fraction, 4),
        "threshold_pass": th["pass_threshold"],
        "interior_peak_count": interior_count,
        "interpretation_note": "see outputs/unimodal_sensitivity.json",
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# H6 — Curve param stability across temperatures (bootstrap)
# ---------------------------------------------------------------------------


def _r_at_n_from_p(p: float, N: int) -> float:
    """R(N) = P(majority of N i.i.d. Bernoulli(p) votes correct).

    Exact binomial formula; ties (even N, exactly N/2 correct) count 50%.
    """
    if N == 1:
        return p
    k_maj = N // 2 + 1
    prob = float(sp_binom.sf(k_maj - 1, N, p))
    if N % 2 == 0:
        prob += 0.5 * float(sp_binom.pmf(N // 2, N, p))
    return prob


def _bootstrap_param_ci(
    correct: list[bool],
    family_name: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, tuple[float, float]]:
    """Bootstrap 2.5th/97.5th-pct CI (95%) for each param of the given curve family.

    Resamples binary correct/incorrect outcomes, reconstructs R(N) via the
    binomial formula (exact under i.i.d. samples), then re-fits the family.
    """
    from pilot.fitting import (
        fit_constant,
        fit_gompertz,
        fit_logistic,
        fit_shifted_logistic,
        fit_unimodal,
    )

    fit_fn_map = {
        "constant": fit_constant,
        "logistic": fit_logistic,
        "gompertz": fit_gompertz,
        "shifted_logistic": fit_shifted_logistic,
        "unimodal": fit_unimodal,
    }
    fit_fn = fit_fn_map[family_name]
    n = len(correct)
    correct_arr = np.array(correct, dtype=float)
    c_arr = np.array(_N_VALUES_SIDE, dtype=float)

    param_boot: list[list[float]] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        p = float(correct_arr[indices].mean())
        R_arr = np.array(
            [_r_at_n_from_p(p, int(N)) for N in _N_VALUES_SIDE], dtype=float
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_fn(c_arr, R_arr)
        param_boot.append(list(result.params))

    arr = np.array(param_boot)  # shape (n_bootstrap, n_params)
    ci: dict[str, tuple[float, float]] = {}
    for i in range(arr.shape[1]):
        ci[f"p{i}"] = (
            float(np.percentile(arr[:, i], 2.5)),
            float(np.percentile(arr[:, i], 97.5)),
        )
    return ci


def _ci_overlap_rate(
    ci1: dict[str, tuple[float, float]], ci2: dict[str, tuple[float, float]]
) -> float:
    """Fraction of params whose 95% bootstrap CIs overlap (binary indicator)."""
    overlaps = []
    for pk in ci1:
        if pk not in ci2:
            continue
        lo = max(ci1[pk][0], ci2[pk][0])
        hi = min(ci1[pk][1], ci2[pk][1])
        overlaps.append(1.0 if hi >= lo else 0.0)
    return float(np.mean(overlaps)) if overlaps else 0.0


def _load_side_correct(pid: str, temperature: float) -> list[bool]:
    """Return correct[] for pid at temperature, sorted by sample_idx."""
    path = _SAMPLES_DIR / f"{pid}.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if abs(obj.get("temperature", -1) - temperature) < 1e-6:
            rows.append(obj)
    rows.sort(key=lambda s: s.get("sample_idx", 0))
    return [bool(r.get("correct", False)) for r in rows]


def _evaluate_h6(side_test_ids: list[str]) -> dict[str, Any]:
    """H6: bootstrap-CI overlap rate for fitted params across temperatures >= 0.70."""
    from pilot.config import HYPOTHESIS_THRESHOLDS

    th = HYPOTHESIS_THRESHOLDS["H6"]

    # Replicate Phase 7 subsample RNG exactly (same seed, same iteration order).
    side_subsample_rng = np.random.default_rng(42)
    boot_rng = np.random.default_rng(7)

    temp_pairs = [(0.3, 0.7), (0.7, 1.0), (0.3, 1.0)]
    pair_keys = ["0.3_0.7", "0.7_1.0", "0.3_1.0"]

    all_overlaps: list[float] = []
    per_pair: dict[str, list[float]] = {k: [] for k in pair_keys}
    per_problem: list[dict] = []

    for pid in side_test_ids:
        fit07_path = _FITS_DIR / f"side_test_{pid}_t0.7.json"
        if not fit07_path.exists():
            logger.warning("Missing T=0.7 fit for %s — skipping from H6", pid)
            # Advance subsample RNG for all 3 temperatures to keep state consistent.
            for t in _TEMPERATURES_SIDE:
                all_c = _load_side_correct(pid, t)
                side_subsample_rng.choice(len(all_c) or 1, size=1, replace=False)
            continue

        ref_family = json.loads(fit07_path.read_text())["best_fit_family"]

        # Subsample 16 per temperature — must match Phase 7 RNG calls exactly.
        correct_by_temp: dict[float, list[bool]] = {}
        for t in _TEMPERATURES_SIDE:
            all_correct = _load_side_correct(pid, t)
            n_avail = len(all_correct)
            take = min(n_avail, _N_SAMPLES_SIDE)
            indices = side_subsample_rng.choice(n_avail, size=take, replace=False)
            correct_by_temp[t] = [all_correct[i] for i in sorted(indices)]

        # Bootstrap CI for each temperature using the reference family.
        ci_by_temp: dict[float, dict[str, tuple[float, float]]] = {}
        for t in _TEMPERATURES_SIDE:
            ci_by_temp[t] = _bootstrap_param_ci(
                correct_by_temp[t], ref_family, boot_rng, _N_BOOTSTRAP_H6
            )

        # Compute pairwise overlap rates.
        problem_pair_overlaps: dict[str, float] = {}
        for pk, (t1, t2) in zip(pair_keys, temp_pairs):
            rate = _ci_overlap_rate(ci_by_temp[t1], ci_by_temp[t2])
            problem_pair_overlaps[pk] = round(rate, 4)
            per_pair[pk].append(rate)
            all_overlaps.append(rate)

        per_problem.append(
            {
                "problem_id": pid,
                "ref_family": ref_family,
                "pair_overlap_rates": problem_pair_overlaps,
            }
        )
        logger.info(
            "  H6 %s (%s): %s",
            pid,
            ref_family,
            {k: f"{v:.2f}" for k, v in problem_pair_overlaps.items()},
        )

    mean_overlap = float(np.mean(all_overlaps)) if all_overlaps else 0.0
    per_pair_mean = {
        k: round(float(np.mean(v)), 4) if v else 0.0 for k, v in per_pair.items()
    }

    if mean_overlap >= th["pass_threshold"]:
        verdict = "PASS"
    elif mean_overlap < th["fail_threshold"]:
        verdict = "FAIL"
    else:
        verdict = "AMBIGUOUS"

    return {
        "mean_ci_overlap_rate": round(mean_overlap, 4),
        "per_pair_overlap": per_pair_mean,
        "n_problems": len(per_problem),
        "threshold_pass": th["pass_threshold"],
        "threshold_fail": th["fail_threshold"],
        "n_bootstrap_resamples": _N_BOOTSTRAP_H6,
        "per_problem": per_problem,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Go / No-Go decision
# ---------------------------------------------------------------------------


def _go_no_go(h1: dict, h2: dict) -> str:
    """Primary tier GO/NO_GO: both H1 and H2 must PASS for GO."""
    v1 = h1["verdict"]
    v2 = h2["overall_verdict"]
    if v1 == "PASS" and v2 == "PASS":
        return "GO"
    if v1 == "FAIL" or v2 == "FAIL":
        return "NO_GO"
    return "CONDITIONAL"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_VERDICT_SYMBOL = {
    "PASS": "PASS",
    "FAIL": "FAIL",
    "AMBIGUOUS": "AMBG",
    "DEFERRED": "DEFR",
}


def _print_summary(results: dict[str, Any]) -> None:
    """Print human-readable verdict table to stdout."""
    from pilot.config import HYPOTHESIS_THRESHOLDS

    sep = "=" * 76
    print(f"\n{sep}")
    print("Phase 9 — Confirmatory Analysis (H1–H6)")
    print(sep)

    h1 = results["H1"]
    h2 = results["H2"]
    h3 = results["H3"]
    h5 = results["H5"]
    h6 = results["H6"]

    rows = [
        (
            "H1",
            "Median residual SE",
            f"{h1['measured']:.4f}",
            f"< {h1['threshold_pass']}",
            h1["verdict"],
        ),
        (
            "H2a",
            "Families w/ mean BIC weight >=0.10",
            str(h2["criterion_a"]["n_families"]),
            f">= {h2['criterion_a']['min_required']}",
            h2["criterion_a"]["verdict"],
        ),
        (
            "H2b",
            "Fraction >=2 close families",
            f"{h2['criterion_b']['fraction_2plus_close']:.2%}",
            f">= {h2['criterion_b']['min_fraction']:.0%}",
            h2["criterion_b"]["verdict"],
        ),
        (
            "H2",
            "Mode diversity (a OR b)",
            "—",
            "—",
            h2["overall_verdict"],
        ),
        (
            "H3",
            "AUC(diversity) vs AUC(entropy)",
            (f"{h3['auc_diversity']:.3f}" f" (delta={h3['delta_auc_entropy']:+.3f})"),
            (
                f"AUC>={HYPOTHESIS_THRESHOLDS['H3']['pass_auc']}"
                f" & delta>={HYPOTHESIS_THRESHOLDS['H3']['pass_delta_auc']}"
            ),
            h3["verdict"],
        ),
        ("H4", "Domain distribution shift", "—", "—", "DEFERRED"),
        (
            "H5",
            "Unimodal-win fraction",
            f"{h5['unimodal_win_count']}/{h5['n_total']}"
            f" ({h5['unimodal_fraction']:.1%})",
            f">= {h5['threshold_pass']:.0%}",
            h5["verdict"],
        ),
        (
            "H6",
            "Param CI overlap across T",
            f"{h6['mean_ci_overlap_rate']:.3f}",
            f">= {h6['threshold_pass']}",
            h6["verdict"],
        ),
    ]

    print(f"\n  {'H':<5}  {'Metric':<38}  {'Measured':<28}  {'Threshold':<26}  Verdict")
    print(f"  {'-'*5}  {'-'*38}  {'-'*28}  {'-'*26}  {'-'*8}")
    for h, metric, measured, threshold, verdict in rows:
        sym = _VERDICT_SYMBOL.get(verdict, verdict)
        print(f"  {h:<5}  {metric:<38}  {measured:<28}  {threshold:<26}  {sym}")

    summary = results["summary"]
    print()
    print(f"  Hypotheses PASS     : {summary['n_passed']}")
    print(f"  Hypotheses FAIL     : {summary['n_failed']}")
    print(f"  Hypotheses AMBIGUOUS: {summary['n_ambiguous']}")
    print(f"  Hypotheses DEFERRED : {summary['n_deferred']}")
    print()
    print(f"  Overall go/no-go    : {summary['overall_go_no_go']}")
    print()
    print(
        f"  H3 detail — AUC(diversity)={h3['auc_diversity']:.3f}  "
        f"AUC(entropy)={h3['auc_entropy']:.3f}  "
        f"AUC(NLL)={h3['auc_nll']:.3f}"
    )
    print()
    print("  H6 per-pair CI overlap:")
    for pair, rate in h6["per_pair_overlap"].items():
        print(f"    T={pair}: {rate:.3f}")

    print("\n  Written to: outputs/hypothesis_results.json")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Phase 9 confirmatory analysis.

    Pass --h6-only to re-run only H6 and patch the existing
    outputs/hypothesis_results.json (preserves H1–H5 values).
    """
    import argparse

    from pilot.config import OUTPUTS_DIR, SCHEMA_VERSION

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h6-only",
        action="store_true",
        help="Re-evaluate H6 only; patch existing hypothesis_results.json.",
    )
    args = parser.parse_args()

    side_test_ids = _load_side_test_ids()

    if args.h6_only:
        out_path = OUTPUTS_DIR / "hypothesis_results.json"
        if not out_path.exists():
            logger.error("No existing %s — run without --h6-only first.", out_path)
            sys.exit(1)
        results = json.loads(out_path.read_text())

        logger.info("Evaluating H6 only (bootstrap, ~45 min)…")
        h6 = _evaluate_h6(side_test_ids)
        results["H6"] = h6
        results["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        verdicts = [
            results["H1"]["verdict"],
            results["H2"]["overall_verdict"],
            results["H3"]["verdict"],
            results["H5"]["verdict"],
            h6["verdict"],
        ]
        results["summary"]["n_passed"] = verdicts.count("PASS")
        results["summary"]["n_failed"] = verdicts.count("FAIL")
        results["summary"]["n_ambiguous"] = verdicts.count("AMBIGUOUS")
        results["summary"]["overall_go_no_go"] = _go_no_go(results["H1"], results["H2"])

        out_path.write_text(json.dumps(results, indent=2))
        _print_summary(results)
        return

    locked_ids = _load_locked_ids()
    gate_ids = _load_gate_ids()
    main_pilot_ids = _select_main_pilot_ids(locked_ids, gate_ids)

    fits_summary_path = OUTPUTS_DIR / "fits_summary.json"
    if not fits_summary_path.exists():
        logger.error(
            "Missing %s — run scripts/run_fitting.py first.", fits_summary_path
        )
        sys.exit(1)
    fits_summary = json.loads(fits_summary_path.read_text())

    logger.info("Evaluating H1…")
    h1 = _evaluate_h1(fits_summary)

    logger.info("Evaluating H2…")
    h2 = _evaluate_h2(fits_summary)

    logger.info("Evaluating H3 (AUC)…")
    h3 = _evaluate_h3(main_pilot_ids)

    h4 = _evaluate_h4()

    logger.info("Evaluating H5…")
    h5 = _evaluate_h5(fits_summary)

    logger.info("Evaluating H6 (bootstrap, ~45 min)…")
    h6 = _evaluate_h6(side_test_ids)

    verdicts = [
        h1["verdict"],
        h2["overall_verdict"],
        h3["verdict"],
        h5["verdict"],
        h6["verdict"],
    ]
    n_passed = verdicts.count("PASS")
    n_failed = verdicts.count("FAIL")
    n_ambiguous = verdicts.count("AMBIGUOUS")

    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "h1_to_h6_confirmatory",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "H5": h5,
        "H6": h6,
        "summary": {
            "n_passed": n_passed,
            "n_failed": n_failed,
            "n_ambiguous": n_ambiguous,
            "n_deferred": 1,
            "overall_go_no_go": _go_no_go(h1, h2),
        },
    }

    out_path = OUTPUTS_DIR / "hypothesis_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    _print_summary(results)


if __name__ == "__main__":
    main()
