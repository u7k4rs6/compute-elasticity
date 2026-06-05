"""Phase 12 -- Gate analysis on model-2 samples (Meta-Llama-3-8B-Instruct-Lite).

READ-ONLY on all model-1 outputs. Reads only outputs/samples_model2/.
Writes to outputs/gate_model2/ (mv_curves/ + gate_summary.json).

Mirrors the logic in run_realizable.py + run_gate.py for model 1.
MV curve computation: exact enumeration if C(n,N) <= 5000, else 2000 MC draws.
Gate simulation: k in {4,8}, tau swept 0.50-1.00, 2000 MC draws, seed 42/43.

Usage:
    source .venv/bin/activate
    python scripts/run_model2_gate.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_OUTPUTS_DIR = ROOT / "outputs"
_SAMPLES2_DIR = _OUTPUTS_DIR / "samples_model2"
_GATE2_DIR = _OUTPUTS_DIR / "gate_model2"
_MV2_CURVES_DIR = _GATE2_DIR / "mv_curves"

_T_MAIN: float = 0.7
_N_VALUES: list[int] = [1, 2, 4, 8, 16, 32, 64]
_EXACT_THRESHOLD: int = 5000
_N_MC: int = 2000
_SEED_MV: int = 42

_N_DRAWS_GATE: int = 2000
_SEED_K4: int = 42
_SEED_K8: int = 43
_SEED_BOOTSTRAP: int = 100
_N_BOOTSTRAP: int = 1000
_K_VALUES: list[int] = [4, 8]
_TAU_GRID: list[float] = [round(t, 2) for t in np.arange(0.5, 1.01, 0.05)]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def _main_pilot_ids() -> list[str]:
    locked = json.loads((ROOT / "data" / "problem_ids.json").read_text())
    gate_set = set(
        json.loads((_OUTPUTS_DIR / "gate_minus_1_labels.json").read_text())[
            "gate_problems"
        ]
    )
    return sorted(pid for pid in locked if pid not in gate_set)


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _load_samples(pid: str) -> tuple[list[str | None], str, int]:
    """Return (answers, ground_truth, n_total) for T=0.7 samples."""
    path = _SAMPLES2_DIR / f"{pid}.jsonl"
    if not path.exists():
        return [], "", 0
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if abs(obj.get("temperature", -1) - _T_MAIN) < 1e-6:
                records.append(obj)
        except json.JSONDecodeError:
            continue
    records.sort(key=lambda s: s.get("sample_idx", 0))
    if not records:
        return [], "", 0
    gt = str(records[0]["ground_truth"])
    answers: list[str | None] = [
        (
            r.get("extracted_answer")
            if r.get("extracted_answer") != "UNPARSEABLE"
            else "UNPARSEABLE"
        )
        for r in records
    ]
    return answers, gt, len(answers)


# ---------------------------------------------------------------------------
# MV curve computation (mirrors run_realizable.py)
# ---------------------------------------------------------------------------


def _mv_correct(answers: list[str | None], gt: str, rng: np.random.Generator) -> float:
    """Majority-vote correctness for one draw. Ground-truth-agnostic tie-breaking."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return 0.0
    max_c = max(counts.values())
    tied = sorted(a for a, c in counts.items() if c == max_c)
    winner = tied[0] if len(tied) == 1 else str(rng.choice(tied))
    return 1.0 if winner == gt else 0.0


def _mv_acc_at_n(
    answers: list[str | None],
    gt: str,
    n: int,
    n_total: int,
    rng: np.random.Generator,
) -> float:
    """E[MV(N)] via exact enumeration or MC."""
    from itertools import combinations

    if n > n_total:
        n = n_total

    if comb(n_total, n) <= _EXACT_THRESHOLD:
        # Exact enumeration over all size-N subsets of indices
        total = 0.0
        count = 0
        for subset in combinations(range(n_total), n):
            subset_ans = [answers[i] for i in subset]
            counts = Counter(a for a in subset_ans if a is not None)
            if not counts:
                count += 1
                continue
            max_c = max(counts.values())
            tied = sorted(a for a, c in counts.items() if c == max_c)
            # For exact: take the first tied (lexicographic) -- no rng needed
            winner = tied[0]
            total += 1.0 if winner == gt else 0.0
            count += 1
        return total / count if count > 0 else 0.0
    else:
        # MC approximation
        total = 0.0
        for _ in range(_N_MC):
            idx = rng.choice(n_total, size=n, replace=False)
            sub = [answers[i] for i in idx]
            total += _mv_correct(sub, gt, rng)
        return total / _N_MC


# ---------------------------------------------------------------------------
# Parse rate computation
# ---------------------------------------------------------------------------


def _compute_parse_rate(main_ids: list[str]) -> tuple[float, int, int]:
    """Fraction of extracted_answer that is a valid letter (not UNPARSEABLE/None)."""
    total = 0
    parseable = 0
    for pid in main_ids:
        path = _SAMPLES2_DIR / f"{pid}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if abs(obj.get("temperature", -1) - _T_MAIN) < 1e-6:
                    total += 1
                    ea = obj.get("extracted_answer")
                    if ea and ea != "UNPARSEABLE" and ea in "ABCD":
                        parseable += 1
            except json.JSONDecodeError:
                continue
    rate = parseable / total if total > 0 else 0.0
    return rate, parseable, total


# ---------------------------------------------------------------------------
# Majority-vote helpers for gate simulation
# ---------------------------------------------------------------------------


def _plurality_winner(
    answers: list[str | None], rng: np.random.Generator
) -> tuple[str | None, float]:
    """(winner, agreement_fraction). Ground-truth-agnostic."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None, 0.0
    max_c = max(counts.values())
    tied = sorted(a for a, c in counts.items() if c == max_c)
    winner = tied[0] if len(tied) == 1 else str(rng.choice(tied))
    return winner, max_c / len(answers)


def _simulate_agreement_gate(
    answers_by_pid: dict[str, tuple[list[str | None], str]],
    main_ids: list[str],
    k: int,
    n_draws: int,
    rng: np.random.Generator,
) -> dict[str, list[tuple[float, float, float]]]:
    """Simulate agreement gate. Returns {pid: [(probe_agreement, probe_correct, full64_correct)]}."""
    results: dict[str, list[tuple[float, float, float]]] = {}
    for pid in main_ids:
        if pid not in answers_by_pid:
            continue
        answers_arr, gt = answers_by_pid[pid]
        n_total = len(answers_arr)
        full_n = min(64, n_total)

        draws: list[tuple[float, float, float]] = []
        for _ in range(n_draws):
            idx = rng.choice(n_total, size=full_n, replace=False)
            probe_ans = [answers_arr[i] for i in idx[:k]]
            full_ans = [answers_arr[i] for i in idx]

            probe_winner, probe_agreement = _plurality_winner(probe_ans, rng)
            probe_correct = (
                float(probe_winner == gt) if probe_winner is not None else 0.0
            )

            full_winner, _ = _plurality_winner(full_ans, rng)
            full_correct = float(full_winner == gt) if full_winner is not None else 0.0

            draws.append((probe_agreement, probe_correct, full_correct))
        results[pid] = draws
    return results


def _gate_metrics_at_tau(
    draws_by_pid: dict[str, list[tuple[float, float, float]]],
    main_ids: list[str],
    k: int,
    tau: float,
) -> dict[str, float]:
    total_correct = 0.0
    total_cost = 0.0
    total_stopped = 0
    total_draws = 0

    for pid in main_ids:
        if pid not in draws_by_pid:
            continue
        for pa, pc, fc in draws_by_pid[pid]:
            if pa >= tau:
                total_correct += pc
                total_cost += k
                total_stopped += 1
            else:
                total_correct += fc
                total_cost += 64
            total_draws += 1

    if total_draws == 0:
        return {"acc": 0.0, "compute": float(k), "stop_rate": 0.0}
    return {
        "acc": total_correct / total_draws,
        "compute": total_cost / total_draws,
        "stop_rate": total_stopped / total_draws,
    }


def _bootstrap_gate_ci(
    draws_by_pid: dict[str, list[tuple[float, float, float]]],
    main_ids: list[str],
    k: int,
    tau: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    pids = [p for p in main_ids if p in draws_by_pid]
    n = len(pids)
    boot_accs: list[float] = []
    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        total_c = 0.0
        total_d = 0
        for i in idxs:
            pid = pids[i]
            for pa, pc, fc in draws_by_pid[pid]:
                total_c += pc if pa >= tau else fc
                total_d += 1
        boot_accs.append(total_c / total_d if total_d else 0.0)
    return float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Compute MV curves and gate analysis for model 2."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    _GATE2_DIR.mkdir(parents=True, exist_ok=True)
    _MV2_CURVES_DIR.mkdir(parents=True, exist_ok=True)

    main_ids = _main_pilot_ids()
    if len(main_ids) != 47:
        logger.error("Expected 47 main IDs, got %d", len(main_ids))
        sys.exit(1)

    # ---- Parse rate check ----
    parse_rate, n_parsed, n_total_samples = _compute_parse_rate(main_ids)
    logger.info("Parse rate: %d/%d = %.4f", n_parsed, n_total_samples, parse_rate)
    if parse_rate < 0.90:
        logger.warning(
            "PARSE RATE BELOW 90%%: %.4f (%d/%d). "
            "Model-2 answer format may differ from model-1. "
            "Unparseable samples scored as wrong.",
            parse_rate,
            n_parsed,
            n_total_samples,
        )

    # ---- MV curve computation ----
    logger.info("Computing MV curves (seed=%d, mc_draws=%d)...", _SEED_MV, _N_MC)
    rng_mv = np.random.default_rng(_SEED_MV)

    answers_by_pid: dict[str, tuple[list[str | None], str]] = {}
    mv_curves: dict[str, dict] = {}

    pids_ok: list[str] = []
    for num, pid in enumerate(main_ids, start=1):
        answers, gt, n_samp = _load_samples(pid)
        if n_samp < 1:
            logger.warning(
                "[%d/%d] %s: no samples -- skipping", num, len(main_ids), pid
            )
            continue

        p_exact = sum(1 for a in answers if a == gt) / n_samp
        c_correct = sum(1 for a in answers if a == gt)

        mv_acc: dict[int, float] = {}
        oracle_pass: dict[int, float] = {}
        for n_val in _N_VALUES:
            mv_acc[n_val] = _mv_acc_at_n(answers, gt, n_val, n_samp, rng_mv)
            # Oracle pass@N = 1 - C(n-c,N)/C(n,N) when n >= N and c >= 0
            if n_val <= n_samp:
                num_c = c_correct
                denom = comb(n_samp, n_val)
                complement = (
                    comb(n_samp - num_c, n_val) if n_samp - num_c >= n_val else 0
                )
                oracle_pass[n_val] = 1.0 - complement / denom if denom > 0 else 0.0
            else:
                oracle_pass[n_val] = float("nan")

        mv_gain = mv_acc[64] - mv_acc[1]
        mv_gain_max = max(mv_acc.values()) - mv_acc[1]
        interior_peak = mv_gain_max > 0 and max(mv_acc, key=mv_acc.get) not in (1, 64)

        curve = {
            "problem_id": pid,
            "n_total": n_samp,
            "c_correct": c_correct,
            "p": round(p_exact, 6),
            "mv_acc": {str(n): round(v, 6) for n, v in mv_acc.items()},
            "oracle_pass_at_n": {
                str(n): (
                    round(v, 6) if not isinstance(v, float) or not np.isnan(v) else None
                )
                for n, v in oracle_pass.items()
            },
            "mv_gain": round(mv_gain, 6),
            "mv_gain_max": round(mv_gain_max, 6),
            "interior_peak": interior_peak,
        }
        (_MV2_CURVES_DIR / f"{pid}.json").write_text(json.dumps(curve, indent=2))
        mv_curves[pid] = curve
        answers_by_pid[pid] = (answers, gt)
        pids_ok.append(pid)
        logger.info(
            "[%d/%d] %s  n=%d  p=%.3f  mv_gain=%.4f",
            num,
            len(main_ids),
            pid,
            n_samp,
            p_exact,
            mv_gain,
        )

    logger.info("MV curves saved for %d problems", len(pids_ok))

    # ---- Fixed-budget baseline ----
    fixed_budget_acc: dict[int, float] = {}
    for n_val in _N_VALUES:
        accs = [mv_curves[pid]["mv_acc"][str(n_val)] for pid in pids_ok]
        fixed_budget_acc[n_val] = float(np.mean(accs))

    # ---- Backfire stats ----
    mv_gains = [mv_curves[pid]["mv_gain"] for pid in pids_ok]
    n_backfire = sum(1 for g in mv_gains if g < 0)
    n_pos = sum(1 for g in mv_gains if g > 0)
    n_zero = sum(1 for g in mv_gains if g == 0)

    logger.info(
        "Backfire: %d/%d (%.0f%%)  zero=%d  pos=%d",
        n_backfire,
        len(pids_ok),
        100 * n_backfire / len(pids_ok),
        n_zero,
        n_pos,
    )

    # ---- Oracle gate ----
    oracle_accs: list[float] = []
    oracle_costs: list[float] = []
    n_go_to_64 = 0
    for pid in pids_ok:
        acc1 = mv_curves[pid]["mv_acc"]["1"]
        acc64 = mv_curves[pid]["mv_acc"]["64"]
        if acc64 > acc1:
            oracle_accs.append(acc64)
            oracle_costs.append(64.0)
            n_go_to_64 += 1
        else:
            oracle_accs.append(acc1)
            oracle_costs.append(1.0)

    oracle_acc = float(np.mean(oracle_accs))
    oracle_compute = float(np.mean(oracle_costs))

    # ---- Agreement gate simulation ----
    logger.info("Running agreement gate simulation...")
    gate_results: dict[int, dict[str, list[tuple[float, float, float]]]] = {}
    for k in _K_VALUES:
        seed = _SEED_K4 if k == 4 else _SEED_K8
        rng_gate = np.random.default_rng(seed)
        gate_results[k] = _simulate_agreement_gate(
            answers_by_pid, pids_ok, k, _N_DRAWS_GATE, rng_gate
        )

    gate_tau_metrics: dict[int, dict[float, dict[str, float]]] = {}
    for k in _K_VALUES:
        gate_tau_metrics[k] = {}
        for tau in _TAU_GRID:
            gate_tau_metrics[k][tau] = _gate_metrics_at_tau(
                gate_results[k], pids_ok, k, tau
            )

    # ---- Bootstrap CIs ----
    rng_boot = np.random.default_rng(_SEED_BOOTSTRAP)
    ci_points: dict[str, tuple[float, float, float, float]] = {}
    for k in _K_VALUES:
        best_tau = min(
            _TAU_GRID,
            key=lambda t: abs(gate_tau_metrics[k][t]["compute"] - float(k)),
        )
        m = gate_tau_metrics[k][best_tau]
        lo, hi = _bootstrap_gate_ci(
            gate_results[k], pids_ok, k, best_tau, _N_BOOTSTRAP, rng_boot
        )
        ci_points[f"k={k}_tau={best_tau}"] = (m["compute"], m["acc"], lo, hi)

    # ---- Pareto analysis ----
    def _pareto_dominates(cx: float, cy: float) -> bool:
        for n_val, fb_acc in fixed_budget_acc.items():
            if float(n_val) <= cx and fb_acc >= cy:
                return False
        return True

    gate_pareto: dict[str, Any] = {}
    for k in _K_VALUES:
        any_dominates = False
        best_gain = -999.0
        best_tau_k = _TAU_GRID[0]
        for tau in _TAU_GRID:
            m = gate_tau_metrics[k][tau]
            cx, cy = m["compute"], m["acc"]
            fb_at_or_below = {n: a for n, a in fixed_budget_acc.items() if n <= cx}
            if fb_at_or_below:
                nearest_n = max(fb_at_or_below.keys())
                gain = cy - fixed_budget_acc[nearest_n]
                if gain > best_gain:
                    best_gain = gain
                    best_tau_k = tau
            if _pareto_dominates(cx, cy):
                any_dominates = True
        gate_pareto[f"k{k}"] = {
            "pareto_dominates_fixed_budget": any_dominates,
            "best_gain_vs_matched_compute": round(best_gain, 4),
            "best_gain_tau": best_tau_k,
        }

    acc_64 = fixed_budget_acc[64]
    oracle_ceiling_gain = oracle_acc - acc_64
    best_gate_acc_k8 = max(gate_tau_metrics[8][t]["acc"] for t in _TAU_GRID)
    frac_oracle = (
        (best_gate_acc_k8 - acc_64) / oracle_ceiling_gain
        if oracle_ceiling_gain > 0
        else float("nan")
    )

    # ---- Save gate_summary ----
    summary: dict[str, Any] = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "n_problems": len(pids_ok),
        "parse_rate": round(parse_rate, 4),
        "n_parsed": n_parsed,
        "n_total_samples": n_total_samples,
        "parse_rate_flagged": parse_rate < 0.90,
        "fixed_budget": {
            str(n): {"acc": round(fixed_budget_acc[n], 4), "compute": n}
            for n in _N_VALUES
        },
        "backfire": {
            "n_backfire": n_backfire,
            "n_positive_gain": n_pos,
            "n_zero_gain": n_zero,
            "fraction_backfire": round(n_backfire / len(pids_ok), 4),
            "mean_gain": round(float(np.mean(mv_gains)), 4),
            "median_gain": round(float(np.median(mv_gains)), 4),
            "min_gain": round(float(np.min(mv_gains)), 4),
            "max_gain": round(float(np.max(mv_gains)), 4),
        },
        "oracle_gate": {
            "acc": round(oracle_acc, 4),
            "mean_compute": round(oracle_compute, 2),
            "fraction_go_to_64": round(n_go_to_64 / len(pids_ok), 4),
            "gain_over_fixed_64": round(oracle_acc - acc_64, 4),
            "note": "upper bound: uses ground truth to select best of {N=1, N=64}",
        },
        "realistic_gate": {
            f"k{k}": {
                f"tau_{str(tau).replace('.', '_')}": {
                    "acc": round(gate_tau_metrics[k][tau]["acc"], 4),
                    "compute": round(gate_tau_metrics[k][tau]["compute"], 2),
                    "stop_rate": round(gate_tau_metrics[k][tau]["stop_rate"], 4),
                }
                for tau in _TAU_GRID
            }
            for k in _K_VALUES
        },
        "bootstrap_ci_95": {
            label: {
                "compute": round(v[0], 2),
                "acc": round(v[1], 4),
                "ci_lo": round(v[2], 4),
                "ci_hi": round(v[3], 4),
            }
            for label, v in ci_points.items()
        },
        "pareto_analysis": gate_pareto,
        "verdict": {
            "fixed_budget_acc_64": round(acc_64, 4),
            "oracle_acc": round(oracle_acc, 4),
            "oracle_ceiling_gain": round(oracle_ceiling_gain, 4),
            "best_gate_acc_k8": round(best_gate_acc_k8, 4),
            "fraction_oracle_ceiling_captured_k8": (
                round(frac_oracle, 4)
                if not (isinstance(frac_oracle, float) and np.isnan(frac_oracle))
                else None
            ),
            "realistic_gate_pareto_dominates": any(
                gate_pareto[f"k{k}"]["pareto_dominates_fixed_budget"] for k in _K_VALUES
            ),
        },
    }

    out_path = _GATE2_DIR / "gate_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s", out_path)

    sep = "=" * 72
    print(f"\n{sep}")
    print("Model-2 Gate Analysis Results")
    print(sep)
    print(f"\nParse rate: {parse_rate:.4f} ({n_parsed}/{n_total_samples})", end="")
    print("  [FLAGGED: < 90%]" if parse_rate < 0.90 else "  [OK]")

    print("\nFixed-budget SC baseline:")
    for n_val in _N_VALUES:
        print(f"  N={n_val:<2}  acc={fixed_budget_acc[n_val]:.4f}")

    print(
        f"\nBackfire: {n_backfire}/{len(pids_ok)} (fraction={n_backfire/len(pids_ok):.4f})"
    )
    print(
        f"  mv_gain mean={np.mean(mv_gains):.4f}  median={np.median(mv_gains):.4f}"
        f"  min={np.min(mv_gains):.4f}  max={np.max(mv_gains):.4f}"
    )
    print(
        f"\nOracle gate: acc={oracle_acc:.4f}  compute={oracle_compute:.2f}"
        f"  gain_over_64={oracle_ceiling_gain:+.4f}"
    )

    for k in _K_VALUES:
        print(f"\nAgreement gate k={k}:")
        print(f"  {'tau':>5}  {'acc':>6}  {'compute':>8}  {'stop_rate':>10}")
        for tau in _TAU_GRID:
            m = gate_tau_metrics[k][tau]
            print(
                f"  {tau:>5.2f}  {m['acc']:>6.4f}  {m['compute']:>8.2f}"
                f"  {m['stop_rate']:>10.4f}"
            )
    print(sep)


if __name__ == "__main__":
    main()
