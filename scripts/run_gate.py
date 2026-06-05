"""Backfire characterization + deploy-time agreement gate analysis.

READ-ONLY on all existing outputs. No API calls.

Parts:
  1. Fixed-budget SC baseline (from outputs/realizable/mv_curves/)
  2. Backfire characterization -> outputs/gate/backfire.png
  3. Oracle gate upper bound (uses ground truth to select best-of-{1,64})
  4. Realistic agreement gate simulation (k in {4,8}, tau sweep 0.5-1.0)
  5. Pareto comparison figure -> outputs/gate/pareto.png
  6. Verdict -> outputs/gate/gate_summary.json

Usage:
    source .venv/bin/activate
    python scripts/run_gate.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib  # must set backend before pyplot import
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_OUTPUTS_DIR = ROOT / "outputs"
_MV_CURVES_DIR = _OUTPUTS_DIR / "realizable" / "mv_curves"
_SAMPLES_DIR = _OUTPUTS_DIR / "samples"
_GATE_DIR = _OUTPUTS_DIR / "gate"

_N_DRAWS: int = 2000
_SEED_K4: int = 42
_SEED_K8: int = 43
_SEED_BOOTSTRAP: int = 100
_N_BOOTSTRAP: int = 1000
_N_VALUES: list[int] = [1, 2, 4, 8, 16, 32, 64]
_K_VALUES: list[int] = [4, 8]
_TAU_GRID: list[float] = [round(t, 2) for t in np.arange(0.5, 1.01, 0.05)]
_T_MAIN: float = 0.7

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
# Data loading
# ---------------------------------------------------------------------------


def _load_mv_curves(main_ids: list[str]) -> dict[str, dict]:
    """Load per-problem mv_acc and metadata from mv_curves files."""
    curves: dict[str, dict] = {}
    for pid in main_ids:
        path = _MV_CURVES_DIR / f"{pid}.json"
        if not path.exists():
            logger.warning("%s: mv_curve missing", pid)
            continue
        data = json.loads(path.read_text())
        curves[pid] = {
            "mv_acc": {int(k): float(v) for k, v in data["mv_acc"].items()},
            "mv_gain": float(data["mv_gain"]),
            "p": float(data["p"]),
            "n_total": int(data["n_total"]),
        }
    return curves


def _load_raw_samples(main_ids: list[str]) -> dict[str, tuple[list[str | None], str]]:
    """Return {pid: (answers_list, ground_truth)} for T=0.7 samples."""
    out: dict[str, tuple[list[str | None], str]] = {}
    for pid in main_ids:
        path = _SAMPLES_DIR / f"{pid}.jsonl"
        if not path.exists():
            logger.warning("%s: sample file missing", pid)
            continue
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
            logger.warning("%s: no T=0.7 samples", pid)
            continue
        gt = str(records[0]["ground_truth"])
        answers: list[str | None] = [r.get("extracted_answer") for r in records]
        out[pid] = (answers, gt)
    return out


# ---------------------------------------------------------------------------
# Majority-vote helpers
# ---------------------------------------------------------------------------


def _plurality_winner(
    answers: list[str | None], rng: np.random.Generator
) -> tuple[str | None, float]:
    """Return (plurality_winner, agreement_fraction).

    agreement_fraction = max_count / len(answers).
    Ground-truth-agnostic: None values excluded from Counter.
    """
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None, 0.0
    max_c = max(counts.values())
    tied = sorted(a for a, c in counts.items() if c == max_c)
    winner = tied[0] if len(tied) == 1 else str(rng.choice(tied))
    agreement = max_c / len(answers)
    return winner, agreement


# ---------------------------------------------------------------------------
# Part 4: Agreement gate simulation
# ---------------------------------------------------------------------------


def _simulate_agreement_gate(
    answers_by_pid: dict[str, tuple[list[str | None], str]],
    main_ids: list[str],
    k: int,
    n_draws: int,
    rng: np.random.Generator,
) -> dict[str, list[tuple[float, float, float]]]:
    """Simulate agreement gate for each problem.

    Each draw: sample 64 indices without replacement from the pool.
    Probe = first k indices; full vote = all 64 indices (probe included).
    Returns {pid: [(probe_agreement, probe_correct, full64_correct), ...]}.

    Ground truth is used ONLY to evaluate correctness, never in the gate
    decision itself.
    """
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
    """Aggregate gate metrics across problems and draws for a given tau."""
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


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_gate_ci(
    draws_by_pid: dict[str, list[tuple[float, float, float]]],
    main_ids: list[str],
    k: int,
    tau: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% bootstrap percentile CI on gate accuracy at (k, tau)."""
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
                if pa >= tau:
                    total_c += pc
                else:
                    total_c += fc
                total_d += 1
        boot_accs.append(total_c / total_d if total_d else 0.0)
    lo = float(np.percentile(boot_accs, 2.5))
    hi = float(np.percentile(boot_accs, 97.5))
    return lo, hi


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _make_backfire_figure(
    mv_gains: list[float],
    p_vals: list[float],
    out_path: Path,
) -> None:
    """Two-panel figure: histogram of mv_gain (left) and scatter vs p (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    gains = np.array(mv_gains)
    n_neg = int(np.sum(gains < 0))
    n_pos = int(np.sum(gains > 0))
    n_zero = int(np.sum(gains == 0))

    # Left: histogram
    ax = axes[0]
    neg_gains = gains[gains < 0]
    nonneg_gains = gains[gains >= 0]
    bins = np.linspace(min(gains) - 0.02, max(gains) + 0.02, 20)
    ax.hist(
        neg_gains,
        bins=bins,
        color="#555555",
        edgecolor="white",
        linewidth=0.5,
        label=f"backfire (n={n_neg})",
    )
    ax.hist(
        nonneg_gains,
        bins=bins,
        color="#cccccc",
        edgecolor="white",
        linewidth=0.5,
        label=f"zero/gain (n={n_pos + n_zero})",
    )
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("mv_gain = MV_acc(64) - MV_acc(1)")
    ax.set_ylabel("problems")
    ax.set_title("[EXPLORATORY] Backfire distribution")
    ax.legend(fontsize=9)

    # Right: scatter vs p
    ax = axes[1]
    ax.scatter(p_vals, mv_gains, color="black", s=22, alpha=0.7, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.9, linestyle="--")
    ax.set_xlabel("p = proportion correct answers")
    ax.set_ylabel("mv_gain = MV_acc(64) - MV_acc(1)")
    ax.set_title("[EXPLORATORY] mv_gain vs p")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _make_pareto_figure(
    fixed_budget_curve: dict[int, float],
    oracle_point: tuple[float, float],
    gate_curves: dict[int, list[tuple[float, float]]],
    ci_points: dict[str, tuple[float, float, float, float]],
    out_path: Path,
) -> None:
    """Pareto figure: accuracy vs mean compute.

    fixed_budget_curve: {N: mean_acc}
    oracle_point: (compute, acc)
    gate_curves: {k: [(compute, acc), ...]} sorted by compute
    ci_points: {label: (compute, acc, ci_lo, ci_hi)} for annotated operating points
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Fixed-budget curve
    fb_x = sorted(fixed_budget_curve.keys())
    fb_y = [fixed_budget_curve[n] for n in fb_x]
    ax.plot(
        fb_x,
        fb_y,
        "k-o",
        markersize=5,
        linewidth=1.5,
        label="Fixed-budget SC",
        zorder=4,
    )

    # Oracle gate
    ox, oy = oracle_point
    ax.plot(ox, oy, "k*", markersize=12, label="Oracle gate (upper bound)", zorder=5)

    # Realistic gate curves
    linestyles = {4: "--", 8: ":"}
    markers = {4: "s", 8: "^"}
    for k, pts in gate_curves.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            linestyle=linestyles[k],
            marker=markers[k],
            color="black",
            markersize=5,
            linewidth=1.2,
            label=f"Agreement gate k={k}",
            zorder=3,
        )

    # CI error bars for annotated operating points
    for label, (cx, cy, lo, hi) in ci_points.items():
        ax.errorbar(
            cx,
            cy,
            yerr=[[cy - lo], [hi - cy]],
            fmt="none",
            color="gray",
            capsize=4,
            linewidth=1.2,
        )

    ax.set_xlabel("Mean compute (samples / problem)")
    ax.set_ylabel("Mean MV accuracy")
    ax.set_title("[EXPLORATORY] Accuracy vs compute: gate vs fixed-budget")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full gate analysis and save outputs."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    _GATE_DIR.mkdir(parents=True, exist_ok=True)

    main_ids = _main_pilot_ids()
    if len(main_ids) != 47:
        logger.error("Expected 47 main-pilot IDs, got %d", len(main_ids))
        sys.exit(1)

    logger.info("Loading mv_curves...")
    mv_curves = _load_mv_curves(main_ids)

    logger.info("Loading raw samples...")
    answers_by_pid = _load_raw_samples(main_ids)

    pids_ok = sorted(set(mv_curves.keys()) & set(answers_by_pid.keys()))
    if len(pids_ok) != 47:
        logger.warning(
            "Only %d / 47 problems have both mv_curve and samples", len(pids_ok)
        )

    # ------------------------------------------------------------------
    # PART 1: Fixed-budget baseline
    # ------------------------------------------------------------------
    logger.info("Part 1: Fixed-budget curve")
    fixed_budget_acc: dict[int, float] = {}
    for n in _N_VALUES:
        accs = [mv_curves[pid]["mv_acc"].get(n, float("nan")) for pid in pids_ok]
        fixed_budget_acc[n] = float(np.nanmean(accs))
        logger.info("  N=%-2d  acc=%.4f", n, fixed_budget_acc[n])

    # ------------------------------------------------------------------
    # PART 2: Backfire characterization
    # ------------------------------------------------------------------
    logger.info("Part 2: Backfire characterization")
    mv_gains = [mv_curves[pid]["mv_gain"] for pid in pids_ok]
    p_vals = [mv_curves[pid]["p"] for pid in pids_ok]

    n_backfire = sum(1 for g in mv_gains if g < 0)
    n_pos_gain = sum(1 for g in mv_gains if g > 0)
    n_zero_gain = sum(1 for g in mv_gains if g == 0)

    logger.info(
        "  Backfire (mv_gain < 0): %d / %d problems  (%.0f%%)",
        n_backfire,
        len(pids_ok),
        100 * n_backfire / len(pids_ok),
    )
    logger.info("  Zero gain: %d  Positive gain: %d", n_zero_gain, n_pos_gain)
    logger.info(
        "  mv_gain: mean=%.4f  median=%.4f  min=%.4f  max=%.4f",
        float(np.mean(mv_gains)),
        float(np.median(mv_gains)),
        float(np.min(mv_gains)),
        float(np.max(mv_gains)),
    )

    backfire_path = _GATE_DIR / "backfire.png"
    _make_backfire_figure(mv_gains, p_vals, backfire_path)

    # ------------------------------------------------------------------
    # PART 3: Oracle gate (upper bound, uses ground truth)
    # ------------------------------------------------------------------
    logger.info("Part 3: Oracle gate")
    oracle_accs: list[float] = []
    oracle_costs: list[float] = []
    n_go_to_64 = 0
    for pid in pids_ok:
        acc_1 = mv_curves[pid]["mv_acc"].get(1, 0.0)
        acc_64 = mv_curves[pid]["mv_acc"].get(64, 0.0)
        if acc_64 > acc_1:
            oracle_accs.append(acc_64)
            oracle_costs.append(64.0)
            n_go_to_64 += 1
        else:
            oracle_accs.append(acc_1)
            oracle_costs.append(1.0)

    oracle_acc = float(np.mean(oracle_accs))
    oracle_compute = float(np.mean(oracle_costs))
    logger.info(
        "  Oracle acc=%.4f  mean_compute=%.2f  problems_at_N64=%d/%d",
        oracle_acc,
        oracle_compute,
        n_go_to_64,
        len(pids_ok),
    )

    # ------------------------------------------------------------------
    # PART 4: Realistic agreement gate simulation
    # ------------------------------------------------------------------
    logger.info(
        "Part 4: Agreement gate simulation  (k=%s, n_draws=%d)", _K_VALUES, _N_DRAWS
    )
    gate_results: dict[int, dict[str, list[tuple[float, float, float]]]] = {}
    seeds = {4: _SEED_K4, 8: _SEED_K8}

    for k in _K_VALUES:
        logger.info("  Simulating k=%d ...", k)
        rng_gate = np.random.default_rng(seeds[k])
        gate_results[k] = _simulate_agreement_gate(
            answers_by_pid, pids_ok, k, _N_DRAWS, rng_gate
        )
        logger.info("  k=%d: simulation complete", k)

    # Compute metrics at each tau
    gate_tau_metrics: dict[int, dict[float, dict[str, float]]] = {}
    for k in _K_VALUES:
        gate_tau_metrics[k] = {}
        for tau in _TAU_GRID:
            m = _gate_metrics_at_tau(gate_results[k], pids_ok, k, tau)
            gate_tau_metrics[k][tau] = m
        logger.info(
            "  k=%d: tau=0.50 -> acc=%.4f compute=%.2f | "
            "tau=0.75 -> acc=%.4f compute=%.2f | "
            "tau=1.00 -> acc=%.4f compute=%.2f",
            k,
            gate_tau_metrics[k][0.50]["acc"],
            gate_tau_metrics[k][0.50]["compute"],
            gate_tau_metrics[k][0.75]["acc"],
            gate_tau_metrics[k][0.75]["compute"],
            gate_tau_metrics[k][1.00]["acc"],
            gate_tau_metrics[k][1.00]["compute"],
        )

    # ------------------------------------------------------------------
    # PART 5: Bootstrap CIs + Pareto figure
    # ------------------------------------------------------------------
    logger.info("Part 5: Bootstrap CIs + Pareto figure")

    # Pick one operating point per k for bootstrapping:
    # k=4: tau that gives compute closest to fixed N=4 (compute=4)
    # k=8: tau that gives compute closest to fixed N=8 (compute=8)
    ci_points: dict[str, tuple[float, float, float, float]] = {}
    rng_boot = np.random.default_rng(_SEED_BOOTSTRAP)

    for k in _K_VALUES:
        target_compute = float(k)  # match fixed budget at N=k
        best_tau = min(
            _TAU_GRID,
            key=lambda t: abs(gate_tau_metrics[k][t]["compute"] - target_compute),
        )
        m = gate_tau_metrics[k][best_tau]
        lo, hi = _bootstrap_gate_ci(
            gate_results[k], pids_ok, k, best_tau, _N_BOOTSTRAP, rng_boot
        )
        label = f"k={k}_tau={best_tau}"
        ci_points[label] = (m["compute"], m["acc"], lo, hi)
        logger.info(
            "  Bootstrap CI  k=%d  tau=%.2f  acc=%.4f  CI=[%.4f, %.4f]  compute=%.2f",
            k,
            best_tau,
            m["acc"],
            lo,
            hi,
            m["compute"],
        )

    # Build gate curves for figure
    gate_curves: dict[int, list[tuple[float, float]]] = {}
    for k in _K_VALUES:
        pts = [
            (gate_tau_metrics[k][t]["compute"], gate_tau_metrics[k][t]["acc"])
            for t in _TAU_GRID
        ]
        pts.sort(key=lambda x: x[0])
        gate_curves[k] = pts

    pareto_path = _GATE_DIR / "pareto.png"
    _make_pareto_figure(
        fixed_budget_acc,
        (oracle_compute, oracle_acc),
        gate_curves,
        ci_points,
        pareto_path,
    )

    # ------------------------------------------------------------------
    # PART 6: Verdict + save JSON
    # ------------------------------------------------------------------
    logger.info("Part 6: Verdict")

    acc_64 = fixed_budget_acc[64]
    acc_1 = fixed_budget_acc[1]
    oracle_ceiling_gain = oracle_acc - acc_64

    # For each gate, find the best (compute, acc) relative to fixed-budget
    # Pareto domination: gate point (cx, cy) dominates if there's no fixed-budget
    # point with compute <= cx AND acc >= cy
    def _pareto_dominates(cx: float, cy: float) -> bool:
        """True if (cx, cy) is not dominated by any fixed-budget point."""
        for n, fb_acc in fixed_budget_acc.items():
            if float(n) <= cx and fb_acc >= cy:
                return False
        return True

    gate_pareto: dict[str, Any] = {}
    for k in _K_VALUES:
        any_dominates = False
        best_gain = -999.0
        best_tau_for_gain = _TAU_GRID[0]
        for tau in _TAU_GRID:
            m = gate_tau_metrics[k][tau]
            cx, cy = m["compute"], m["acc"]
            # gain vs nearest fixed-budget (same or less compute)
            fb_at_or_below = {n: a for n, a in fixed_budget_acc.items() if n <= cx}
            if fb_at_or_below:
                nearest_n = max(fb_at_or_below.keys())
                gain = cy - fixed_budget_acc[nearest_n]
                if gain > best_gain:
                    best_gain = gain
                    best_tau_for_gain = tau
            if _pareto_dominates(cx, cy):
                any_dominates = True
        gate_pareto[f"k{k}"] = {
            "pareto_dominates_fixed_budget": any_dominates,
            "best_gain_vs_matched_compute": round(best_gain, 4),
            "best_gain_tau": best_tau_for_gain,
        }
        logger.info(
            "  k=%d: Pareto-dominates=%s  best_gain=%.4f at tau=%.2f",
            k,
            any_dominates,
            best_gain,
            best_tau_for_gain,
        )

    # Fraction of oracle ceiling captured at best gate point (k=8, highest acc)
    best_gate_acc_k8 = max(gate_tau_metrics[8][t]["acc"] for t in _TAU_GRID)
    if oracle_ceiling_gain > 0:
        frac_oracle = (best_gate_acc_k8 - acc_64) / oracle_ceiling_gain
    else:
        frac_oracle = float("nan")

    pareto_dominates_any = any(
        gate_pareto[f"k{k}"]["pareto_dominates_fixed_budget"] for k in _K_VALUES
    )

    summary: dict[str, Any] = {
        "n_problems": len(pids_ok),
        "n_draws_per_problem": _N_DRAWS,
        "fixed_budget": {
            str(n): {"acc": round(fixed_budget_acc[n], 4), "compute": n}
            for n in _N_VALUES
        },
        "backfire": {
            "n_backfire": n_backfire,
            "n_positive_gain": n_pos_gain,
            "n_zero_gain": n_zero_gain,
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
            "note": "upper bound: uses ground truth to select best of {N=1, N=64} per problem",
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
            "realistic_gate_pareto_dominates_fixed_budget": pareto_dominates_any,
            "best_gate_acc_k8": round(best_gate_acc_k8, 4),
            "fixed_budget_acc_64": round(acc_64, 4),
            "oracle_acc": round(oracle_acc, 4),
            "oracle_ceiling_gain": round(oracle_ceiling_gain, 4),
            "fraction_oracle_ceiling_captured_k8": (
                round(frac_oracle, 4)
                if not (isinstance(frac_oracle, float) and np.isnan(frac_oracle))
                else None
            ),
            "summary": (
                "Realistic gate DOES Pareto-dominate fixed-budget at some operating point."
                if pareto_dominates_any
                else "Realistic gate does NOT Pareto-dominate fixed-budget at any operating point. "
                "High probe agreement does not reliably signal correctness (backfire is "
                "indistinguishable from a confident majority-wrong vote)."
            ),
        },
    }

    out_path = _GATE_DIR / "gate_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s", out_path)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print("Gate Analysis Results")
    print(sep)

    print("\nFixed-budget SC baseline:")
    for n in _N_VALUES:
        print(f"  N={n:<2}  acc={fixed_budget_acc[n]:.4f}  compute={n}")

    print(
        f"\nBackfire: {n_backfire}/{len(pids_ok)} problems (mv_gain < 0)  "
        f"| zero={n_zero_gain}  pos={n_pos_gain}"
    )
    print(
        f"  mv_gain  mean={np.mean(mv_gains):.4f}  median={np.median(mv_gains):.4f}  "
        f"min={np.min(mv_gains):.4f}  max={np.max(mv_gains):.4f}"
    )

    print("\nOracle gate (upper bound, uses ground truth):")
    print(
        f"  acc={oracle_acc:.4f}  mean_compute={oracle_compute:.2f}  "
        f"gain_over_N64={oracle_acc - acc_64:+.4f}"
    )

    for k in _K_VALUES:
        print(f"\nAgreement gate k={k}:")
        print(f"  {'tau':>5}  {'acc':>6}  {'compute':>8}  {'stop_rate':>10}")
        for tau in _TAU_GRID:
            m = gate_tau_metrics[k][tau]
            print(
                f"  {tau:>5.2f}  {m['acc']:>6.4f}  {m['compute']:>8.2f}  "
                f"{m['stop_rate']:>10.4f}"
            )

    print("\nPareto analysis:")
    for k in _K_VALUES:
        gp = gate_pareto[f"k{k}"]
        print(
            f"  k={k}: dominates={gp['pareto_dominates_fixed_budget']}  "
            f"best_gain={gp['best_gain_vs_matched_compute']:+.4f}  "
            f"at tau={gp['best_gain_tau']:.2f}"
        )

    print(f"\nVerdict: {summary['verdict']['summary']}")
    print(f"  Oracle ceiling gain={oracle_ceiling_gain:+.4f}")
    frac = summary["verdict"]["fraction_oracle_ceiling_captured_k8"]
    if frac is not None:
        print(f"  Fraction of oracle ceiling captured (k=8 best): {frac:.4f}")
    print(sep)


if __name__ == "__main__":
    main()
