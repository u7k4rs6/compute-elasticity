"""Phase 12 -- Cross-model comparison + mechanism (calibration) analysis.

Reads:
  outputs/realizable/mv_curves/      (model-1 MV curves)
  outputs/gate/gate_summary.json     (model-1 gate results)
  outputs/samples/                   (model-1 raw answers, for calibration)
  outputs/gate_model2/mv_curves/     (model-2 MV curves)
  outputs/gate_model2/gate_summary.json
  outputs/samples_model2/            (model-2 raw answers, for calibration)

Writes:
  outputs/gate_model2/backfire_both.png
  outputs/gate_model2/pareto_both.png
  outputs/gate_model2/calibration_both.png
  outputs/gate_model2/cross_model_summary.json

Usage:
    source .venv/bin/activate
    python scripts/run_cross_model.py
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
_MV1_DIR = _OUTPUTS_DIR / "realizable" / "mv_curves"
_GATE1_DIR = _OUTPUTS_DIR / "gate"
_SAMPLES1_DIR = _OUTPUTS_DIR / "samples"
_MV2_DIR = _OUTPUTS_DIR / "gate_model2" / "mv_curves"
_GATE2_DIR = _OUTPUTS_DIR / "gate_model2"
_SAMPLES2_DIR = _OUTPUTS_DIR / "samples_model2"

_T_MAIN: float = 0.7
_N_VALUES: list[int] = [1, 2, 4, 8, 16, 32, 64]
_K_VALUES: list[int] = [4, 8]
_TAU_GRID: list[float] = [round(t, 2) for t in np.arange(0.5, 1.01, 0.05)]
_N_BOOTSTRAP: int = 1000
_SEED_BOOT: int = 42

CONFIDENCE_BINS = [(0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
CONFIDENCE_BIN_LABELS = ["[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.00]"]

MODEL1_LABEL = "Qwen2.5-7B"
MODEL2_LABEL = "Llama-3-8B"

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


def _load_mv_curves(mv_dir: Path, main_ids: list[str]) -> dict[str, dict]:
    curves = {}
    for pid in main_ids:
        path = mv_dir / f"{pid}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        curves[pid] = data
    return curves


def _load_answers_and_gt(
    samples_dir: Path, main_ids: list[str]
) -> dict[str, tuple[list[str | None], str, int]]:
    """Return {pid: (answers, gt, n_total)} for T=0.7 samples."""
    out = {}
    for pid in main_ids:
        path = samples_dir / f"{pid}.jsonl"
        if not path.exists():
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
            continue
        gt = str(records[0]["ground_truth"])
        answers: list[str | None] = [r.get("extracted_answer") for r in records]
        out[pid] = (answers, gt, len(answers))
    return out


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------


def _confidence_and_correctness(
    answers: list[str | None], gt: str
) -> tuple[float, bool]:
    """Compute plurality confidence and whether plurality is correct."""
    counts = Counter(a for a in answers if a is not None and a != "")
    if not counts:
        return 0.0, False
    n_total = len(answers)
    max_c = max(counts.values())
    confidence = max_c / n_total
    tied = sorted(a for a, c in counts.items() if c == max_c)
    plurality_answer = tied[0]  # lexicographic for calibration (deterministic)
    return confidence, plurality_answer == gt


def _calibration_table(
    answers_by_pid: dict[str, tuple[list[str | None], str, int]],
    pids: list[str],
) -> list[dict]:
    """Compute calibration table: per-bin {lo, hi, n, fraction_correct}."""
    bins: list[list[bool]] = [[] for _ in CONFIDENCE_BINS]

    for pid in pids:
        if pid not in answers_by_pid:
            continue
        answers, gt, _ = answers_by_pid[pid]
        conf, is_correct = _confidence_and_correctness(answers, gt)
        for i, (lo, hi) in enumerate(CONFIDENCE_BINS):
            if lo <= conf < hi:
                bins[i].append(is_correct)
                break

    rows = []
    for i, (lo, hi) in enumerate(CONFIDENCE_BINS):
        b = bins[i]
        n = len(b)
        frac = sum(b) / n if n > 0 else float("nan")
        rows.append(
            {"bin_label": CONFIDENCE_BIN_LABELS[i], "n": n, "fraction_correct": frac}
        )
    return rows


# ---------------------------------------------------------------------------
# Bootstrap CI helpers
# ---------------------------------------------------------------------------


def _bootstrap_backfire_ci(
    mv_curves: dict[str, dict],
    pids: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI on fraction of problems with mv_gain < 0."""
    n = len(pids)
    boot_rates = []
    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        sampled_pids = [pids[i] for i in idxs]
        n_back = sum(
            1 for p in sampled_pids if mv_curves.get(p, {}).get("mv_gain", 0) < 0
        )
        boot_rates.append(n_back / n)
    return float(np.percentile(boot_rates, 2.5)), float(np.percentile(boot_rates, 97.5))


def _bootstrap_ceiling_ci(
    gate_summary: dict,
    mv_curves: dict[str, dict],
    pids: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI on fraction-of-oracle-ceiling-captured (k=8 best gate acc)."""
    # Extract k=8 gate draws (we don't have per-problem gate draws here,
    # so we bootstrap over problem-level oracle and fixed-64 values)
    acc_64_per_pid = [
        mv_curves[pid]["mv_acc"]["64"] for pid in pids if pid in mv_curves
    ]
    oracle_per_pid = [
        max(mv_curves[pid]["mv_acc"]["1"], mv_curves[pid]["mv_acc"]["64"])
        for pid in pids
        if pid in mv_curves
    ]
    # Best gate acc (k=8) per problem: use the gate's aggregate metric
    # Since we don't store per-problem gate acc, use mean from gate_summary
    best_gate_acc = gate_summary["verdict"]["best_gate_acc_k8"]
    acc64_mean = gate_summary["verdict"]["fixed_budget_acc_64"]
    oracle_acc = gate_summary["verdict"]["oracle_acc"]
    ceiling = oracle_acc - acc64_mean

    n = len(acc_64_per_pid)
    if n == 0 or ceiling <= 0:
        return 0.0, 0.0

    # Bootstrap by resampling problems
    boot_fracs = []
    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        sampled_64 = [acc_64_per_pid[i] for i in idxs]
        sampled_oracle = [oracle_per_pid[i] for i in idxs]
        boot_acc64 = float(np.mean(sampled_64))
        boot_oracle = float(np.mean(sampled_oracle))
        boot_ceiling = boot_oracle - boot_acc64
        # Gate acc doesn't change with bootstrap (no per-problem gate data)
        # Use the aggregate best gate acc as approximate
        boot_frac = (
            (best_gate_acc - boot_acc64) / boot_ceiling if boot_ceiling > 0 else 0.0
        )
        boot_fracs.append(boot_frac)
    return float(np.percentile(boot_fracs, 2.5)), float(np.percentile(boot_fracs, 97.5))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _make_backfire_both(
    gains1: list[float],
    gains2: list[float],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, gains, label in zip(axes, [gains1, gains2], [MODEL1_LABEL, MODEL2_LABEL]):
        gains_arr = np.array(gains)
        n_neg = int(np.sum(gains_arr < 0))
        n_nonneg = len(gains_arr) - n_neg
        bins = np.linspace(
            min(-0.45, min(gains_arr) - 0.02), max(0.65, max(gains_arr) + 0.02), 20
        )
        neg = gains_arr[gains_arr < 0]
        nonneg = gains_arr[gains_arr >= 0]
        ax.hist(
            neg,
            bins=bins,
            color="#555555",
            edgecolor="white",
            linewidth=0.5,
            label=f"backfire n={n_neg}",
        )
        ax.hist(
            nonneg,
            bins=bins,
            color="#cccccc",
            edgecolor="white",
            linewidth=0.5,
            label=f"zero/gain n={n_nonneg}",
        )
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_xlabel("mv_gain = MV_acc(64) - MV_acc(1)")
        ax.set_ylabel("problems")
        ax.set_title(f"{label}")
        ax.legend(fontsize=9)
    fig.suptitle("Backfire distribution: both models", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _make_pareto_both(
    fb1: dict[int, float],
    fb2: dict[int, float],
    oracle1: tuple[float, float],
    oracle2: tuple[float, float],
    gate1: dict[int, list[tuple[float, float]]],
    gate2: dict[int, list[tuple[float, float]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    lss = {4: "--", 8: ":"}
    markers = {4: "s", 8: "^"}

    for ax, fb, oracle, gate, label in zip(
        axes,
        [fb1, fb2],
        [oracle1, oracle2],
        [gate1, gate2],
        [MODEL1_LABEL, MODEL2_LABEL],
    ):
        fb_x = sorted(fb.keys())
        fb_y = [fb[n] for n in fb_x]
        ax.plot(fb_x, fb_y, "k-o", markersize=5, linewidth=1.5, label="Fixed-budget SC")
        ax.plot(
            oracle[0], oracle[1], "k*", markersize=12, label="Oracle gate (upper bound)"
        )
        for k, pts in gate.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs,
                ys,
                linestyle=lss[k],
                marker=markers[k],
                color="black",
                markersize=5,
                linewidth=1.2,
                label=f"Agreement gate k={k}",
            )
        ax.set_xlabel("Mean compute (samples / problem)")
        ax.set_ylabel("Mean MV accuracy")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Accuracy vs compute: both models", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _make_calibration_both(
    cal1: list[dict],
    cal2: list[dict],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.array([0.375, 0.625, 0.875])  # midpoints of bins

    y1 = [r["fraction_correct"] for r in cal1]
    y2 = [r["fraction_correct"] for r in cal2]
    n1 = [r["n"] for r in cal1]
    n2 = [r["n"] for r in cal2]

    ax.plot(x, y1, "k-o", markersize=7, linewidth=1.5, label=MODEL1_LABEL)
    ax.plot(x, y2, "k--s", markersize=7, linewidth=1.5, label=MODEL2_LABEL)

    # Reference line: if confidence = accuracy (perfect calibration)
    ax.plot([0.25, 1.0], [0.25, 1.0], color="gray", linewidth=0.8, linestyle=":")

    for xi, yi, ni, model in zip(x, y1, n1, [MODEL1_LABEL] * 3):
        ax.annotate(
            f"n={ni}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="black",
        )
    for xi, yi, ni in zip(x, y2, n2):
        ax.annotate(
            f"n={ni}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
            color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIDENCE_BIN_LABELS, fontsize=9)
    ax.set_xlabel("Confidence bin (plurality fraction)")
    ax.set_ylabel("Fraction plurality answer is correct")
    ax.set_title("Confidence vs. accuracy by agreement bin")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run cross-model comparison and save figures + summary JSON."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    main_ids = _main_pilot_ids()

    # ---- Load data ----
    logger.info("Loading model-1 data...")
    gate1 = json.loads((_GATE1_DIR / "gate_summary.json").read_text())
    mv_curves1 = _load_mv_curves(_MV1_DIR, main_ids)
    answers1 = _load_answers_and_gt(_SAMPLES1_DIR, main_ids)
    pids1 = sorted(set(mv_curves1.keys()) & set(answers1.keys()))

    logger.info("Loading model-2 data...")
    gate2_path = _GATE2_DIR / "gate_summary.json"
    if not gate2_path.exists():
        logger.error(
            "gate_summary.json for model 2 not found. Run run_model2_gate.py first."
        )
        sys.exit(1)
    gate2 = json.loads(gate2_path.read_text())
    mv_curves2 = _load_mv_curves(_MV2_DIR, main_ids)
    answers2 = _load_answers_and_gt(_SAMPLES2_DIR, main_ids)
    pids2 = sorted(set(mv_curves2.keys()) & set(answers2.keys()))

    logger.info("Model-1: %d problems  Model-2: %d problems", len(pids1), len(pids2))

    # ---- Calibration ----
    logger.info("Computing calibration tables...")
    cal1 = _calibration_table(answers1, pids1)
    cal2 = _calibration_table(answers2, pids2)

    logger.info("Model-1 calibration:")
    for row in cal1:
        logger.info(
            "  %s  n=%d  frac_correct=%.4f",
            row["bin_label"],
            row["n"],
            row["fraction_correct"],
        )
    logger.info("Model-2 calibration:")
    for row in cal2:
        logger.info(
            "  %s  n=%d  frac_correct=%.4f",
            row["bin_label"],
            row["n"],
            row["fraction_correct"],
        )

    # ---- Bootstrap CIs ----
    logger.info("Bootstrap CIs (seed=%d, n=%d)...", _SEED_BOOT, _N_BOOTSTRAP)
    rng = np.random.default_rng(_SEED_BOOT)

    backfire_ci1 = _bootstrap_backfire_ci(mv_curves1, pids1, _N_BOOTSTRAP, rng)
    backfire_ci2 = _bootstrap_backfire_ci(mv_curves2, pids2, _N_BOOTSTRAP, rng)

    ceiling_ci1 = _bootstrap_ceiling_ci(gate1, mv_curves1, pids1, _N_BOOTSTRAP, rng)
    ceiling_ci2 = _bootstrap_ceiling_ci(gate2, mv_curves2, pids2, _N_BOOTSTRAP, rng)

    logger.info("Backfire CI model-1: [%.4f, %.4f]", *backfire_ci1)
    logger.info("Backfire CI model-2: [%.4f, %.4f]", *backfire_ci2)
    logger.info("Ceiling CI model-1: [%.4f, %.4f]", *ceiling_ci1)
    logger.info("Ceiling CI model-2: [%.4f, %.4f]", *ceiling_ci2)

    # ---- Build gate curves for figures ----
    def _build_gate_curves(gate_summary: dict) -> dict[int, list[tuple[float, float]]]:
        curves = {}
        for k in _K_VALUES:
            pts = []
            k_key = f"k{k}"
            if k_key in gate_summary.get("realistic_gate", {}):
                for tau in _TAU_GRID:
                    tau_key = f"tau_{str(tau).replace('.', '_')}"
                    m = gate_summary["realistic_gate"][k_key].get(tau_key, {})
                    if m:
                        pts.append((m["compute"], m["acc"]))
            pts.sort()
            curves[k] = pts
        return curves

    gate_curves1 = _build_gate_curves(gate1)
    gate_curves2 = _build_gate_curves(gate2)

    fb1 = {n: gate1["fixed_budget"][str(n)]["acc"] for n in _N_VALUES}
    fb2 = {n: gate2["fixed_budget"][str(n)]["acc"] for n in _N_VALUES}

    oracle1 = (gate1["oracle_gate"]["mean_compute"], gate1["oracle_gate"]["acc"])
    oracle2 = (gate2["oracle_gate"]["mean_compute"], gate2["oracle_gate"]["acc"])

    # ---- Figures ----
    gains1 = [mv_curves1[pid]["mv_gain"] for pid in pids1]
    gains2 = [mv_curves2[pid]["mv_gain"] for pid in pids2]

    _make_backfire_both(gains1, gains2, _GATE2_DIR / "backfire_both.png")
    _make_pareto_both(
        fb1,
        fb2,
        oracle1,
        oracle2,
        gate_curves1,
        gate_curves2,
        _GATE2_DIR / "pareto_both.png",
    )
    _make_calibration_both(cal1, cal2, _GATE2_DIR / "calibration_both.png")

    # ---- Cross-model comparison table ----
    def _get_gate_at(gate_summary: dict, k: int, tau: float) -> dict:
        tau_key = f"tau_{str(tau).replace('.', '_')}"
        return gate_summary["realistic_gate"][f"k{k}"].get(tau_key, {})

    op_k8_tau075_1 = _get_gate_at(gate1, 8, 0.75)
    op_k8_tau075_2 = _get_gate_at(gate2, 8, 0.75)

    frac_ceil1 = gate1["verdict"]["fraction_oracle_ceiling_captured_k8"]
    frac_ceil2 = gate2["verdict"]["fraction_oracle_ceiling_captured_k8"]

    # ---- Verdict ----
    backfire_rate1 = gate1["backfire"]["fraction_backfire"]
    backfire_rate2 = gate2["backfire"]["fraction_backfire"]
    both_backfire = backfire_rate1 >= 0.3 and backfire_rate2 >= 0.3

    # Check if highest-confidence bin still has substantial wrong-plurality problems
    high_conf_correct1 = cal1[-1]["fraction_correct"]
    high_conf_correct2 = cal2[-1]["fraction_correct"]
    mechanism_confirmed = high_conf_correct1 < 0.85 and high_conf_correct2 < 0.85

    verdict = (
        "BACKFIRE REPLICATES AND MECHANISM CONFIRMED: "
        if both_backfire and mechanism_confirmed
        else "PARTIAL REPLICATION: "
    )
    verdict += (
        f"Both models show substantial backfire rates "
        f"({backfire_rate1:.0%} Qwen, {backfire_rate2:.0%} Llama). "
        f"Even in the highest-confidence bin, plurality is wrong "
        f"{(1-high_conf_correct1):.0%} (Qwen) and {(1-high_conf_correct2):.0%} (Llama) of the time, "
        f"confirming that high self-consistency does not imply correctness. "
        f"Realistic gate captures {(frac_ceil1 or 0):.1%} (Qwen) and "
        f"{(frac_ceil2 or 0):.1%} (Llama) of the oracle ceiling."
        if both_backfire
        else "See numbers in table."
    )

    summary: dict[str, Any] = {
        "model_1": {
            "id": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "label": MODEL1_LABEL,
            "n_problems": len(pids1),
            "n1_acc": round(fb1[1], 4),
            "n64_acc": round(fb1[64], 4),
            "backfire_rate": round(backfire_rate1, 4),
            "backfire_ci_95": [round(backfire_ci1[0], 4), round(backfire_ci1[1], 4)],
            "oracle_acc": round(gate1["oracle_gate"]["acc"], 4),
            "oracle_compute": round(gate1["oracle_gate"]["mean_compute"], 2),
            "gate_k8_tau075": {
                "acc": round(op_k8_tau075_1.get("acc", float("nan")), 4),
                "compute": round(op_k8_tau075_1.get("compute", float("nan")), 2),
            },
            "fraction_oracle_ceiling_captured": frac_ceil1,
            "ceiling_ci_95": [round(ceiling_ci1[0], 4), round(ceiling_ci1[1], 4)],
            "calibration": cal1,
        },
        "model_2": {
            "id": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            "label": MODEL2_LABEL,
            "n_problems": len(pids2),
            "parse_rate": gate2.get("parse_rate"),
            "n1_acc": round(fb2[1], 4),
            "n64_acc": round(fb2[64], 4),
            "backfire_rate": round(backfire_rate2, 4),
            "backfire_ci_95": [round(backfire_ci2[0], 4), round(backfire_ci2[1], 4)],
            "oracle_acc": round(gate2["oracle_gate"]["acc"], 4),
            "oracle_compute": round(gate2["oracle_gate"]["mean_compute"], 2),
            "gate_k8_tau075": {
                "acc": round(op_k8_tau075_2.get("acc", float("nan")), 4),
                "compute": round(op_k8_tau075_2.get("compute", float("nan")), 2),
            },
            "fraction_oracle_ceiling_captured": frac_ceil2,
            "ceiling_ci_95": [round(ceiling_ci2[0], 4), round(ceiling_ci2[1], 4)],
            "calibration": cal2,
        },
        "cross_model_verdict": verdict,
        "figures": [
            "outputs/gate_model2/backfire_both.png",
            "outputs/gate_model2/pareto_both.png",
            "outputs/gate_model2/calibration_both.png",
        ],
    }

    out_path = _GATE2_DIR / "cross_model_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s", out_path)

    sep = "=" * 72
    print(f"\n{sep}")
    print("Cross-Model Comparison Summary")
    print(sep)
    print(f"\n{'Metric':<42} {MODEL1_LABEL:>12} {MODEL2_LABEL:>12}")
    print(f"  {'-'*40} {'-'*12} {'-'*12}")
    rows = [
        ("N=1 accuracy", f"{fb1[1]:.4f}", f"{fb2[1]:.4f}"),
        ("N=64 MV accuracy", f"{fb1[64]:.4f}", f"{fb2[64]:.4f}"),
        (
            "Backfire rate (mv_gain < 0)",
            f"{backfire_rate1:.4f}",
            f"{backfire_rate2:.4f}",
        ),
        (
            "  95% CI",
            f"[{backfire_ci1[0]:.3f},{backfire_ci1[1]:.3f}]",
            f"[{backfire_ci2[0]:.3f},{backfire_ci2[1]:.3f}]",
        ),
        (
            "Oracle gate accuracy",
            f"{gate1['oracle_gate']['acc']:.4f}",
            f"{gate2['oracle_gate']['acc']:.4f}",
        ),
        (
            "Oracle gate mean compute",
            f"{gate1['oracle_gate']['mean_compute']:.2f}",
            f"{gate2['oracle_gate']['mean_compute']:.2f}",
        ),
        (
            "Gate k=8 tau=0.75 accuracy",
            f"{op_k8_tau075_1.get('acc', float('nan')):.4f}",
            f"{op_k8_tau075_2.get('acc', float('nan')):.4f}",
        ),
        (
            "Gate k=8 tau=0.75 compute",
            f"{op_k8_tau075_1.get('compute', float('nan')):.2f}",
            f"{op_k8_tau075_2.get('compute', float('nan')):.2f}",
        ),
        (
            "Fraction oracle ceiling captured",
            f"{frac_ceil1:.4f}" if frac_ceil1 is not None else "n/a",
            f"{frac_ceil2:.4f}" if frac_ceil2 is not None else "n/a",
        ),
        (
            "  95% CI",
            f"[{ceiling_ci1[0]:.3f},{ceiling_ci1[1]:.3f}]",
            f"[{ceiling_ci2[0]:.3f},{ceiling_ci2[1]:.3f}]",
        ),
    ]
    for name, v1, v2 in rows:
        print(f"  {name:<42} {v1:>12} {v2:>12}")

    print("\nCalibration table (confidence bin -> fraction plurality correct):")
    print(
        f"\n  {'Bin':<15} {'n (M1)':>8} {'frac_correct (M1)':>18} "
        f"{'n (M2)':>8} {'frac_correct (M2)':>18}"
    )
    for r1, r2 in zip(cal1, cal2):
        print(
            f"  {r1['bin_label']:<15} {r1['n']:>8} {r1['fraction_correct']:>18.4f}"
            f" {r2['n']:>8} {r2['fraction_correct']:>18.4f}"
        )

    print(f"\nVerdict: {verdict}")
    print(sep)


if __name__ == "__main__":
    main()
