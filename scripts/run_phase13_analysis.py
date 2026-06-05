"""Phase 13 -- Full 198 analysis: gates, PH1-PH4, entropy gate, figures.

Reads:
  outputs/samples/              (Qwen, all 198 problems)
  outputs/samples_model2/       (Llama, all 198 problems)

Writes:
  outputs/gate/gate_summary.json                    (overwrite, Qwen 198)
  outputs/gate_model2/gate_summary.json             (overwrite, Llama 198)
  outputs/gate_model2/cross_model_summary.json      (overwrite, with splits)
  outputs/gate_model2/confirmatory_results.json     (PH1-PH4 verdict)
  outputs/gate_model2/backfire_both.png
  outputs/gate_model2/pareto_both.png
  outputs/gate_model2/calibration_both.png
  outputs/gate_model2/gate_sweep_both.png

Usage:
    source .venv/bin/activate
    python scripts/run_phase13_analysis.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_OUTPUTS_DIR = ROOT / "outputs"
_SAMPLES1_DIR = _OUTPUTS_DIR / "samples"
_SAMPLES2_DIR = _OUTPUTS_DIR / "samples_model2"
_GATE1_DIR = _OUTPUTS_DIR / "gate"
_GATE2_DIR = _OUTPUTS_DIR / "gate_model2"

_T_MAIN: float = 0.7
_N_VALUES: list[int] = [1, 2, 4, 8, 16, 32, 64]
_K_VALUES: list[int] = [4, 8]
_TAU_GRID: list[float] = [round(t, 2) for t in np.arange(0.5, 1.01, 0.05)]
_N_MC: int = 2000
_SEED_K4: int = 42
_SEED_K8: int = 43
_N_BOOTSTRAP: int = 1000
_SEED_BOOT: int = 42

# Entropy gate
_ENTROPY_K: int = 4  # probe samples for entropy signal
_N_ENTROPY_THRESHOLDS: int = 40
_ENTROPY_SEED: int = 44

CONFIDENCE_BINS = [(0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
CONFIDENCE_BIN_LABELS = ["[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.00]"]

MODEL1_LABEL = "Qwen2.5-7B"
MODEL2_LABEL = "Llama-3-8B"

# Pre-registered confirmatory hypothesis thresholds
_PH1_THRESHOLD = 0.33   # backfire rate >= this PASSES
_PH2_THRESHOLD = 0.10   # ceiling captured <= this PASSES (agreement gate)
_PH3_THRESHOLD = 0.70   # top-bin accuracy <= this PASSES
_PH4_THRESHOLD = 0.10   # ceiling captured <= this PASSES (entropy gate)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Problem set
# ---------------------------------------------------------------------------


def _load_problem_splits() -> tuple[list[str], list[str], list[str]]:
    """Return (exploratory_ids, confirmatory_ids, all_ids) sorted."""
    locked = set(json.loads((ROOT / "data" / "problem_ids.json").read_text()))
    gate_set = set(
        json.loads((_OUTPUTS_DIR / "gate_minus_1_labels.json").read_text())["gate_problems"]
    )
    exploratory = sorted(locked - gate_set)  # 47 main-pilot problems

    # Load full 198 from HuggingFace to get all IDs
    from pilot.data_loader import load_gpqa_diamond
    all_problems = load_gpqa_diamond()
    all_ids = sorted(p.id for p in all_problems)

    confirmatory = sorted(pid for pid in all_ids if pid not in set(exploratory))
    return exploratory, confirmatory, all_ids


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_samples(
    samples_dir: Path, pids: list[str]
) -> dict[str, dict[str, Any]]:
    """Load T=0.7 samples for all pids. Returns {pid: {answers, gt, n_total, entropies}}."""
    out: dict[str, dict[str, Any]] = {}
    for pid in pids:
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
        answers: list[str | None] = [
            r.get("extracted_answer") if r.get("extracted_answer") not in ("UNPARSEABLE", None, "")
            else None
            for r in records
        ]
        entropies: list[float | None] = [
            r.get("mean_token_entropy") for r in records
        ]
        out[pid] = {
            "answers": answers,
            "gt": gt,
            "n_total": len(answers),
            "entropies": entropies,  # per-sample mean per-token entropy; None if no logprobs
        }
    return out


# ---------------------------------------------------------------------------
# MV accuracy computation
# ---------------------------------------------------------------------------


def _plurality(answers: list[str | None]) -> str | None:
    """Lexicographic tie-breaking, ground-truth agnostic."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    max_c = max(counts.values())
    tied = sorted(a for a, c in counts.items() if c == max_c)
    return tied[0]


def _mv_acc_at_n(
    answers: list[str | None],
    gt: str,
    n: int,
    rng: np.random.Generator,
    max_exact: int = 5000,
) -> float:
    """Mean MV accuracy at subset size n using exact or MC enumeration."""
    from math import comb

    valid = [a for a in answers if a is not None]
    n_total = len(valid)
    if n_total < n:
        n = n_total
    if n == 0:
        return 0.0

    if comb(n_total, n) <= max_exact:
        from itertools import combinations
        hits = sum(
            1 for subset in combinations(valid, n)
            if _plurality(list(subset)) == gt
        )
        return hits / comb(n_total, n)
    else:
        hits = 0
        for _ in range(_N_MC):
            idxs = rng.choice(n_total, size=n, replace=False)
            subset = [valid[i] for i in idxs]
            if _plurality(subset) == gt:
                hits += 1
        return hits / _N_MC


def _compute_mv_curve(
    data: dict[str, Any], rng: np.random.Generator
) -> dict[str, float]:
    """Return {str(n): mv_acc} for all N_VALUES."""
    answers = data["answers"]
    gt = data["gt"]
    return {str(n): _mv_acc_at_n(answers, gt, n, rng) for n in _N_VALUES}


# ---------------------------------------------------------------------------
# Agreement gate simulation
# ---------------------------------------------------------------------------


def _simulate_agreement_gate(
    answers: list[str | None],
    gt: str,
    k: int,
    tau: float,
    mv_acc_64: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Monte Carlo agreement gate. Returns (accuracy, mean_compute)."""
    valid = [a for a in answers if a is not None]
    n_total = len(valid)
    full_n = min(64, n_total)

    gate_correct = 0
    gate_compute = 0.0
    n_draws = _N_MC

    for _ in range(n_draws):
        # Probe: draw k samples
        probe_idxs = rng.choice(n_total, size=min(k, n_total), replace=False)
        probe = [valid[i] for i in probe_idxs]
        c = Counter(probe)
        max_c = max(c.values()) if c else 0
        agreement = max_c / k

        if agreement >= tau:
            # Stop early; return probe plurality
            ans = _plurality(probe)
            gate_correct += int(ans == gt)
            gate_compute += k
        else:
            # Run full N=64
            full_idxs = rng.choice(n_total, size=full_n, replace=False)
            full_ans = _plurality([valid[i] for i in full_idxs])
            gate_correct += int(full_ans == gt)
            gate_compute += 64

    acc = gate_correct / n_draws
    mean_compute = gate_compute / n_draws
    return acc, mean_compute


# ---------------------------------------------------------------------------
# Entropy gate simulation
# ---------------------------------------------------------------------------


def _simulate_entropy_gate(
    data: dict[str, Any],
    k: int,
    threshold: float,
    mv_acc_64: float,
    rng: np.random.Generator,
) -> tuple[float, float] | None:
    """Entropy gate using mean per-token entropy of k probe samples.

    Returns (accuracy, mean_compute) or None if logprobs unavailable.
    """
    answers = data["answers"]
    gt = data["gt"]
    entropies = data["entropies"]

    # Check logprob availability
    valid_pairs = [
        (a, e)
        for a, e in zip(answers, entropies)
        if a is not None and e is not None
    ]
    if not valid_pairs:
        return None  # No logprobs available for this problem

    valid_answers = [p[0] for p in valid_pairs]
    valid_entropies = [p[1] for p in valid_pairs]
    n_total = len(valid_answers)
    full_n = min(64, n_total)

    gate_correct = 0
    gate_compute = 0.0
    n_draws = _N_MC

    for _ in range(n_draws):
        probe_k = min(k, n_total)
        probe_idxs = rng.choice(n_total, size=probe_k, replace=False)
        probe_ans = [valid_answers[i] for i in probe_idxs]
        probe_ent = [valid_entropies[i] for i in probe_idxs]
        mean_ent = sum(probe_ent) / probe_k

        if mean_ent < threshold:
            # Low entropy: confident; return probe plurality (cost = k)
            ans = _plurality(probe_ans)
            gate_correct += int(ans == gt)
            gate_compute += k
        else:
            # High entropy: uncertain; run full N=64
            full_idxs = rng.choice(n_total, size=full_n, replace=False)
            full_ans = _plurality([valid_answers[i] for i in full_idxs])
            gate_correct += int(full_ans == gt)
            gate_compute += 64

    return gate_correct / n_draws, gate_compute / n_draws


# ---------------------------------------------------------------------------
# Gate metrics over a problem set
# ---------------------------------------------------------------------------


def _gate_metrics(
    samples: dict[str, dict],
    pids: list[str],
    mv_curves: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compute full gate analysis for a set of problems."""
    valid_pids = [p for p in pids if p in samples and p in mv_curves]

    # Fixed-budget accuracy
    fixed_budget: dict[str, dict] = {}
    for n in _N_VALUES:
        accs = [mv_curves[p][str(n)] for p in valid_pids]
        fixed_budget[str(n)] = {"acc": float(np.mean(accs)), "compute": n}

    # Backfire
    gains = [mv_curves[p]["64"] - mv_curves[p]["1"] for p in valid_pids]
    n_backfire = sum(1 for g in gains if g < 0)
    backfire = {
        "n_backfire": n_backfire,
        "n_positive_gain": sum(1 for g in gains if g > 0),
        "n_zero_gain": sum(1 for g in gains if g == 0),
        "fraction_backfire": n_backfire / len(valid_pids) if valid_pids else 0.0,
        "mean_gain": float(np.mean(gains)),
        "median_gain": float(np.median(gains)),
        "min_gain": float(np.min(gains)),
        "max_gain": float(np.max(gains)),
    }

    # Oracle gate
    oracle_accs = [max(mv_curves[p]["1"], mv_curves[p]["64"]) for p in valid_pids]
    n_go_full = sum(1 for p in valid_pids if mv_curves[p]["64"] >= mv_curves[p]["1"])
    oracle_acc = float(np.mean(oracle_accs))
    fixed_64_acc = fixed_budget["64"]["acc"]
    oracle_compute = float(
        np.mean([1.0 if mv_curves[p]["1"] >= mv_curves[p]["64"] else 64.0 for p in valid_pids])
    )
    oracle_gate = {
        "acc": oracle_acc,
        "mean_compute": oracle_compute,
        "fraction_go_to_64": n_go_full / len(valid_pids),
        "gain_over_fixed_64": oracle_acc - fixed_64_acc,
    }

    # Agreement gate sweep
    rng_k4 = np.random.default_rng(_SEED_K4)
    rng_k8 = np.random.default_rng(_SEED_K8)

    realistic_gate: dict[str, dict] = {"k4": {}, "k8": {}}
    for tau in _TAU_GRID:
        tau_key = f"tau_{str(tau).replace('.', '_')}"
        # k=4
        results_k4 = [
            _simulate_agreement_gate(
                samples[p]["answers"], samples[p]["gt"], 4, tau,
                mv_curves[p]["64"], rng_k4,
            )
            for p in valid_pids
        ]
        acc_k4 = float(np.mean([r[0] for r in results_k4]))
        comp_k4 = float(np.mean([r[1] for r in results_k4]))
        stop_k4 = float(np.mean([1.0 if r[1] <= 4 + 1e-6 else 0.0 for r in results_k4]))
        realistic_gate["k4"][tau_key] = {"acc": acc_k4, "compute": comp_k4, "stop_rate": stop_k4}
        # k=8
        results_k8 = [
            _simulate_agreement_gate(
                samples[p]["answers"], samples[p]["gt"], 8, tau,
                mv_curves[p]["64"], rng_k8,
            )
            for p in valid_pids
        ]
        acc_k8 = float(np.mean([r[0] for r in results_k8]))
        comp_k8 = float(np.mean([r[1] for r in results_k8]))
        stop_k8 = float(np.mean([1.0 if r[1] <= 8 + 1e-6 else 0.0 for r in results_k8]))
        realistic_gate["k8"][tau_key] = {"acc": acc_k8, "compute": comp_k8, "stop_rate": stop_k8}

    # Entropy gate sweep
    rng_ent = np.random.default_rng(_ENTROPY_SEED)
    entropy_gate: dict[str, Any] = {"available": False}

    ent_pids = [p for p in valid_pids if any(e is not None for e in samples[p]["entropies"])]
    if ent_pids:
        # Determine threshold range from data
        all_ents = [
            e
            for p in ent_pids
            for e in samples[p]["entropies"]
            if e is not None
        ]
        t_min = float(np.percentile(all_ents, 5))
        t_max = float(np.percentile(all_ents, 95))
        thresholds = np.linspace(t_min, t_max, _N_ENTROPY_THRESHOLDS).tolist()

        ent_sweep: list[dict] = []
        for thresh in thresholds:
            per_pid = [
                _simulate_entropy_gate(samples[p], _ENTROPY_K, thresh, mv_curves[p]["64"], rng_ent)
                for p in ent_pids
            ]
            valid_results = [r for r in per_pid if r is not None]
            if not valid_results:
                continue
            acc = float(np.mean([r[0] for r in valid_results]))
            comp = float(np.mean([r[1] for r in valid_results]))
            ceiling = oracle_acc - fixed_64_acc
            cap = float((acc - fixed_64_acc) / ceiling) if ceiling > 1e-9 else 0.0
            ent_sweep.append({"threshold": thresh, "acc": acc, "compute": comp, "ceiling_captured": cap})

        best = max(ent_sweep, key=lambda x: x["ceiling_captured"]) if ent_sweep else {}
        entropy_gate = {
            "available": True,
            "n_problems_with_logprobs": len(ent_pids),
            "k": _ENTROPY_K,
            "sweep": ent_sweep,
            "best": best,
        }
    else:
        entropy_gate = {"available": False, "n_problems_with_logprobs": 0}

    # Pareto dominance
    fb_points = [(v["compute"], v["acc"]) for v in fixed_budget.values()]

    def _pareto_dominates(cx: float, cy: float) -> bool:
        return not any(bx <= cx and by >= cy for bx, by in fb_points)

    k8_tau075_key = "tau_0_75"
    k8_tau075 = realistic_gate["k8"].get(k8_tau075_key, {})
    ceiling = oracle_acc - fixed_64_acc
    frac_ceil_k8 = float(
        (k8_tau075.get("acc", fixed_64_acc) - fixed_64_acc) / ceiling
    ) if ceiling > 1e-9 else 0.0
    best_k8_acc = max(
        v["acc"] for v in realistic_gate["k8"].values()
    )

    verdict = {
        "realistic_gate_pareto_dominates_fixed_budget": any(
            _pareto_dominates(v["compute"], v["acc"])
            for d in realistic_gate.values()
            for v in d.values()
        ),
        "best_gate_acc_k8": best_k8_acc,
        "fixed_budget_acc_64": fixed_64_acc,
        "oracle_acc": oracle_acc,
        "oracle_ceiling_gain": ceiling,
        "fraction_oracle_ceiling_captured_k8": frac_ceil_k8,
    }

    # Calibration
    calibration = _calibration_table(samples, valid_pids)

    return {
        "n_problems": len(valid_pids),
        "fixed_budget": fixed_budget,
        "backfire": backfire,
        "oracle_gate": oracle_gate,
        "realistic_gate": realistic_gate,
        "entropy_gate": entropy_gate,
        "calibration": calibration,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _calibration_table(samples: dict, pids: list[str]) -> list[dict]:
    bins: list[list[bool]] = [[] for _ in CONFIDENCE_BINS]
    for pid in pids:
        if pid not in samples:
            continue
        answers = samples[pid]["answers"]
        gt = samples[pid]["gt"]
        valid = [a for a in answers if a is not None]
        if not valid:
            continue
        counts = Counter(valid)
        max_c = max(counts.values())
        conf = max_c / len(valid)
        tied = sorted(a for a, c in counts.items() if c == max_c)
        plurality = tied[0]
        is_correct = plurality == gt
        for i, (lo, hi) in enumerate(CONFIDENCE_BINS):
            if lo <= conf < hi:
                bins[i].append(is_correct)
                break
    rows = []
    for i, _bin in enumerate(CONFIDENCE_BINS):
        b = bins[i]
        n = len(b)
        rows.append({
            "bin_label": CONFIDENCE_BIN_LABELS[i],
            "n": n,
            "fraction_correct": float(sum(b) / n) if n > 0 else float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def _bootstrap_backfire_ci(
    mv_curves: dict[str, dict[str, float]],
    pids: list[str],
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(pids)
    rates = []
    for _ in range(_N_BOOTSTRAP):
        idxs = rng.integers(0, n, size=n)
        sp = [pids[i] for i in idxs]
        rate = sum(1 for p in sp if mv_curves.get(p, {}).get("64", 0) - mv_curves.get(p, {}).get("1", 0) < 0) / n
        rates.append(rate)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def _bootstrap_ceiling_ci(
    gate_acc: float,
    mv_curves: dict[str, dict[str, float]],
    pids: list[str],
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(pids)
    fracs = []
    for _ in range(_N_BOOTSTRAP):
        idxs = rng.integers(0, n, size=n)
        sp = [pids[i] for i in idxs]
        acc64 = float(np.mean([mv_curves[p]["64"] for p in sp if p in mv_curves]))
        oracle = float(np.mean([max(mv_curves[p]["1"], mv_curves[p]["64"]) for p in sp if p in mv_curves]))
        ceiling = oracle - acc64
        fracs.append(float((gate_acc - acc64) / ceiling) if ceiling > 1e-9 else 0.0)
    return float(np.percentile(fracs, 2.5)), float(np.percentile(fracs, 97.5))


# ---------------------------------------------------------------------------
# PH1-PH4 evaluation
# ---------------------------------------------------------------------------


def _evaluate_ph(
    confirmatory_metrics1: dict,
    confirmatory_metrics2: dict,
    conf_mv1: dict[str, dict[str, float]],
    conf_mv2: dict[str, dict[str, float]],
    conf_pids: list[str],
) -> dict[str, Any]:
    """Evaluate PH1-PH4 on the confirmatory set."""
    rng = np.random.default_rng(_SEED_BOOT)

    # PH1: backfire rate >= 0.33 for BOTH models
    rate1 = confirmatory_metrics1["backfire"]["fraction_backfire"]
    rate2 = confirmatory_metrics2["backfire"]["fraction_backfire"]
    ci1 = _bootstrap_backfire_ci(conf_mv1, conf_pids, rng)
    rng2 = np.random.default_rng(_SEED_BOOT + 1)
    ci2 = _bootstrap_backfire_ci(conf_mv2, conf_pids, rng2)
    ph1 = {
        "hypothesis": "Backfire rate >= 0.33 for BOTH models",
        "threshold": _PH1_THRESHOLD,
        "model_1": {"value": round(rate1, 4), "ci_95": [round(ci1[0], 4), round(ci1[1], 4)]},
        "model_2": {"value": round(rate2, 4), "ci_95": [round(ci2[0], 4), round(ci2[1], 4)]},
        "pass": rate1 >= _PH1_THRESHOLD and rate2 >= _PH1_THRESHOLD,
    }

    # PH2: agreement gate ceiling captured <= 0.10 for BOTH models (k=8, tau=0.75)
    k8_075_1 = confirmatory_metrics1["realistic_gate"]["k8"].get("tau_0_75", {})
    k8_075_2 = confirmatory_metrics2["realistic_gate"]["k8"].get("tau_0_75", {})
    fixed64_1 = confirmatory_metrics1["fixed_budget"]["64"]["acc"]
    fixed64_2 = confirmatory_metrics2["fixed_budget"]["64"]["acc"]
    oracle1 = confirmatory_metrics1["oracle_gate"]["acc"]
    oracle2 = confirmatory_metrics2["oracle_gate"]["acc"]
    ceil1 = oracle1 - fixed64_1
    ceil2 = oracle2 - fixed64_2
    gate_acc1 = k8_075_1.get("acc", fixed64_1)
    gate_acc2 = k8_075_2.get("acc", fixed64_2)
    cap1 = float((gate_acc1 - fixed64_1) / ceil1) if ceil1 > 1e-9 else 0.0
    cap2 = float((gate_acc2 - fixed64_2) / ceil2) if ceil2 > 1e-9 else 0.0
    rng3 = np.random.default_rng(_SEED_BOOT + 2)
    rng4 = np.random.default_rng(_SEED_BOOT + 3)
    ci_cap1 = _bootstrap_ceiling_ci(gate_acc1, conf_mv1, conf_pids, rng3)
    ci_cap2 = _bootstrap_ceiling_ci(gate_acc2, conf_mv2, conf_pids, rng4)
    ph2 = {
        "hypothesis": "Agreement gate (k=8, tau=0.75) captures <= 10% of oracle ceiling for BOTH",
        "threshold": _PH2_THRESHOLD,
        "model_1": {"value": round(cap1, 4), "ci_95": [round(ci_cap1[0], 4), round(ci_cap1[1], 4)]},
        "model_2": {"value": round(cap2, 4), "ci_95": [round(ci_cap2[0], 4), round(ci_cap2[1], 4)]},
        "pass": cap1 <= _PH2_THRESHOLD and cap2 <= _PH2_THRESHOLD,
    }

    # PH3: top-bin accuracy <= 0.70 for BOTH models
    top1 = confirmatory_metrics1["calibration"][-1]["fraction_correct"]
    top2 = confirmatory_metrics2["calibration"][-1]["fraction_correct"]
    ph3 = {
        "hypothesis": "Top-confidence bin accuracy <= 0.70 for BOTH models",
        "threshold": _PH3_THRESHOLD,
        "model_1": {"value": round(top1, 4)},
        "model_2": {"value": round(top2, 4)},
        "pass": top1 <= _PH3_THRESHOLD and top2 <= _PH3_THRESHOLD,
    }

    # PH4: entropy gate ceiling captured <= 0.10 for BOTH (contingent on logprobs)
    ent1 = confirmatory_metrics1.get("entropy_gate", {})
    ent2 = confirmatory_metrics2.get("entropy_gate", {})
    ent1_avail = ent1.get("available", False)
    ent2_avail = ent2.get("available", False)
    ent_best_cap1 = ent1.get("best", {}).get("ceiling_captured", float("nan")) if ent1_avail else float("nan")
    ent_best_cap2 = ent2.get("best", {}).get("ceiling_captured", float("nan")) if ent2_avail else float("nan")

    import math
    ph4_pass: bool | None = None
    if ent1_avail and ent2_avail:
        ph4_pass = (not math.isnan(ent_best_cap1) and ent_best_cap1 <= _PH4_THRESHOLD and
                    not math.isnan(ent_best_cap2) and ent_best_cap2 <= _PH4_THRESHOLD)
    ph4 = {
        "hypothesis": "Entropy gate captures <= 10% of oracle ceiling for BOTH models",
        "threshold": _PH4_THRESHOLD,
        "contingent_on_logprobs": True,
        "model_1": {
            "logprobs_available": ent1_avail,
            "best_ceiling_captured": round(ent_best_cap1, 4) if not math.isnan(ent_best_cap1) else None,
        },
        "model_2": {
            "logprobs_available": ent2_avail,
            "best_ceiling_captured": round(ent_best_cap2, 4) if not math.isnan(ent_best_cap2) else None,
        },
        "pass": ph4_pass,
        "note": "Not evaluated" if (not ent1_avail or not ent2_avail) else "",
    }

    results = {"PH1": ph1, "PH2": ph2, "PH3": ph3, "PH4": ph4}
    logger.info("PH1 %s (model1=%.3f, model2=%.3f, threshold=%.2f)",
                "PASS" if ph1["pass"] else "FAIL", rate1, rate2, _PH1_THRESHOLD)
    logger.info("PH2 %s (model1=%.3f, model2=%.3f, threshold=%.2f)",
                "PASS" if ph2["pass"] else "FAIL", cap1, cap2, _PH2_THRESHOLD)
    logger.info("PH3 %s (model1=%.3f, model2=%.3f, threshold=%.2f)",
                "PASS" if ph3["pass"] else "FAIL", top1, top2, _PH3_THRESHOLD)
    logger.info("PH4 %s",
                "PASS" if ph4["pass"] else ("FAIL" if ph4["pass"] is False else "NOT EVALUATED"))
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _make_backfire_both(gains1: list[float], gains2: list[float], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, gains, label in zip(axes, [gains1, gains2], [MODEL1_LABEL, MODEL2_LABEL]):
        arr = np.array(gains)
        n_neg = int(np.sum(arr < 0))
        n_nonneg = len(arr) - n_neg
        bins = np.linspace(min(-0.45, min(arr) - 0.02), max(0.65, max(arr) + 0.02), 20)
        ax.hist(arr[arr < 0], bins=bins, color="#555555", edgecolor="white",
                linewidth=0.5, label=f"backfire n={n_neg}")
        ax.hist(arr[arr >= 0], bins=bins, color="#cccccc", edgecolor="white",
                linewidth=0.5, label=f"zero/gain n={n_nonneg}")
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_xlabel("mv_gain = MV_acc(64) - MV_acc(1)")
        ax.set_ylabel("problems")
        ax.set_title(label)
        ax.legend(fontsize=9)
    fig.suptitle("Backfire distribution: both models (n=198)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _make_pareto_both(
    metrics1: dict, metrics2: dict,
    ent_conf1: dict | None, ent_conf2: dict | None,
    out_path: Path,
) -> None:
    """Pareto figure with agreement gate + entropy gate (confirmatory set)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    lss = {4: "--", 8: ":"}
    markers = {4: "s", 8: "^"}

    for ax, metrics, ent_metrics, label in zip(
        axes, [metrics1, metrics2], [ent_conf1, ent_conf2], [MODEL1_LABEL, MODEL2_LABEL]
    ):
        fb = metrics["fixed_budget"]
        fb_x = [int(k) for k in sorted(fb.keys(), key=int)]
        fb_y = [fb[str(n)]["acc"] for n in fb_x]
        ax.plot(fb_x, fb_y, "k-o", markersize=5, linewidth=1.5, label="Fixed-budget SC")

        oracle = metrics["oracle_gate"]
        ax.plot(oracle["mean_compute"], oracle["acc"], "k*", markersize=12,
                label="Oracle gate (upper bound)")

        for k in _K_VALUES:
            rgate = metrics["realistic_gate"][f"k{k}"]
            pts = sorted([(v["compute"], v["acc"]) for v in rgate.values()])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linestyle=lss[k], marker=markers[k], color="black",
                    markersize=5, linewidth=1.2, label=f"Agreement gate k={k}")

        # Entropy gate (confirmatory only)
        if ent_metrics and ent_metrics.get("entropy_gate", {}).get("available"):
            ent_sweep = ent_metrics["entropy_gate"].get("sweep", [])
            if ent_sweep:
                ent_x = [s["compute"] for s in ent_sweep]
                ent_y = [s["acc"] for s in ent_sweep]
                ax.plot(ent_x, ent_y, linestyle="-.", marker="D", color="black",
                        markersize=4, linewidth=1.2, label=f"Entropy gate k={_ENTROPY_K} (n=151)")

        ax.set_xlabel("Mean compute (samples / problem)")
        ax.set_ylabel("Mean MV accuracy")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Accuracy vs compute: both models", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _make_calibration_both(
    cal1: list[dict], cal2: list[dict], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.array([0.375, 0.625, 0.875])
    y1 = [r["fraction_correct"] for r in cal1]
    y2 = [r["fraction_correct"] for r in cal2]
    n1 = [r["n"] for r in cal1]
    n2 = [r["n"] for r in cal2]
    ax.plot(x, y1, "k-o", markersize=7, linewidth=1.5, label=MODEL1_LABEL)
    ax.plot(x, y2, "k--s", markersize=7, linewidth=1.5, label=MODEL2_LABEL)
    ax.plot([0.25, 1.0], [0.25, 1.0], color="gray", linewidth=0.8, linestyle=":")
    for xi, yi, ni in zip(x, y1, n1):
        ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    for xi, yi, ni in zip(x, y2, n2):
        ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIDENCE_BIN_LABELS, fontsize=9)
    ax.set_xlabel("Confidence bin (plurality fraction)")
    ax.set_ylabel("Fraction plurality answer is correct")
    ax.set_title("Confidence vs. accuracy by agreement bin")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _make_gate_sweep_both(metrics1: dict, metrics2: dict, out_path: Path) -> None:
    """Heatmap-style grid: agreement-gate accuracy and delta vs fixed-budget across k x tau."""
    tau_vals = _TAU_GRID
    k_labels = ["k=4", "k=8"]
    k_keys = ["k4", "k8"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Agreement gate sweep: accuracy and gain over fixed-budget N=64", fontsize=11)

    for col, (metrics, label) in enumerate([(metrics1, MODEL1_LABEL), (metrics2, MODEL2_LABEL)]):
        fb64 = metrics["fixed_budget"]["64"]["acc"]
        # Build accuracy grid [k x tau]
        acc_grid = np.zeros((2, len(tau_vals)))
        delta_grid = np.zeros((2, len(tau_vals)))
        for ki, kkey in enumerate(k_keys):
            for ti, tau in enumerate(tau_vals):
                tkey = f"tau_{str(tau).replace('.', '_')}"
                entry = metrics["realistic_gate"][kkey].get(tkey, {})
                acc = entry.get("acc", fb64)
                acc_grid[ki, ti] = acc
                delta_grid[ki, ti] = acc - fb64

        # Top row: accuracy
        ax_acc = axes[0, col]
        im1 = ax_acc.imshow(acc_grid, aspect="auto", cmap="RdYlGn",
                             vmin=min(fb64 * 0.95, acc_grid.min()),
                             vmax=max(fb64 * 1.05, acc_grid.max()))
        ax_acc.set_xticks(range(len(tau_vals)))
        ax_acc.set_xticklabels([f"{t:.2f}" for t in tau_vals], fontsize=7, rotation=45)
        ax_acc.set_yticks(range(2))
        ax_acc.set_yticklabels(k_labels)
        ax_acc.set_title(f"{label} — accuracy")
        ax_acc.axhline(1.5, color="white", linewidth=0.5)
        # Annotate cells
        for ki in range(2):
            for ti in range(len(tau_vals)):
                ax_acc.text(ti, ki, f"{acc_grid[ki, ti]:.3f}", ha="center", va="center",
                            fontsize=6, color="black")
        plt.colorbar(im1, ax=ax_acc, fraction=0.046)

        # Bottom row: delta = acc - fixed_N64
        ax_delta = axes[1, col]
        max_abs = max(abs(delta_grid).max(), 0.001)
        im2 = ax_delta.imshow(delta_grid, aspect="auto", cmap="RdYlGn",
                               vmin=-max_abs, vmax=max_abs)
        ax_delta.set_xticks(range(len(tau_vals)))
        ax_delta.set_xticklabels([f"{t:.2f}" for t in tau_vals], fontsize=7, rotation=45)
        ax_delta.set_yticks(range(2))
        ax_delta.set_yticklabels(k_labels)
        ax_delta.set_xlabel("tau")
        ax_delta.set_title(f"{label} — accuracy - fixed N=64 ({fb64:.3f})")
        for ki in range(2):
            for ti in range(len(tau_vals)):
                ax_delta.text(ti, ki, f"{delta_grid[ki, ti]:+.3f}", ha="center", va="center",
                              fontsize=6, color="black")
        plt.colorbar(im2, ax=ax_delta, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full phase-13 analysis."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    exploratory_ids, confirmatory_ids, all_ids = _load_problem_splits()
    logger.info(
        "Splits: exploratory=%d, confirmatory=%d, pooled=%d",
        len(exploratory_ids), len(confirmatory_ids), len(all_ids),
    )

    # Load samples
    logger.info("Loading samples (Qwen)...")
    s1_all = _load_samples(_SAMPLES1_DIR, all_ids)
    logger.info("Loading samples (Llama)...")
    s2_all = _load_samples(_SAMPLES2_DIR, all_ids)

    # Availability check
    for label, s, ids in [("Qwen", s1_all, all_ids), ("Llama", s2_all, all_ids)]:
        found = [p for p in ids if p in s]
        total_samp = sum(s[p]["n_total"] for p in found)
        parsed = sum(
            sum(1 for a in s[p]["answers"] if a is not None)
            for p in found
        )
        parse_rate = parsed / total_samp if total_samp > 0 else 0.0
        logger.info(
            "%s: %d/%d problems loaded, %d total samples, parse rate=%.4f",
            label, len(found), len(ids), total_samp, parse_rate,
        )

    # Compute MV curves for all 198 × 2 models
    logger.info("Computing MV curves (Qwen, 198 problems)...")
    rng_mv1 = np.random.default_rng(42)
    mv1_all = {p: _compute_mv_curve(s1_all[p], rng_mv1) for p in all_ids if p in s1_all}
    logger.info("Computing MV curves (Llama, 198 problems)...")
    rng_mv2 = np.random.default_rng(42)
    mv2_all = {p: _compute_mv_curve(s2_all[p], rng_mv2) for p in all_ids if p in s2_all}

    # Split MV curves (conf splits used for PH1-PH4, pooled for figures)
    mv1_conf = {p: mv1_all[p] for p in confirmatory_ids if p in mv1_all}
    mv2_conf = {p: mv2_all[p] for p in confirmatory_ids if p in mv2_all}

    # Gate metrics per split and pooled
    logger.info("Computing gate metrics...")
    valid_expl = sorted(set(exploratory_ids) & s1_all.keys() & mv1_all.keys())
    valid_conf = sorted(set(confirmatory_ids) & s1_all.keys() & mv1_all.keys())
    valid_all = sorted(set(all_ids) & s1_all.keys() & mv1_all.keys())

    valid_expl_2 = sorted(set(exploratory_ids) & s2_all.keys() & mv2_all.keys())
    valid_conf_2 = sorted(set(confirmatory_ids) & s2_all.keys() & mv2_all.keys())
    valid_all_2 = sorted(set(all_ids) & s2_all.keys() & mv2_all.keys())

    logger.info("Qwen: expl=%d conf=%d pool=%d", len(valid_expl), len(valid_conf), len(valid_all))
    logger.info("Llama: expl=%d conf=%d pool=%d", len(valid_expl_2), len(valid_conf_2), len(valid_all_2))

    m1_expl = _gate_metrics(s1_all, valid_expl, mv1_all)
    m1_conf = _gate_metrics(s1_all, valid_conf, mv1_all)
    m1_pool = _gate_metrics(s1_all, valid_all, mv1_all)

    m2_expl = _gate_metrics(s2_all, valid_expl_2, mv2_all)
    m2_conf = _gate_metrics(s2_all, valid_conf_2, mv2_all)
    m2_pool = _gate_metrics(s2_all, valid_all_2, mv2_all)

    # PH1-PH4 on confirmatory set
    logger.info("Evaluating PH1-PH4 on confirmatory set...")
    ph_results = _evaluate_ph(m1_conf, m2_conf, mv1_conf, mv2_conf, valid_conf)

    # Bootstrap CIs for pooled
    rng_b1 = np.random.default_rng(_SEED_BOOT)
    rng_b2 = np.random.default_rng(_SEED_BOOT + 10)
    bf_ci1_pool = _bootstrap_backfire_ci(mv1_all, valid_all, rng_b1)
    bf_ci2_pool = _bootstrap_backfire_ci(mv2_all, valid_all_2, rng_b2)

    # Save gate summaries
    _GATE1_DIR.mkdir(parents=True, exist_ok=True)
    _GATE2_DIR.mkdir(parents=True, exist_ok=True)

    gate1_summary = {
        "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "n_problems_total": len(valid_all),
        "split": {
            "exploratory": len(valid_expl),
            "confirmatory": len(valid_conf),
            "pooled": len(valid_all),
        },
        "exploratory": m1_expl,
        "confirmatory": m1_conf,
        "pooled": m1_pool,
        "bootstrap_ci_95_backfire_pooled": list(bf_ci1_pool),
    }
    (_GATE1_DIR / "gate_summary.json").write_text(json.dumps(gate1_summary, indent=2))

    gate2_summary = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "n_problems_total": len(valid_all_2),
        "split": {
            "exploratory": len(valid_expl_2),
            "confirmatory": len(valid_conf_2),
            "pooled": len(valid_all_2),
        },
        "exploratory": m2_expl,
        "confirmatory": m2_conf,
        "pooled": m2_pool,
        "bootstrap_ci_95_backfire_pooled": list(bf_ci2_pool),
    }
    (_GATE2_DIR / "gate_summary.json").write_text(json.dumps(gate2_summary, indent=2))

    # Save confirmatory results
    (_GATE2_DIR / "confirmatory_results.json").write_text(json.dumps(ph_results, indent=2))

    # Cross-model summary
    cross_summary = {
        "model_1": {
            "id": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "label": MODEL1_LABEL,
            "exploratory": _compact_metrics(m1_expl),
            "confirmatory": _compact_metrics(m1_conf),
            "pooled": _compact_metrics(m1_pool),
            "backfire_ci_pooled_95": list(bf_ci1_pool),
        },
        "model_2": {
            "id": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            "label": MODEL2_LABEL,
            "exploratory": _compact_metrics(m2_expl),
            "confirmatory": _compact_metrics(m2_conf),
            "pooled": _compact_metrics(m2_pool),
            "backfire_ci_pooled_95": list(bf_ci2_pool),
        },
        "confirmatory_hypotheses": ph_results,
        "figures": [
            "outputs/gate_model2/backfire_both.png",
            "outputs/gate_model2/pareto_both.png",
            "outputs/gate_model2/calibration_both.png",
            "outputs/gate_model2/gate_sweep_both.png",
        ],
    }
    (_GATE2_DIR / "cross_model_summary.json").write_text(json.dumps(cross_summary, indent=2))
    logger.info("Saved gate summaries and cross_model_summary.json")

    # Figures (pooled 198, entropy gate on confirmatory)
    logger.info("Generating figures...")
    gains1 = [mv1_all[p]["64"] - mv1_all[p]["1"] for p in valid_all]
    gains2 = [mv2_all[p]["64"] - mv2_all[p]["1"] for p in valid_all_2]
    _make_backfire_both(gains1, gains2, _GATE2_DIR / "backfire_both.png")
    _make_pareto_both(m1_pool, m2_pool, m1_conf, m2_conf, _GATE2_DIR / "pareto_both.png")
    _make_calibration_both(
        m1_pool["calibration"], m2_pool["calibration"], _GATE2_DIR / "calibration_both.png"
    )
    _make_gate_sweep_both(m1_pool, m2_pool, _GATE2_DIR / "gate_sweep_both.png")

    # Print summary table
    _print_summary(m1_expl, m1_conf, m1_pool, m2_expl, m2_conf, m2_pool, ph_results, bf_ci1_pool, bf_ci2_pool)


def _compact_metrics(m: dict) -> dict:
    """Extract headline numbers from gate_metrics dict."""
    return {
        "n_problems": m["n_problems"],
        "n1_acc": round(m["fixed_budget"]["1"]["acc"], 4),
        "n64_acc": round(m["fixed_budget"]["64"]["acc"], 4),
        "backfire_rate": round(m["backfire"]["fraction_backfire"], 4),
        "backfire_n": m["backfire"]["n_backfire"],
        "mv_gain_min": round(m["backfire"]["min_gain"], 4),
        "mv_gain_max": round(m["backfire"]["max_gain"], 4),
        "oracle_acc": round(m["oracle_gate"]["acc"], 4),
        "oracle_compute": round(m["oracle_gate"]["mean_compute"], 2),
        "oracle_gain_over_64": round(m["oracle_gate"]["gain_over_fixed_64"], 4),
        "gate_k8_tau075": {
            "acc": round(m["realistic_gate"]["k8"].get("tau_0_75", {}).get("acc", float("nan")), 4),
            "compute": round(m["realistic_gate"]["k8"].get("tau_0_75", {}).get("compute", float("nan")), 2),
        },
        "fraction_oracle_ceiling_captured_k8": round(
            m["verdict"]["fraction_oracle_ceiling_captured_k8"], 4
        ),
        "entropy_gate_best_ceiling_captured": (
            round(m["entropy_gate"]["best"].get("ceiling_captured", float("nan")), 4)
            if m.get("entropy_gate", {}).get("available")
            else None
        ),
        "calibration": m["calibration"],
    }


def _print_summary(
    m1e: dict, m1c: dict, m1p: dict,
    m2e: dict, m2c: dict, m2p: dict,
    ph: dict,
    bf_ci1: tuple, bf_ci2: tuple,
) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("Phase 13 Analysis Summary")
    print(sep)

    for split_label, m1, m2 in [
        ("Exploratory (n=47)", m1e, m2e),
        ("Confirmatory (n=151)", m1c, m2c),
        ("Pooled (n=198)", m1p, m2p),
    ]:
        print(f"\n--- {split_label} ---")
        fb1 = m1["fixed_budget"]
        fb2 = m2["fixed_budget"]
        rows = [
            ("N=1 acc",                f"{fb1['1']['acc']:.4f}",           f"{fb2['1']['acc']:.4f}"),
            ("N=64 MV acc",            f"{fb1['64']['acc']:.4f}",          f"{fb2['64']['acc']:.4f}"),
            ("Backfire rate",          f"{m1['backfire']['fraction_backfire']:.4f}",
                                       f"{m2['backfire']['fraction_backfire']:.4f}"),
            ("Backfire n",             str(m1["backfire"]["n_backfire"]),  str(m2["backfire"]["n_backfire"])),
            ("mv_gain min",            f"{m1['backfire']['min_gain']:.4f}", f"{m2['backfire']['min_gain']:.4f}"),
            ("mv_gain max",            f"{m1['backfire']['max_gain']:.4f}", f"{m2['backfire']['max_gain']:.4f}"),
            ("Oracle acc",             f"{m1['oracle_gate']['acc']:.4f}",  f"{m2['oracle_gate']['acc']:.4f}"),
            ("Oracle compute",         f"{m1['oracle_gate']['mean_compute']:.2f}",
                                       f"{m2['oracle_gate']['mean_compute']:.2f}"),
            ("Oracle gain over N=64",  f"{m1['oracle_gate']['gain_over_fixed_64']:.4f}",
                                       f"{m2['oracle_gate']['gain_over_fixed_64']:.4f}"),
            ("Gate k=8 tau=0.75 acc",  f"{m1['realistic_gate']['k8'].get('tau_0_75',{}).get('acc',0):.4f}",
                                       f"{m2['realistic_gate']['k8'].get('tau_0_75',{}).get('acc',0):.4f}"),
            ("Gate k=8 tau=0.75 cmp",  f"{m1['realistic_gate']['k8'].get('tau_0_75',{}).get('compute',0):.2f}",
                                       f"{m2['realistic_gate']['k8'].get('tau_0_75',{}).get('compute',0):.2f}"),
            ("Ceil captured k8",       f"{m1['verdict']['fraction_oracle_ceiling_captured_k8']:.4f}",
                                       f"{m2['verdict']['fraction_oracle_ceiling_captured_k8']:.4f}"),
        ]
        ent1 = m1.get("entropy_gate", {})
        ent2 = m2.get("entropy_gate", {})
        if ent1.get("available"):
            ec1 = ent1.get("best", {}).get("ceiling_captured", float("nan"))
            rows.append(("Entropy gate best ceil", f"{ec1:.4f}", ""))
        if ent2.get("available"):
            ec2 = ent2.get("best", {}).get("ceiling_captured", float("nan"))
            rows[-1] = rows[-1][:2] + (f"{ec2:.4f}",)

        print(f"  {'Metric':<38} {MODEL1_LABEL:>12} {MODEL2_LABEL:>12}")
        print(f"  {'-'*38} {'-'*12} {'-'*12}")
        for name, v1, v2 in rows:
            print(f"  {name:<38} {v1:>12} {v2:>12}")

        # Calibration
        print("\n  Calibration (confidence bin -> fraction correct):")
        cal1 = m1["calibration"]
        cal2 = m2["calibration"]
        for r1, r2 in zip(cal1, cal2):
            print(f"    {r1['bin_label']:15} n={r1['n']:3d} frac={r1['fraction_correct']:.4f}   "
                  f"n={r2['n']:3d} frac={r2['fraction_correct']:.4f}")

    # Pooled backfire CI
    print(f"\n  Pooled backfire 95% CI (Qwen):  [{bf_ci1[0]:.3f}, {bf_ci1[1]:.3f}]")
    print(f"  Pooled backfire 95% CI (Llama): [{bf_ci2[0]:.3f}, {bf_ci2[1]:.3f}]")

    # PH1-PH4
    print("\n--- Confirmatory Hypotheses (PH1-PH4) ---")
    for phname, phdata in ph.items():
        v1 = phdata.get("model_1", {}).get("value", phdata.get("model_1", {}).get("best_ceiling_captured", "N/A"))
        v2 = phdata.get("model_2", {}).get("value", phdata.get("model_2", {}).get("best_ceiling_captured", "N/A"))
        result = "PASS" if phdata["pass"] else ("FAIL" if phdata["pass"] is False else "NOT EVALUATED")
        print(f"  {phname}: {result}  (M1={v1}, M2={v2}, threshold={phdata['threshold']})")

    print(sep)


if __name__ == "__main__":
    main()
