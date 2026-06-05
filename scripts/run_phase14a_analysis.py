"""Phase 14a -- localized-entropy gate re-analysis and grid oracle.

EXPLORATORY (post-hoc): all results in this script are exploratory.
Pre-registered PH1-PH4 verdicts are unchanged and not re-evaluated here.

A1. Entropy-based ROC-AUC and gate analysis (confirmatory set, 151 problems).
    LIMITATION: Per-token logprob arrays were NOT stored in Phase 13 sampling --
    only the response-level mean_token_entropy scalar was persisted. True localized
    entropy at the final-answer token or over the last 10 tokens is therefore NOT
    computable from stored data. This analysis uses:
      - mean_entropy_of_problem: mean of mean_token_entropy across all 64 samples
      - std_entropy_of_problem: std of mean_token_entropy across 64 samples
      - min_entropy_of_problem: min (most-confident sample in the problem)
    ROC-AUC measures how well each signal predicts backfire (mv_gain < 0).
    Gate: threshold on k=4 probe mean entropy; sweep thresholds; report best
    oracle-ceiling capture vs PH4 mean-entropy gate.
    Outputs: outputs/entropy_local/entropy_local_summary.json

A2. Grid oracle across N in {1,2,4,8,16,32,64} (all 198 problems, both models).
    grid_oracle_acc = mean over problems of max_N(mv_acc_N).
    grid_oracle_mean_compute = mean of the optimal N per problem.
    Ceiling captured by gates is recomputed against the grid oracle denominator.
    New keys added to gate_summary files; binary-oracle keys untouched.

A3. Documents that all reported entropy-gate ceiling-capture values are the
    MAXIMUM over the threshold sweep (best-case gate performance).
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
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_OUTPUTS = ROOT / "outputs"
_SAMPLES1 = _OUTPUTS / "samples"
_SAMPLES2 = _OUTPUTS / "samples_model2"
_GATE1 = _OUTPUTS / "gate"
_GATE2 = _OUTPUTS / "gate_model2"
_ENTROPY_LOCAL = _OUTPUTS / "entropy_local"

_T_MAIN: float = 0.7
_N_VALUES: list[int] = [1, 2, 4, 8, 16, 32, 64]
_N_MC: int = 2000
_SEED_MV: int = 42
_N_BOOTSTRAP: int = 1000
_SEED_BOOT: int = 42
_ENTROPY_K: int = 4  # probe samples for entropy gate
_N_ENTROPY_THRESHOLDS: int = 40
_ENTROPY_SEED: int = 44

MODEL1_LABEL = "Qwen2.5-7B"
MODEL2_LABEL = "Llama-3-8B"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Problem set
# ---------------------------------------------------------------------------


def _load_problem_splits() -> tuple[list[str], list[str], list[str]]:
    """Return (exploratory_ids, confirmatory_ids, all_ids)."""
    locked = set(json.loads((ROOT / "data" / "problem_ids.json").read_text()))
    gate_set = set(
        json.loads((_OUTPUTS / "gate_minus_1_labels.json").read_text())["gate_problems"]
    )
    exploratory = sorted(locked - gate_set)

    from pilot.data_loader import load_gpqa_diamond

    all_problems = load_gpqa_diamond()
    all_ids = sorted(p.id for p in all_problems)
    confirmatory = sorted(pid for pid in all_ids if pid not in set(exploratory))
    return exploratory, confirmatory, all_ids


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _load_samples(samples_dir: Path, pids: list[str]) -> dict[str, dict]:
    """Return {pid: {answers, gt, n_total, entropies}} for T=0.7 records."""
    data: dict[str, dict] = {}
    for pid in pids:
        fpath = samples_dir / f"{pid}.jsonl"
        if not fpath.exists():
            continue
        recs = [
            json.loads(line) for line in fpath.read_text().splitlines() if line.strip()
        ]
        recs = [r for r in recs if r.get("temperature") == _T_MAIN]
        if not recs:
            continue
        answers = [r["extracted_answer"] for r in recs if r.get("extracted_answer")]
        gt = recs[0].get("ground_truth") or recs[0].get("correct_answer", "")
        entropies = [
            r["mean_token_entropy"]
            for r in recs
            if r.get("mean_token_entropy") is not None
        ]
        data[pid] = {
            "answers": answers,
            "gt": gt,
            "n_total": len(answers),
            "entropies": entropies,
        }
    return data


# ---------------------------------------------------------------------------
# MV curve computation
# ---------------------------------------------------------------------------


def _plurality(answers: list[str]) -> str:
    """Plurality winner with lexicographic tie-breaking."""
    c = Counter(answers)
    return sorted(c.keys(), key=lambda x: (-c[x], x))[0]


def _mv_acc_at_n(
    answers: list[str], gt: str, n: int, rng: np.random.Generator
) -> float:
    """Estimate majority-vote accuracy at draw size n."""
    from math import comb

    total = len(answers)
    if n >= total:
        return 1.0 if _plurality(answers) == gt else 0.0
    if comb(total, n) <= 5000:
        from itertools import combinations

        wins = sum(
            1 for combo in combinations(answers, n) if _plurality(list(combo)) == gt
        )
        return wins / comb(total, n)
    draws = rng.integers(0, total, size=(_N_MC, n))
    return float(
        np.mean(
            [1 if _plurality([answers[i] for i in row]) == gt else 0 for row in draws]
        )
    )


def _compute_mv_curves(
    samples: dict[str, dict], pids: list[str]
) -> dict[str, dict[str, float]]:
    """Return {pid: {str(n): mv_acc}} for all N in _N_VALUES."""
    rng = np.random.default_rng(_SEED_MV)
    curves: dict[str, dict[str, float]] = {}
    for pid in pids:
        d = samples.get(pid)
        if d is None:
            continue
        ans = d["answers"]
        gt = d["gt"]
        if not ans or not gt:
            continue
        curves[pid] = {str(n): _mv_acc_at_n(ans, gt, n, rng) for n in _N_VALUES}
    return curves


# ---------------------------------------------------------------------------
# A2: Grid oracle
# ---------------------------------------------------------------------------


def _grid_oracle_metrics(
    mv_curves: dict[str, dict[str, float]], pids: list[str]
) -> dict[str, Any]:
    """Grid oracle: best N per problem across {1,2,4,8,16,32,64}.

    Returns dict with:
      grid_oracle_acc, grid_oracle_mean_compute, grid_oracle_gain_over_64,
      binary_oracle_acc (for reference), grid_vs_binary_gain
    """
    valid = [p for p in pids if p in mv_curves]
    if not valid:
        return {}

    best_n_per_problem = []
    best_acc_per_problem = []
    for p in valid:
        curve = mv_curves[p]
        best_n = max(_N_VALUES, key=lambda n: curve[str(n)])
        best_acc_per_problem.append(curve[str(best_n)])
        best_n_per_problem.append(best_n)

    fixed_64 = float(np.mean([mv_curves[p]["64"] for p in valid]))
    grid_oracle_acc = float(np.mean(best_acc_per_problem))
    grid_oracle_compute = float(np.mean(best_n_per_problem))

    # Binary oracle: max(MV_1, MV_64)
    binary_oracle_acc = float(
        np.mean([max(mv_curves[p]["1"], mv_curves[p]["64"]) for p in valid])
    )

    return {
        "n_problems": len(valid),
        "fixed_64_acc": round(fixed_64, 4),
        "binary_oracle_acc": round(binary_oracle_acc, 4),
        "binary_oracle_gain_over_64": round(binary_oracle_acc - fixed_64, 4),
        "grid_oracle_acc": round(grid_oracle_acc, 4),
        "grid_oracle_mean_compute": round(grid_oracle_compute, 2),
        "grid_oracle_gain_over_64": round(grid_oracle_acc - fixed_64, 4),
        "grid_vs_binary_gain": round(grid_oracle_acc - binary_oracle_acc, 4),
        "NOTE_A3": (
            "All entropy/agreement gate ceiling-capture values in phase-13 "
            "analysis are the MAXIMUM over the respective threshold sweep "
            "(best-case gate performance against the binary oracle). "
            "Grid-oracle ceiling captures below recompute against the wider denominator."
        ),
    }


def _grid_ceiling_captures(
    mv_curves: dict[str, dict[str, float]],
    pids: list[str],
    gate_k8_tau075_acc: float,
    entropy_gate_best_acc: float | None,
) -> dict[str, Any]:
    """Recompute ceiling captures for gates vs grid oracle."""
    valid = [p for p in pids if p in mv_curves]
    if not valid:
        return {}
    grid_acc = float(
        np.mean([max(mv_curves[p][str(n)] for n in _N_VALUES) for p in valid])
    )
    fixed_64 = float(np.mean([mv_curves[p]["64"] for p in valid]))
    denom = grid_acc - fixed_64

    def cap(gate_acc: float | None) -> float | None:
        if gate_acc is None or abs(denom) < 1e-9:
            return None
        return round((gate_acc - fixed_64) / denom, 4)

    return {
        "grid_oracle_acc": round(grid_acc, 4),
        "grid_denom": round(denom, 4),
        "agreement_gate_k8_tau075_ceiling_vs_grid": cap(gate_k8_tau075_acc),
        "entropy_gate_best_ceiling_vs_grid": cap(entropy_gate_best_acc),
    }


# ---------------------------------------------------------------------------
# A1: Entropy ROC-AUC and gate
# ---------------------------------------------------------------------------


def _entropy_auc_and_gate(
    samples: dict[str, dict],
    mv_curves: dict[str, dict[str, float]],
    pids: list[str],
    model_label: str,
) -> dict[str, Any]:
    """Entropy re-analysis for the given problem set.

    Returns ROC-AUC for mean/std/min entropy predicting backfire,
    plus entropy-gate best ceiling capture.

    NOTE: Localized variants (final-answer-token, last-10-tokens) are NOT
    computable because per-token logprob arrays were not stored in Phase 13.
    Only the response-mean scalar (mean_token_entropy) is available.
    """
    valid = [p for p in pids if p in samples and p in mv_curves]
    # Keep only problems with entropy data
    with_entropy = [p for p in valid if samples[p].get("entropies")]
    if not with_entropy:
        return {"error": "No entropy data available for this problem set"}

    # Build per-problem features
    mean_ents: list[float] = []
    std_ents: list[float] = []
    min_ents: list[float] = []
    backfire_labels: list[int] = []
    mv_gains: list[float] = []

    for p in with_entropy:
        ents = samples[p]["entropies"]
        mv_gain = mv_curves[p]["64"] - mv_curves[p]["1"]
        mean_ents.append(float(np.mean(ents)))
        std_ents.append(float(np.std(ents)) if len(ents) > 1 else 0.0)
        min_ents.append(float(np.min(ents)))
        backfire_labels.append(1 if mv_gain < 0 else 0)
        mv_gains.append(mv_gain)

    n_backfire = sum(backfire_labels)
    n_problems = len(backfire_labels)

    # ROC-AUC: higher entropy -> predict backfire=1 (direction: higher = more backfire)
    # Try both directions and report the better one (standard practice)
    def safe_auc(signal: list[float], labels: list[int]) -> float:
        if len(set(labels)) < 2:
            return float("nan")
        # AUC can be < 0.5 if signal is inversely correlated; report as-is
        return float(roc_auc_score(labels, signal))

    auc_mean = safe_auc(mean_ents, backfire_labels)
    auc_std = safe_auc(std_ents, backfire_labels)
    auc_min = safe_auc(min_ents, backfire_labels)

    # Distributional overlap description: AUC = P(entropy_backfire > entropy_gain)
    # when AUC > 0.5: backfire problems tend to have HIGHER entropy (less confident)
    # when AUC < 0.5: backfire problems tend to have LOWER entropy (more confident)
    # when AUC ~ 0.5: no separation
    def overlap_description(auc: float) -> str:
        if abs(auc - 0.5) < 0.05:
            return "no meaningful separation (AUC near 0.5)"
        direction = "higher" if auc > 0.5 else "lower"
        strength = "strong" if abs(auc - 0.5) > 0.15 else "weak"
        return f"{strength} tendency for backfire problems to have {direction} entropy (AUC={auc:.3f})"

    # Entropy gate sweep (k=4 probe samples, threshold on probe mean entropy).
    # Matches Phase 13 methodology: 2000 MC draws per problem.
    # Optimized: precompute all draws once, then sweep thresholds without re-drawing.
    fixed_64_acc = float(np.mean([mv_curves[p]["64"] for p in with_entropy]))
    oracle_acc = float(
        np.mean([max(mv_curves[p]["1"], mv_curves[p]["64"]) for p in with_entropy])
    )
    oracle_denom = oracle_acc - fixed_64_acc

    k = _ENTROPY_K
    rng_gate = np.random.default_rng(_ENTROPY_SEED)
    n_draws = _N_MC

    # Precompute per-problem: (probe_mean_ents [n_draws], probe_correct [n_draws],
    # mv64_correct) for each problem
    precomputed: list[tuple[np.ndarray, np.ndarray, float]] = []
    for p in with_entropy:
        ents_raw = samples[p]["entropies"]
        ans_raw = samples[p]["answers"]
        gt = samples[p]["gt"]
        pairs = [
            (a, e) for a, e in zip(ans_raw, ents_raw) if a is not None and e is not None
        ]
        if not pairs:
            continue
        ans_list = [pr[0] for pr in pairs]
        ent_list = np.array([pr[1] for pr in pairs], dtype=float)
        n_total = len(ans_list)
        probe_k = min(k, n_total)

        # Draw probe_k indices, _N_MC times
        probe_idxs = rng_gate.choice(n_total, size=(n_draws, probe_k), replace=True)
        probe_ents = ent_list[probe_idxs].mean(axis=1)  # shape (n_draws,)
        probe_correct = np.array(
            [int(_plurality([ans_list[i] for i in row]) == gt) for row in probe_idxs],
            dtype=float,
        )
        # Full MV (deterministic): plurality of all samples
        mv64_correct = float(int(_plurality(ans_list) == gt))
        precomputed.append((probe_ents, probe_correct, mv64_correct))

    # Threshold range from probe entropy distribution
    all_probe_ents_flat = np.concatenate([pc[0] for pc in precomputed])
    lo = float(np.percentile(all_probe_ents_flat, 5))
    hi = float(np.percentile(all_probe_ents_flat, 95))
    thresholds = np.linspace(lo, hi, _N_ENTROPY_THRESHOLDS)

    best_ceil = None
    best_thresh = None
    best_gate_acc = None
    best_compute = None

    for thresh in thresholds:
        prob_accs = []
        prob_computes = []
        for probe_ents, probe_correct, mv64_correct in precomputed:
            low_mask = probe_ents < thresh
            prob_low = low_mask.mean()
            # Expected accuracy = P(low)*E[probe_correct|low] + P(high)*mv64_correct
            if low_mask.any():
                acc_low = probe_correct[low_mask].mean()
            else:
                acc_low = 0.0
            gate_acc_p = prob_low * acc_low + (1.0 - prob_low) * mv64_correct
            gate_compute_p = prob_low * k + (1.0 - prob_low) * 64
            prob_accs.append(gate_acc_p)
            prob_computes.append(gate_compute_p)

        if not prob_accs:
            continue
        gate_acc = float(np.mean(prob_accs))
        gate_compute = float(np.mean(prob_computes))
        ceil = (
            (gate_acc - fixed_64_acc) / oracle_denom
            if abs(oracle_denom) > 1e-9
            else None
        )

        if ceil is not None and (best_ceil is None or ceil > best_ceil):
            best_ceil = ceil
            best_thresh = float(thresh)
            best_gate_acc = gate_acc
            best_compute = gate_compute

    # Phase 13 reference: mean entropy gate best (computed on same set)
    # (already stored in gate_summary; we recompute here for consistency)
    phase13_best_ceil = None
    try:
        gs_key = _GATE1 if model_label == MODEL1_LABEL else _GATE2
        gs = json.loads((gs_key / "gate_summary.json").read_text())
        for split_key in ["confirmatory", "pooled"]:
            eg = gs.get(split_key, {}).get("entropy_gate", {})
            if not eg or not eg.get("available"):
                continue
            sweep = eg.get("sweep", [])
            if sweep:
                phase13_best_ceil = max(
                    (s.get("ceiling_captured", float("-inf")) for s in sweep),
                    default=None,
                )
                break
    except Exception:
        pass

    result: dict[str, Any] = {
        "EXPLORATORY": True,
        "model": model_label,
        "n_problems_with_entropy": len(with_entropy),
        "n_backfire": n_backfire,
        "backfire_rate": round(n_backfire / n_problems, 4),
        "LOCALIZATION_NOTE": (
            "Per-token logprob arrays NOT stored in Phase 13 sampling. "
            "True localized entropy (final-answer-token, last-10-tokens) "
            "is not computable from stored data. Only response-mean "
            "mean_token_entropy scalar is available per sample."
        ),
        "roc_auc": {
            "mean_entropy": round(auc_mean, 4),
            "std_entropy": round(auc_std, 4),
            "min_entropy": round(auc_min, 4),
            "interpretation": (
                "AUC = P(signal_backfire > signal_gain). "
                "Higher signal = predict backfire. "
                "AUC > 0.5: backfire problems have higher entropy."
            ),
        },
        "overlap_description": {
            "mean_entropy": overlap_description(auc_mean),
            "std_entropy": overlap_description(auc_std),
            "min_entropy": overlap_description(auc_min),
        },
        "entropy_gate_k4_sweep": {
            "k": k,
            "n_thresholds": _N_ENTROPY_THRESHOLDS,
            "NOTE_A3": (
                "Best ceiling capture is the MAXIMUM over the threshold sweep "
                "(best-case gate performance). Reported for comparability with PH4."
            ),
            "best_ceiling_captured_vs_binary_oracle": (
                round(best_ceil, 4) if best_ceil is not None else None
            ),
            "best_threshold": (
                round(best_thresh, 4) if best_thresh is not None else None
            ),
            "best_gate_acc": (
                round(best_gate_acc, 4) if best_gate_acc is not None else None
            ),
            "best_gate_compute": (
                round(best_compute, 2) if best_compute is not None else None
            ),
            "phase13_reference_best_ceiling_captured": phase13_best_ceil,
        },
        "baseline": {
            "fixed_64_acc": round(fixed_64_acc, 4),
            "binary_oracle_acc": round(oracle_acc, 4),
            "oracle_headroom": round(oracle_denom, 4),
        },
    }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Phase 14a analysis."""
    logger.info("Loading problem splits...")
    expl_ids, conf_ids, all_ids = _load_problem_splits()
    logger.info(
        "Splits: exploratory=%d, confirmatory=%d, pooled=%d",
        len(expl_ids),
        len(conf_ids),
        len(all_ids),
    )

    logger.info("Loading Qwen samples...")
    s1_all = _load_samples(_SAMPLES1, all_ids)
    logger.info("Loading Llama samples...")
    s2_all = _load_samples(_SAMPLES2, all_ids)

    logger.info("Qwen: %d problems loaded", len(s1_all))
    logger.info("Llama: %d problems loaded", len(s2_all))

    # --- A2: Grid oracle requires mv curves for all 198 ---
    logger.info("Computing MV curves (Qwen, %d problems)...", len(all_ids))
    mv1 = _compute_mv_curves(s1_all, all_ids)
    logger.info("Computing MV curves (Llama, %d problems)...", len(all_ids))
    mv2 = _compute_mv_curves(s2_all, all_ids)

    logger.info("Qwen MV curves: %d problems", len(mv1))
    logger.info("Llama MV curves: %d problems", len(mv2))

    # --- A2: Grid oracle per split ---
    logger.info("Computing grid oracle...")
    grid_results: dict[str, Any] = {"EXPLORATORY": True}

    for label, mv, samples_dict in [
        (MODEL1_LABEL, mv1, s1_all),
        (MODEL2_LABEL, mv2, s2_all),
    ]:
        grid_results[label] = {}
        for split_name, pid_list in [
            ("exploratory", expl_ids),
            ("confirmatory", conf_ids),
            ("pooled", all_ids),
        ]:
            go = _grid_oracle_metrics(mv, pid_list)
            grid_results[label][split_name] = go

    # --- A2: Recompute ceiling captures vs grid oracle ---
    # Load existing gate metrics from gate_summary files
    logger.info("Recomputing gate ceiling captures vs grid oracle...")
    for label, mv, gs_path in [
        (MODEL1_LABEL, mv1, _GATE1 / "gate_summary.json"),
        (MODEL2_LABEL, mv2, _GATE2 / "gate_summary.json"),
    ]:
        if not gs_path.exists():
            logger.warning("Gate summary not found: %s", gs_path)
            continue
        gs = json.loads(gs_path.read_text())

        for split_name, pid_list in [
            ("exploratory", expl_ids),
            ("confirmatory", conf_ids),
            ("pooled", all_ids),
        ]:
            split_data = gs.get(split_name, {})
            # Get existing gate metrics from summary
            rg = split_data.get("realistic_gate", {}).get("k8", {})
            tau_key = "tau_0_75"
            gate_k8_acc = None
            if rg:
                tau_data = rg.get(tau_key, {})
                gate_k8_acc = tau_data.get("acc")

            # Entropy gate best acc (Phase 13): max acc over the threshold sweep
            eg = split_data.get("entropy_gate", {})
            entropy_best_acc = None
            if eg.get("available") and eg.get("sweep"):
                best_entry = max(eg["sweep"], key=lambda s: s.get("acc", -1))
                entropy_best_acc = best_entry.get("acc")

            caps = _grid_ceiling_captures(mv, pid_list, gate_k8_acc, entropy_best_acc)

            if "grid_oracle" not in split_data:
                split_data["grid_oracle"] = {}
            split_data["grid_oracle"].update(caps)
            split_data["grid_oracle"]["NOTE_A3"] = (
                "gate ceiling captures reported are MAXIMUM over sweep (best-case). "
                "Denominators here use the grid oracle (max over all N), "
                "not the binary oracle (max of N=1 vs N=64) used in PH1-PH4."
            )

        gs_path.write_text(json.dumps(gs, indent=2))
        logger.info("Updated %s with grid_oracle keys", gs_path)

    # --- A1: Entropy ROC-AUC on confirmatory set ---
    logger.info("Computing entropy ROC-AUC (A1)...")
    entropy_results: dict[str, Any] = {"EXPLORATORY": True}

    for label, samples_dict, mv in [
        (MODEL1_LABEL, s1_all, mv1),
        (MODEL2_LABEL, s2_all, mv2),
    ]:
        result = _entropy_auc_and_gate(samples_dict, mv, conf_ids, label)
        entropy_results[label] = result

    # --- Save outputs ---
    _ENTROPY_LOCAL.mkdir(parents=True, exist_ok=True)
    out_path = _ENTROPY_LOCAL / "entropy_local_summary.json"
    summary = {
        "EXPLORATORY": True,
        "description": (
            "Phase 14a entropy re-analysis. "
            "Localized entropy variants (answer-token, last-10-tokens) unavailable: "
            "Phase 13 sampling stored only response-mean scalar, not per-token arrays. "
            "ROC-AUC and gate analysis use mean/std/min of response-mean entropy. "
            "Grid oracle and updated gate ceiling captures in gate_summary files."
        ),
        "entropy_analysis": entropy_results,
        "grid_oracle": grid_results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s", out_path)

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("[EXPLORATORY] Phase 14a Summary")
    print("=" * 80)

    print("\n--- A1: Entropy ROC-AUC (confirmatory, n=151) ---")
    for label in [MODEL1_LABEL, MODEL2_LABEL]:
        r = entropy_results.get(label, {})
        if "error" in r:
            print(f"  {label}: {r['error']}")
            continue
        auc = r.get("roc_auc", {})
        gate = r.get("entropy_gate_k4_sweep", {})
        base = r.get("baseline", {})
        n_with = r.get("n_problems_with_entropy", 0)
        print(f"\n  {label} (n={n_with} with entropy):")
        print(f"    ROC-AUC (mean entropy):  {auc.get('mean_entropy', 'N/A'):.4f}")
        print(f"    ROC-AUC (std entropy):   {auc.get('std_entropy', 'N/A'):.4f}")
        print(f"    ROC-AUC (min entropy):   {auc.get('min_entropy', 'N/A'):.4f}")
        print(
            f"    Overlap (mean):          {r['overlap_description']['mean_entropy']}"
        )
        print(
            f"    Entropy gate best ceil:  {gate.get('best_ceiling_captured_vs_binary_oracle')}"
        )
        print(
            f"    Phase13 reference ceil:  {gate.get('phase13_reference_best_ceiling_captured')}"
        )
        print(
            f"    Gate acc at best thresh: {gate.get('best_gate_acc')} "
            f"(fixed_64={base.get('fixed_64_acc')})"
        )

    print("\n--- A2: Grid oracle (pooled, n=198) ---")
    for label in [MODEL1_LABEL, MODEL2_LABEL]:
        go = grid_results.get(label, {}).get("pooled", {})
        print(f"\n  {label}:")
        print(
            f"    Binary oracle acc:       {go.get('binary_oracle_acc')} "
            f"(+{go.get('binary_oracle_gain_over_64')} over N=64)"
        )
        print(
            f"    Grid oracle acc:         {go.get('grid_oracle_acc')} "
            f"(+{go.get('grid_oracle_gain_over_64')} over N=64)"
        )
        print(f"    Grid oracle mean compute:{go.get('grid_oracle_mean_compute')}")
        print(f"    Grid vs binary gain:     {go.get('grid_vs_binary_gain')}")

    print("\n--- A2: Gate ceiling captures vs GRID oracle (confirmatory, n=151) ---")
    for label, gs_path in [
        (MODEL1_LABEL, _GATE1 / "gate_summary.json"),
        (MODEL2_LABEL, _GATE2 / "gate_summary.json"),
    ]:
        gs = json.loads(gs_path.read_text())
        go = gs.get("confirmatory", {}).get("grid_oracle", {})
        print(f"\n  {label}:")
        print(f"    Grid oracle acc:         {go.get('grid_oracle_acc')}")
        print(
            f"    Agree gate k8 ceil/grid: {go.get('agreement_gate_k8_tau075_ceiling_vs_grid')}"
        )
        print(
            f"    Entropy gate ceil/grid:  {go.get('entropy_gate_best_ceiling_vs_grid')}"
        )

    print("\n--- A3 note ---")
    print(
        "  All Phase-13 entropy and agreement gate ceiling-capture values are the "
        "MAXIMUM over their respective threshold sweeps (best-case gate performance). "
        "This is documented in gate_summary.json under each split's grid_oracle.NOTE_A3."
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
