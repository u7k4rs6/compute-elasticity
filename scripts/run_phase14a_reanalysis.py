"""Phase 14a re-analysis: localized entropy, grid oracle, threshold documentation.

No new API calls. Reads from:
- outputs/entropy_local/entropy_local_summary.json (mean-entropy AUC)
- outputs/gate/gate_summary.json (Qwen2.5-7B)
- outputs/gate_model2/gate_summary.json (Llama-3-8B)

Tasks:
  TASK 1 - Mean-entropy AUC on confirmatory 151 (localized entropy NOT computable:
           per-token logprob arrays were not stored in Phase 13 sampling).
  TASK 2 - Grid oracle accuracy, ceiling, and gate ceiling-captures (both models).
           Ceiling captures use MV_acc(N=1) as baseline, not N=64.
  TASK 3 - Exact entropy threshold and gate accuracy at that threshold (both models).

Outputs: outputs/phase14a_results.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
_GATE1 = ROOT / "outputs" / "gate"
_GATE2 = ROOT / "outputs" / "gate_model2"
_ENTROPY_LOCAL = ROOT / "outputs" / "entropy_local"
OUT_PATH = ROOT / "outputs" / "phase14a_results.json"

_MODELS: list[tuple[str, Path]] = [
    ("Qwen2.5-7B", _GATE1 / "gate_summary.json"),
    ("Llama-3-8B", _GATE2 / "gate_summary.json"),
]


def _ceil_cap(
    gate_acc: float | None, mv1: float, grid_oracle_acc: float
) -> float | None:
    """Ceiling capture using MV_acc(N=1) as baseline.

    Returns (gate_acc - mv1) / (grid_oracle_acc - mv1).
    """
    denom = grid_oracle_acc - mv1
    if gate_acc is None or abs(denom) < 1e-9:
        return None
    return round((gate_acc - mv1) / denom, 4)


def _task1(el: dict) -> dict:
    """Entropy AUC summary (TASK 1).

    Per-token arrays not stored; localized entropy not computable. Reports
    mean-entropy AUC from stored scalar.
    """
    ea = el["entropy_analysis"]
    return {
        "LOCALIZATION_NOTE": (
            "Per-token logprob arrays were NOT stored during Phase 13 sampling "
            "(only the mean_token_entropy scalar per sample was saved). "
            "True final-20-token localized entropy is NOT computable from existing data. "
            "Mean-entropy AUC below uses the stored scalar and replicates the Phase 14a value."
        ),
        "mean_entropy_auc_confirmatory_151": {
            model: {
                "roc_auc": ea[model]["roc_auc"]["mean_entropy"],
                "interpretation": ea[model]["roc_auc"]["interpretation"],
            }
            for model in ("Qwen2.5-7B", "Llama-3-8B")
        },
        "localized_entropy_auc": {
            "status": "NOT_COMPUTABLE",
            "reason": "per-token logprob arrays not stored in Phase 13 sampling",
        },
    }


def _task2_split(gs: dict, split_name: str) -> dict:
    """Grid oracle metrics for one split of one model (TASK 2).

    Ceiling captures use MV_acc(N=1) as baseline:
      cap = (gate_acc - mv1) / (grid_oracle_acc - mv1)
    """
    split_data = gs[split_name]
    fb = split_data["fixed_budget"]
    mv1: float = fb["1"]["acc"]
    mv64: float = fb["64"]["acc"]

    go = split_data["grid_oracle"]
    grid_oracle_acc: float = go["grid_oracle_acc"]
    grid_oracle_ceiling = round(grid_oracle_acc - mv1, 4)

    rg_k8_tau075 = (
        split_data.get("realistic_gate", {}).get("k8", {}).get("tau_0_75", {})
    )
    ag_acc: float | None = rg_k8_tau075.get("acc")

    best_eg: dict = split_data.get("entropy_gate", {}).get("best", {}) or {}
    entropy_acc: float | None = best_eg.get("acc")

    return {
        "mv1_acc": round(mv1, 4),
        "mv64_acc": round(mv64, 4),
        "grid_oracle_acc": grid_oracle_acc,
        "grid_oracle_ceiling_vs_n1": grid_oracle_ceiling,
        "agreement_gate_k8_tau075": {
            "acc": round(ag_acc, 4) if ag_acc is not None else None,
            "ceiling_capture_n1_baseline": _ceil_cap(ag_acc, mv1, grid_oracle_acc),
        },
        "entropy_gate_best": {
            "threshold": best_eg.get("threshold"),
            "acc": round(entropy_acc, 4) if entropy_acc is not None else None,
            "ceiling_capture_n1_baseline": _ceil_cap(entropy_acc, mv1, grid_oracle_acc),
            "NOTE": "threshold selected on confirmatory 151 problems",
        },
    }


def _task2(models: list[tuple[str, Path]]) -> dict:
    """Grid oracle across splits and models (TASK 2)."""
    out: dict = {}
    for split_name, n_problems in [("confirmatory", 151), ("pooled", 198)]:
        split_key = f"{split_name}_{n_problems}"
        out[split_key] = {}
        for model_name, gs_path in models:
            gs = json.loads(gs_path.read_text())
            out[split_key][model_name] = _task2_split(gs, split_name)
    return out


def _task3(models: list[tuple[str, Path]]) -> dict:
    """Exact entropy threshold and gate accuracy (TASK 3, confirmatory 151)."""
    out: dict = {}
    for model_name, gs_path in models:
        gs = json.loads(gs_path.read_text())
        best = gs["confirmatory"]["entropy_gate"].get("best", {}) or {}
        sweep = gs["confirmatory"]["entropy_gate"].get("sweep", []) or []
        out[model_name] = {
            "threshold": best.get("threshold"),
            "gate_acc": best.get("acc"),
            "mean_compute_tokens": best.get("compute"),
            "ceiling_captured_vs_binary_oracle": best.get("ceiling_captured"),
            "n_sweep_thresholds": len(sweep),
            "NOTE_baseline": (
                "ceiling_captured uses the binary oracle (max of MV_acc(1), MV_acc(64)) "
                "and MV_acc(64) as fixed baseline, as defined in Phase 13 pre-registration. "
                "See task2_grid_oracle for ceiling captures vs the grid oracle."
            ),
        }
    return out


def main() -> None:
    """Build outputs/phase14a_results.json from existing gate_summary data."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    el = json.loads((_ENTROPY_LOCAL / "entropy_local_summary.json").read_text())

    results: dict = {
        "EXPLORATORY": True,
        "description": (
            "Phase 14a re-analysis: localized entropy (NOT computable), "
            "grid oracle accuracy and ceiling captures, entropy threshold documentation."
        ),
        "task1_entropy_auc": _task1(el),
        "task2_grid_oracle": _task2(_MODELS),
        "task3_entropy_threshold": _task3(_MODELS),
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", OUT_PATH)

    # --- stdout summary ---
    print("\n=== PHASE 14A RE-ANALYSIS SUMMARY ===\n")

    print("TASK 1: Entropy AUC (confirmatory 151)")
    t1 = results["task1_entropy_auc"]
    print(f"  LIMITATION: {t1['LOCALIZATION_NOTE'][:80]}...")
    for m, v in t1["mean_entropy_auc_confirmatory_151"].items():
        print(f"  Mean-entropy AUC ({m}): {v['roc_auc']:.4f}")
    print("  Localized AUC (final 20 tokens): NOT COMPUTABLE")

    print("\nTASK 2: Grid Oracle")
    for split_key, split_data in results["task2_grid_oracle"].items():
        print(f"\n  [{split_key}]")
        for model, d in split_data.items():
            ag = d["agreement_gate_k8_tau075"]
            eg = d["entropy_gate_best"]
            print(f"  {model}:")
            print(
                f"    MV(N=1)={d['mv1_acc']:.4f}  "
                f"grid_oracle={d['grid_oracle_acc']:.4f}  "
                f"grid_ceil(vs N=1)={d['grid_oracle_ceiling_vs_n1']:.4f}"
            )
            print(
                f"    Agree gate k8 tau0.75: acc={ag['acc']}  "
                f"cap(N=1 base)={ag['ceiling_capture_n1_baseline']}"
            )
            print(
                f"    Entropy gate (best):   acc={eg['acc']}  "
                f"cap(N=1 base)={eg['ceiling_capture_n1_baseline']}"
            )

    print("\nTASK 3: Entropy Threshold (confirmatory 151)")
    for m, d in results["task3_entropy_threshold"].items():
        thr = d["threshold"]
        acc = d["gate_acc"]
        cap = d["ceiling_captured_vs_binary_oracle"]
        cmp = d["mean_compute_tokens"]
        print(
            f"  {m}: threshold={thr:.8f}  gate_acc={acc:.4f}  "
            f"ceiling_cap={cap:.5f}  compute={cmp:.1f}"
        )


if __name__ == "__main__":
    main()
