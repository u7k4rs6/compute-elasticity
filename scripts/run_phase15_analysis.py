"""Phase 15: log-prob margin gate and per-domain backfire breakdown.

MARGIN GATE: NOT COMPUTABLE.
Per-token logprob arrays (tokens/token_logprobs/top_logprobs) were not stored
during Phase 13 sampling. Final-answer log-prob margin requires the rank-1 and
rank-2 log-probabilities at the extracted answer token position, which requires
per-token arrays. Only the mean_token_entropy scalar was saved per sample
(Qwen only; Llama has no entropy data). Re-sampling with per-token logprob
storage is required before this gate can be evaluated.

PER-DOMAIN BREAKDOWN: COMPUTABLE.
For each of 198 problems in both models, compute mv_gain = MV_acc(64) - MV_acc(1)
using Monte Carlo (same methodology as Phase 13 / Phase 14a), then group by domain
(biology, chemistry, physics).

Outputs: outputs/phase15_results.json
"""

from __future__ import annotations

import collections
import glob
import json
import logging
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
_OUTPUTS = ROOT / "outputs"
OUT_PATH = _OUTPUTS / "phase15_results.json"

_N_VALUES: list[int] = [1, 2, 4, 8, 16, 32, 64]
_N_MC: int = 2000
_T_MAIN: float = 0.7
_RNG_SEED: int = 42

_MODELS: list[tuple[str, Path]] = [
    ("Qwen2.5-7B", _OUTPUTS / "samples"),
    ("Llama-3-8B", _OUTPUTS / "samples_model2"),
]


# ---------------------------------------------------------------------------
# Problem split
# ---------------------------------------------------------------------------


def _load_problem_splits() -> tuple[list[str], list[str], list[str]]:
    """Return (exploratory_ids, confirmatory_ids, all_ids) sorted.

    Replicates the same logic as run_phase13_analysis._load_problem_splits.
    """
    locked = set(json.loads((ROOT / "data" / "problem_ids.json").read_text()))
    gate_set = set(
        json.loads((_OUTPUTS / "gate_minus_1_labels.json").read_text())["gate_problems"]
    )
    exploratory = sorted(locked - gate_set)
    all_ids = sorted(
        Path(f).stem
        for f in glob.glob(str(_OUTPUTS / "samples" / "gpqa_diamond_*.jsonl"))
    )
    confirmatory = sorted(p for p in all_ids if p not in set(exploratory))
    return exploratory, confirmatory, all_ids


# ---------------------------------------------------------------------------
# MV curve computation (mirrors Phase 13 / Phase 14a logic)
# ---------------------------------------------------------------------------


def _plurality(answers: list[str]) -> str:
    """Plurality winner with lexicographic tie-breaking on equal counts."""
    c = collections.Counter(answers)
    max_c = max(c.values())
    return sorted(a for a, cnt in c.items() if cnt == max_c)[0]


def _mv_acc_at_n(
    answers: list[str | None],
    gt: str,
    n: int,
    rng: np.random.Generator,
    max_exact: int = 5000,
) -> float:
    """Expected majority-vote accuracy at subset size n."""
    valid = [a for a in answers if a is not None]
    n_total = len(valid)
    if n_total == 0:
        return 0.0
    n = min(n, n_total)
    if comb(n_total, n) <= max_exact:
        hits = sum(
            1 for subset in combinations(valid, n) if _plurality(list(subset)) == gt
        )
        return hits / comb(n_total, n)
    hits = 0
    for _ in range(_N_MC):
        subset = list(rng.choice(valid, size=n, replace=False))
        if _plurality(subset) == gt:
            hits += 1
    return hits / _N_MC


def _compute_mv_curve(
    answers: list[str | None], gt: str, rng: np.random.Generator
) -> dict[str, float]:
    """Return {str(n): mv_acc} for all N_VALUES."""
    return {str(n): _mv_acc_at_n(answers, gt, n, rng) for n in _N_VALUES}


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _load_problem(samples_dir: Path, pid: str) -> dict | None:
    """Load answers, ground truth, and subject for one problem.

    Returns None if the JSONL file is missing or has no T=0.7 records.
    """
    path = samples_dir / f"{pid}.jsonl"
    if not path.exists():
        return None
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    recs = [r for r in recs if abs(r.get("temperature", -1) - _T_MAIN) < 1e-6]
    if not recs:
        return None
    gt = str(recs[0].get("ground_truth", ""))
    subject = recs[0].get("subject", "unknown")
    answers: list[str | None] = [
        (
            r.get("extracted_answer")
            if r.get("extracted_answer") not in (None, "", "UNPARSEABLE")
            else None
        )
        for r in recs
    ]
    return {"gt": gt, "subject": subject, "answers": answers}


# ---------------------------------------------------------------------------
# Per-domain backfire computation
# ---------------------------------------------------------------------------


def _per_domain_backfire(
    model_label: str,
    samples_dir: Path,
    all_ids: list[str],
    rng: np.random.Generator,
) -> dict:
    """Compute per-domain backfire rates for one model across all 198 problems."""
    domain_acc: dict[str, dict] = collections.defaultdict(
        lambda: {"n_backfire": 0, "n_total": 0, "mv_gains": []}
    )

    for pid in all_ids:
        data = _load_problem(samples_dir, pid)
        if data is None:
            logger.warning("Missing data for %s / %s", model_label, pid)
            continue
        mv = _compute_mv_curve(data["answers"], data["gt"], rng)
        mv_gain = mv["64"] - mv["1"]
        domain = data["subject"]
        domain_acc[domain]["mv_gains"].append(mv_gain)
        domain_acc[domain]["n_total"] += 1
        if mv_gain < 0:
            domain_acc[domain]["n_backfire"] += 1

    result: dict = {}
    all_gains: list[float] = []
    all_backfire: list[int] = []
    for domain in sorted(domain_acc):
        d = domain_acc[domain]
        n = d["n_total"]
        bf = d["n_backfire"]
        gains = d["mv_gains"]
        all_gains.extend(gains)
        all_backfire.append(bf)
        result[domain] = {
            "n_problems": n,
            "n_backfire": bf,
            "backfire_rate": round(bf / n, 4) if n > 0 else None,
            "mean_mv_gain": round(float(np.mean(gains)), 4) if gains else None,
            "median_mv_gain": round(float(np.median(gains)), 4) if gains else None,
        }

    total_n = sum(d["n_total"] for d in domain_acc.values())
    total_bf = sum(d["n_backfire"] for d in domain_acc.values())
    result["_pooled"] = {
        "n_problems": total_n,
        "n_backfire": total_bf,
        "backfire_rate": round(total_bf / total_n, 4) if total_n > 0 else None,
        "mean_mv_gain": round(float(np.mean(all_gains)), 4) if all_gains else None,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Compute Phase 15 results and write outputs/phase15_results.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _, _, all_ids = _load_problem_splits()
    rng = np.random.default_rng(_RNG_SEED)

    # --- Margin gate: document limitation ---
    margin_gate: dict = {
        "status": "NOT_COMPUTABLE",
        "reason": (
            "Per-token logprob arrays (tokens/token_logprobs/top_logprobs) were not "
            "stored during Phase 13 sampling. Final-answer log-prob margin = "
            "logprob(rank-1) - logprob(rank-2) at the extracted answer token requires "
            "per-token arrays. Only the mean_token_entropy scalar was saved per sample "
            "(Qwen2.5-7B only; Llama-3-8B has no entropy data). "
            "Re-sampling with per-token logprob storage is required."
        ),
        "what_is_available": {
            "Qwen2.5-7B": "mean_token_entropy scalar per sample (confirmatory 151)",
            "Llama-3-8B": "no entropy or logprob data available",
        },
        "reference_phase14a": {
            "mean_entropy_auc_qwen": 0.6306,
            "mean_entropy_auc_llama": 0.5227,
            "note": (
                "These AUC values use mean per-token entropy (not final-answer margin), "
                "computed in Phase 14a from the stored mean_token_entropy scalar."
            ),
        },
        "auc_comparison": {
            "margin_auc_qwen": None,
            "margin_auc_llama": None,
            "gate_acc_best_threshold_qwen": None,
            "gate_acc_best_threshold_llama": None,
            "ceiling_capture_vs_grid_oracle_qwen": None,
            "ceiling_capture_vs_grid_oracle_llama": None,
        },
    }

    # --- Per-domain backfire ---
    logger.info("Computing per-domain backfire rates (pooled 198)...")
    per_domain: dict = {}
    for model_label, samples_dir in _MODELS:
        logger.info("  %s ...", model_label)
        per_domain[model_label] = _per_domain_backfire(
            model_label, samples_dir, all_ids, rng
        )

    results: dict = {
        "EXPLORATORY": True,
        "description": (
            "Phase 15: log-prob margin gate (NOT COMPUTABLE from existing data) "
            "and per-domain backfire rates (biology/chemistry/physics) for both models."
        ),
        "margin_gate": margin_gate,
        "per_domain_backfire_pooled_198": per_domain,
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", OUT_PATH)

    # --- stdout summary ---
    print("\n=== PHASE 15 SUMMARY ===\n")

    print("MARGIN GATE: NOT COMPUTABLE")
    print("  Per-token logprob arrays not stored in Phase 13.")
    print("  Final-answer log-prob margin requires re-sampling with array storage.")
    print("  Phase 14a reference: mean_entropy AUC = Qwen 0.6306, Llama 0.5227")

    print("\nPER-DOMAIN BACKFIRE RATES (pooled 198 problems)")
    for model, domains in per_domain.items():
        print(f"\n  {model}:")
        for domain, d in sorted(domains.items()):
            if domain == "_pooled":
                continue
            print(
                f"    {domain:<12}: n={d['n_problems']:>3}  "
                f"backfire={d['n_backfire']:>3}/{d['n_problems']}  "
                f"({d['backfire_rate']:.1%})  "
                f"mean_gain={d['mean_mv_gain']:+.3f}"
            )
        p = domains["_pooled"]
        print(
            f"    {'[pooled]':<12}: n={p['n_problems']:>3}  "
            f"backfire={p['n_backfire']:>3}/{p['n_problems']}  "
            f"({p['backfire_rate']:.1%})  "
            f"mean_gain={p['mean_mv_gain']:+.3f}"
        )


if __name__ == "__main__":
    main()
