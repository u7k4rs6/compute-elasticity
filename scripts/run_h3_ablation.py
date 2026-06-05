"""H3 entropy/diversity ablation — workshop paper prep.

Investigates whether H3's entropy > diversity result reflects genuine
signal-class superiority or sample-size asymmetry (entropy: dense per-token
data from 1 call; diversity: only 6 pair distances at N=4).

CASE B applies: Phase 6 sample files do not store logprobs, so entropy_N4
and nll_N4 cannot be computed without new API calls. Only diversity variants
are computed here.

Computed variants:
  diversity_mean_N4  (existing, loaded from outputs/diversity/)
  diversity_max_N4   (recomputed: max pairwise cosine dist, same 4 traces)
  diversity_mean_N8  (recomputed: mean pairwise, 8-trace seed=42 subsample)
  diversity_max_N8   (recomputed: max pairwise, 8-trace seed=42 subsample)

Existing baselines (loaded from hypothesis_results.json):
  entropy_N1, nll_N1

Output: outputs/h3_ablation.json

Usage:
    source .venv/bin/activate
    python scripts/run_h3_ablation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_T_MAIN: float = 0.7
_N_TRACES_N4: int = 4
_N_TRACES_N8: int = 8
_SEED: int = 42

_OUTPUTS_DIR = ROOT / "outputs"
_SAMPLES_DIR = _OUTPUTS_DIR / "samples"
_DIVERSITY_DIR = _OUTPUTS_DIR / "diversity"
_ENTROPY_DIR = _OUTPUTS_DIR / "entropy_baseline"
_FITS_DIR = _OUTPUTS_DIR / "fits"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def _main_pilot_ids() -> list[str]:
    locked = json.loads((ROOT / "data" / "problem_ids.json").read_text())
    gate = set(
        json.loads((_OUTPUTS_DIR / "gate_minus_1_labels.json").read_text())["gate_problems"]
    )
    return sorted(pid for pid in locked if pid not in gate)


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _load_all_traces(problem_id: str, temperature: float) -> list[str]:
    """Return full_response strings for all samples at given temperature, sorted by sample_idx."""
    path = _SAMPLES_DIR / f"{problem_id}.jsonl"
    if not path.exists():
        return []
    samples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if abs(obj.get("temperature", -1) - temperature) < 1e-6:
                samples.append(obj)
        except json.JSONDecodeError:
            continue
    samples.sort(key=lambda s: s.get("sample_idx", 0))
    return [s["full_response"] for s in samples]


# ---------------------------------------------------------------------------
# Pairwise distance helpers
# ---------------------------------------------------------------------------


def _pairwise_cosine_distances(embeddings: np.ndarray) -> list[float]:
    """Return all pairwise cosine distances for normalised embeddings."""
    n = len(embeddings)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            dists.append(1.0 - sim)
    return dists


def _diversity_mean(dists: list[float]) -> float:
    return float(np.mean(dists)) if dists else 0.0


def _diversity_max(dists: list[float]) -> float:
    return float(np.max(dists)) if dists else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Compute H3 ablation variants and save to outputs/h3_ablation.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from pilot.analysis import compute_auc
    from pilot.config import EMBEDDING_MODEL
    from pilot.diversity import truncate_trace

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed")
        sys.exit(1)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Loaded embedder: %s", EMBEDDING_MODEL)

    # Load existing baselines from hypothesis_results.json
    hr = json.loads((_OUTPUTS_DIR / "hypothesis_results.json").read_text())
    auc_entropy_n1 = hr["H3"]["auc_entropy"]
    auc_nll_n1 = hr["H3"]["auc_nll"]
    median_elasticity = hr["H3"]["median_elasticity_threshold"]

    main_ids = _main_pilot_ids()
    rng = np.random.default_rng(_SEED)

    # Per-problem arrays
    existing_div_mean_n4: list[float] = []
    div_max_n4: list[float] = []
    div_mean_n8: list[float] = []
    div_max_n8: list[float] = []
    elasticities: list[float] = []
    problem_ids_used: list[str] = []

    for num, pid in enumerate(main_ids, start=1):
        div_path = _DIVERSITY_DIR / f"{pid}.json"
        fit_path = _FITS_DIR / f"{pid}.json"

        if not div_path.exists() or not fit_path.exists():
            logger.warning("[%d/%d] %s: missing diversity or fit — skipping", num, len(main_ids), pid)
            continue

        all_traces = _load_all_traces(pid, _T_MAIN)
        if len(all_traces) < _N_TRACES_N8:
            logger.warning("[%d/%d] %s: only %d traces, need %d — skipping", num, len(main_ids), pid, len(all_traces), _N_TRACES_N8)
            continue

        # N=4: first 4 traces (consistent with original diversity_mean_N4)
        traces_n4 = all_traces[:_N_TRACES_N4]
        truncated_n4 = [truncate_trace(t)[0] for t in traces_n4]
        emb_n4: np.ndarray = embedder.encode(truncated_n4, normalize_embeddings=True)
        dists_n4 = _pairwise_cosine_distances(emb_n4)

        # N=8: seed=42 subsample from all available traces
        idx8 = rng.choice(len(all_traces), size=_N_TRACES_N8, replace=False)
        traces_n8 = [all_traces[i] for i in sorted(idx8)]
        truncated_n8 = [truncate_trace(t)[0] for t in traces_n8]
        emb_n8: np.ndarray = embedder.encode(truncated_n8, normalize_embeddings=True)
        dists_n8 = _pairwise_cosine_distances(emb_n8)

        # Existing diversity_mean_N4 from file
        existing_div = float(json.loads(div_path.read_text())["diversity"])
        elasticity = float(json.loads(fit_path.read_text())["mean_elasticity_8_64"])

        existing_div_mean_n4.append(existing_div)
        div_max_n4.append(_diversity_max(dists_n4))
        div_mean_n8.append(_diversity_mean(dists_n8))
        div_max_n8.append(_diversity_max(dists_n8))
        elasticities.append(elasticity)
        problem_ids_used.append(pid)

        logger.info(
            "[%d/%d] %s  div_mean_N4=%.4f  div_max_N4=%.4f  div_mean_N8=%.4f  div_max_N8=%.4f",
            num, len(main_ids), pid,
            existing_div,
            div_max_n4[-1],
            div_mean_n8[-1],
            div_max_n8[-1],
        )

    n = len(problem_ids_used)
    logger.info("Processed %d problems", n)

    elasticity_arr = np.array(elasticities)
    labels = (elasticity_arr > median_elasticity).astype(int)

    auc_div_mean_n4 = compute_auc(np.array(existing_div_mean_n4), labels)
    auc_div_max_n4 = compute_auc(np.array(div_max_n4), labels)
    auc_div_mean_n8 = compute_auc(np.array(div_mean_n8), labels)
    auc_div_max_n8 = compute_auc(np.array(div_max_n8), labels)

    # Build interpretation string
    entropy_wins_n4 = auc_entropy_n1 > auc_div_mean_n4
    n8_closes_gap = (auc_div_mean_n8 - auc_div_mean_n4) > 0.05
    max_beats_mean = (auc_div_max_n4 - auc_div_mean_n4) > 0.03 or (auc_div_max_n8 - auc_div_mean_n8) > 0.03

    if entropy_wins_n4 and not n8_closes_gap:
        explanation = "signal-class: entropy_N1 beats diversity at any N tested; doubling traces does not close the gap"
    elif entropy_wins_n4 and n8_closes_gap:
        explanation = "mixed: entropy still leads but diversity_mean_N8 partially closes the gap, suggesting both effects contribute"
    else:
        explanation = "sample-size: diversity_mean_N8 matches or exceeds entropy_N1; N4 gap was an artefact of too few pair comparisons"

    if max_beats_mean:
        explanation += "; max aggregation outperforms mean, suggesting outlier trace pairs carry more elasticity signal"

    result: dict[str, Any] = {
        "n_problems": n,
        "median_elasticity_threshold": median_elasticity,
        "logprobs_in_phase6_samples": False,
        "case": "B",
        "case_note": "Phase 6 sample files store full_response text only (no logprobs). entropy_N4 and nll_N4 require a new sampling pass (~$0.30).",
        "seed_n8_subsample": _SEED,
        "variants": {
            "entropy_N1": {"auc": auc_entropy_n1, "source": "existing"},
            "entropy_N4": None,
            "nll_N1": {"auc": auc_nll_n1, "source": "existing"},
            "nll_N4": None,
            "diversity_mean_N4": {"auc": round(auc_div_mean_n4, 4), "source": "existing"},
            "diversity_max_N4": {"auc": round(auc_div_max_n4, 4), "source": "computed"},
            "diversity_mean_N8": {"auc": round(auc_div_mean_n8, 4), "source": "computed"},
            "diversity_max_N8": {"auc": round(auc_div_max_n8, 4), "source": "computed"},
        },
        "interpretation": explanation,
    }

    out_path = _OUTPUTS_DIR / "h3_ablation.json"
    out_path.write_text(json.dumps(result, indent=2))

    sep = "=" * 72
    print(f"\n{sep}")
    print("H3 Ablation Results")
    print(sep)
    print(f"  Case: B (no logprobs in Phase 6 samples)")
    print(f"  Problems: {n}  |  median elasticity threshold: {median_elasticity:.6f}")
    print()
    print(f"  {'Variant':<24} {'AUC':>6}  Source")
    print(f"  {'-'*24} {'-'*6}  {'-'*8}")
    print(f"  {'entropy_N1':<24} {auc_entropy_n1:>6.4f}  existing")
    print(f"  {'entropy_N4':<24} {'  n/a':>6}  case B")
    print(f"  {'nll_N1':<24} {auc_nll_n1:>6.4f}  existing")
    print(f"  {'nll_N4':<24} {'  n/a':>6}  case B")
    print(f"  {'diversity_mean_N4':<24} {auc_div_mean_n4:>6.4f}  existing")
    print(f"  {'diversity_max_N4':<24} {auc_div_max_n4:>6.4f}  computed")
    print(f"  {'diversity_mean_N8':<24} {auc_div_mean_n8:>6.4f}  computed")
    print(f"  {'diversity_max_N8':<24} {auc_div_max_n8:>6.4f}  computed")
    print()
    print(f"  Interpretation: {explanation}")
    print(sep)


if __name__ == "__main__":
    main()
