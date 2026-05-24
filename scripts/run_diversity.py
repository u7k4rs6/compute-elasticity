"""Phase 8 — Embedding diversity computation.

Computes CoT trace diversity at N=4 per problem via BAAI/bge-small-en-v1.5
sentence-transformer embeddings. Output feeds H3 evaluation in Phase 9.

No API calls — pure local computation.

Usage:
    source .venv/bin/activate
    python scripts/run_diversity.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_N_MAIN_PROBLEMS: int = 47
_T_MAIN: float = 0.7

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


# ---------------------------------------------------------------------------
# Problem ID selection
# ---------------------------------------------------------------------------


def _load_gate_ids() -> list[str]:
    path = ROOT / "outputs" / "gate_minus_1_labels.json"
    return json.loads(path.read_text())["gate_problems"]


def _load_locked_ids() -> list[str]:
    return json.loads((ROOT / "data" / "problem_ids.json").read_text())


def _select_main_pilot_ids(locked_ids: list[str], gate_ids: list[str]) -> list[str]:
    gate = set(gate_ids)
    ids = [pid for pid in locked_ids if pid not in gate]
    ids.sort()
    return ids


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _load_traces_at_n(
    samples_dir: Path,
    problem_id: str,
    temperature: float,
    n: int,
) -> list[str]:
    """Return full_response strings for the first n samples (by sample_idx)."""
    path = samples_dir / f"{problem_id}.jsonl"
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
    return [s["full_response"] for s in samples[:n]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_diversity() -> bool:
    """Run Phase 8 diversity computation; return True on success."""
    from pilot.config import (
        DIVERSITY_DIR,
        EMBEDDING_MODEL,
        EMBEDDING_N,
        OUTPUTS_DIR,
        SAMPLES_DIR,
    )
    from pilot.diversity import compute_diversity, truncate_trace

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
    except ImportError as exc:
        logger.error("sentence-transformers not installed: %s", exc)
        return False

    locked_ids = _load_locked_ids()
    gate_ids = _load_gate_ids()
    main_pilot_ids = _select_main_pilot_ids(locked_ids, gate_ids)

    if len(main_pilot_ids) != _N_MAIN_PROBLEMS:
        logger.error(
            "Expected %d main-pilot IDs, got %d.", _N_MAIN_PROBLEMS, len(main_pilot_ids)
        )
        return False

    DIVERSITY_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Diversity directory: %s", DIVERSITY_DIR)

    per_problem_results: list[dict[str, Any]] = []
    all_truncation_flags: list[bool] = []

    for num, pid in enumerate(main_pilot_ids, start=1):
        traces = _load_traces_at_n(SAMPLES_DIR, pid, _T_MAIN, EMBEDDING_N)
        if len(traces) < EMBEDDING_N:
            logger.warning(
                "[%d/%d] %s: only %d traces, need %d — skipping.",
                num,
                _N_MAIN_PROBLEMS,
                pid,
                len(traces),
                EMBEDDING_N,
            )
            continue

        truncated: list[str] = []
        flags: list[bool] = []
        for trace in traces:
            tt, was_truncated = truncate_trace(trace)
            truncated.append(tt)
            flags.append(was_truncated)
            all_truncation_flags.append(was_truncated)

        truncation_rate = float(sum(flags) / len(flags))
        diversity = compute_diversity(truncated, embedder)

        logger.info(
            "[%d/%d] %s  diversity=%.4f  truncation_rate=%.2f",
            num,
            _N_MAIN_PROBLEMS,
            pid,
            diversity,
            truncation_rate,
        )

        result: dict[str, Any] = {
            "problem_id": pid,
            "n_traces_used": EMBEDDING_N,
            "diversity": diversity,
            "truncation_rate": truncation_rate,
        }
        (DIVERSITY_DIR / f"{pid}.json").write_text(json.dumps(result, indent=2))
        per_problem_results.append(result)

    if not per_problem_results:
        logger.error("No diversity results produced.")
        return False

    diversities = [r["diversity"] for r in per_problem_results]
    median_div = float(np.median(diversities))
    above_median = [
        r["problem_id"] for r in per_problem_results if r["diversity"] >= median_div
    ]
    overall_truncation = (
        float(sum(all_truncation_flags) / len(all_truncation_flags))
        if all_truncation_flags
        else 0.0
    )

    summary: dict[str, Any] = {
        "n_problems": len(per_problem_results),
        "diversity_min": float(np.min(diversities)),
        "diversity_median": median_div,
        "diversity_max": float(np.max(diversities)),
        "overall_truncation_rate": overall_truncation,
        "above_median_problems": above_median,
    }
    (OUTPUTS_DIR / "diversity_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Diversity summary written.")

    sep = "=" * 72
    print(f"\n{sep}")
    print("Phase 8 — Diversity Summary")
    print(sep)
    print(f"  Problems computed      : {len(per_problem_results)}")
    print(f"  Diversity min          : {summary['diversity_min']:.4f}")
    print(f"  Diversity median       : {summary['diversity_median']:.4f}")
    print(f"  Diversity max          : {summary['diversity_max']:.4f}")
    print(f"  Overall truncation     : {overall_truncation:.2%}")
    print(f"  Above-median problems  : {len(above_median)}")
    print()
    print("  Next step: python scripts/run_analysis.py  (Phase 9)")
    print(sep)

    return True


def main() -> None:
    """Entry point."""
    _load_env()
    ok = run_diversity()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
