"""Verify that extra stored samples do not change the backfire classification.

At the analysis temperature of 0.7, Qwen's 151 confirmatory problems hold
exactly 64 samples and its 47 exploratory problems hold 65 to 72, retained from
earlier sampling runs, so sample-pool size and split membership coincide
exactly. Three further problems hold side-test samples at other temperatures,
which the temperature filter below removes.

MV_acc(N) draws subsets of size min(N, n) without replacement from all n stored
samples, so for those 47 problems MV_acc(64) is an expectation over subsets of
up to 72 rather than of 64. This script classifies every problem twice, once
from all stored samples and once from the first 64 only, and reports whether
any problem changes backfire label.

The MV accuracy helper replicates `_mv_acc_at_n` in
scripts/run_phase13_analysis.py: exact enumeration when comb(n_total, n) does
not exceed 5000, Monte Carlo over 2000 draws otherwise, with lexicographic
tie-breaking and UNPARSEABLE extractions mapped to None. It differs in one
respect: the published run draws from a single RNG stream shared across all
problems, whereas this script seeds a fresh generator per problem so the result
does not depend on iteration order. Monte Carlo estimates may therefore differ
from the published ones within sampling error, which affects only the
zero-versus-positive split; backfire itself is mv_gain < 0.

No API calls. Reads outputs/samples/*.jsonl and outputs/gate/gate_summary.json.
Outputs: outputs/truncation_invariance.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
_SAMPLES_DIR = ROOT / "outputs" / "samples"
_GATE_SUMMARY = ROOT / "outputs" / "gate" / "gate_summary.json"
OUT_PATH = ROOT / "outputs" / "truncation_invariance.json"

_T_MAIN: float = 0.7
_TRUNCATE_N: int = 64
_MAX_EXACT: int = 5000
_N_MC: int = 2000
_SEED: int = 42
_UNPARSEABLE: tuple[str | None, ...] = ("UNPARSEABLE", None, "")


def _plurality(answers: list[str | None]) -> str | None:
    """Return the plurality answer, breaking ties lexicographically.

    Ground-truth agnostic, so tie-breaking cannot leak label information.

    Args:
        answers: Extracted answers, with None for unparseable samples.

    Returns:
        The plurality answer, or None if no answer is valid.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    max_c = max(counts.values())
    return sorted(a for a, c in counts.items() if c == max_c)[0]


def _mv_acc_at_n(
    answers: list[str | None], gt: str, n: int, rng: np.random.Generator
) -> float:
    """Expected majority-vote accuracy over size-n subsets of the stored samples.

    Replicates `_mv_acc_at_n` in run_phase13_analysis.py, including the
    exact-versus-Monte-Carlo branch cutoff.

    Args:
        answers: Extracted answers, with None for unparseable samples.
        gt: Ground-truth answer letter.
        n: Vote size.
        rng: Generator used only on the Monte Carlo branch.

    Returns:
        Fraction of size-n subsets whose plurality equals the ground truth.
    """
    valid = [a for a in answers if a is not None]
    n_total = len(valid)
    n = min(n, n_total)
    if n == 0:
        return 0.0
    if comb(n_total, n) <= _MAX_EXACT:
        hits = sum(1 for s in combinations(valid, n) if _plurality(list(s)) == gt)
        return hits / comb(n_total, n)
    hits = 0
    for _ in range(_N_MC):
        idxs = rng.choice(n_total, size=n, replace=False)
        if _plurality([valid[i] for i in idxs]) == gt:
            hits += 1
    return hits / _N_MC


def _load_answers(
    path: Path, truncate: int | None = None
) -> tuple[list[str | None], str]:
    """Load temperature-0.7 answers for one problem, ordered by sample index.

    Mirrors `_load_samples` in run_phase13_analysis.py.

    Args:
        path: Path to the problem's JSONL sample file.
        truncate: If given, keep only the first this-many records after sorting.

    Returns:
        A tuple of (answers, ground_truth).
    """
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if abs(obj.get("temperature", -1) - _T_MAIN) < 1e-6:
            records.append(obj)
    records.sort(key=lambda s: s.get("sample_idx", 0))
    if truncate is not None:
        records = records[:truncate]
    gt = str(records[0]["ground_truth"])
    answers: list[str | None] = [
        (
            r.get("extracted_answer")
            if r.get("extracted_answer") not in _UNPARSEABLE
            else None
        )
        for r in records
    ]
    return answers, gt


def _classify(
    answers: list[str | None], gt: str, rng: np.random.Generator
) -> tuple[float, str]:
    """Return (mv_gain, label) for one problem.

    Backfire is defined strictly as mv_gain < 0, matching run_phase13_analysis.

    Args:
        answers: Extracted answers, with None for unparseable samples.
        gt: Ground-truth answer letter.
        rng: Generator passed through to the Monte Carlo branch.

    Returns:
        A tuple of (mv_gain, one of "backfire", "zero", "positive").
    """
    gain = _mv_acc_at_n(answers, gt, 64, rng) - _mv_acc_at_n(answers, gt, 1, rng)
    if gain < 0:
        return gain, "backfire"
    if gain == 0:
        return gain, "zero"
    return gain, "positive"


def main() -> None:
    """Compare truncated and untruncated backfire classification for Qwen."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = sorted(_SAMPLES_DIR.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no sample files under {_SAMPLES_DIR}")

    full_counts: Counter[str] = Counter()
    trunc_counts: Counter[str] = Counter()
    relabelled: list[dict[str, object]] = []
    n_extra = 0

    for path in paths:
        pid = path.stem
        full_answers, gt = _load_answers(path)
        trunc_answers, _ = _load_answers(path, truncate=_TRUNCATE_N)
        if len(full_answers) > _TRUNCATE_N:
            n_extra += 1

        full_gain, full_label = _classify(
            full_answers, gt, np.random.default_rng(_SEED)
        )
        trunc_gain, trunc_label = _classify(
            trunc_answers, gt, np.random.default_rng(_SEED)
        )
        full_counts[full_label] += 1
        trunc_counts[trunc_label] += 1

        if full_label != trunc_label:
            relabelled.append(
                {
                    "problem_id": pid,
                    "n_stored": len(full_answers),
                    "full_label": full_label,
                    "truncated_label": trunc_label,
                    "full_mv_gain": round(full_gain, 6),
                    "truncated_mv_gain": round(trunc_gain, 6),
                }
            )

    n = len(paths)
    backfire_moved = [
        r for r in relabelled if "backfire" in (r["full_label"], r["truncated_label"])
    ]

    published = json.loads(_GATE_SUMMARY.read_text())["pooled"]["backfire"]
    result = {
        "description": (
            "Backfire classification computed from all stored samples versus from "
            "the first 64 per problem. Backfire is mv_gain < 0."
        ),
        "model": "Qwen2.5-7B",
        "n_problems": n,
        "n_problems_with_extra_samples": n_extra,
        "truncate_n": _TRUNCATE_N,
        "all_stored_samples": {
            "n_backfire": full_counts["backfire"],
            "n_zero_gain": full_counts["zero"],
            "n_positive_gain": full_counts["positive"],
            "backfire_rate": round(full_counts["backfire"] / n, 4),
        },
        "truncated_to_64": {
            "n_backfire": trunc_counts["backfire"],
            "n_zero_gain": trunc_counts["zero"],
            "n_positive_gain": trunc_counts["positive"],
            "backfire_rate": round(trunc_counts["backfire"] / n, 4),
        },
        "published_pooled": {
            "n_backfire": published["n_backfire"],
            "n_zero_gain": published["n_zero_gain"],
            "n_positive_gain": published["n_positive_gain"],
            "backfire_rate": round(published["fraction_backfire"], 4),
        },
        "n_problems_relabelled": len(relabelled),
        "n_problems_crossing_backfire_boundary": len(backfire_moved),
        "relabelled": relabelled,
        "backfire_classification_identical": len(backfire_moved) == 0,
        "backfire_rate_identical": full_counts["backfire"] == trunc_counts["backfire"],
        "NOTE_monte_carlo": (
            "Problems with more than 64 stored samples use the Monte Carlo branch "
            "when untruncated. This script seeds per problem rather than sharing "
            "one stream as the published run does, so zero-versus-positive labels "
            "may differ from the published counts within sampling error."
        ),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    logger.info("Problems                    : %d (%d with >64 stored)", n, n_extra)
    logger.info(
        "All stored samples          : backfire=%d  zero=%d  positive=%d  rate=%.4f",
        full_counts["backfire"],
        full_counts["zero"],
        full_counts["positive"],
        full_counts["backfire"] / n,
    )
    logger.info(
        "Truncated to %d             : backfire=%d  zero=%d  positive=%d  rate=%.4f",
        _TRUNCATE_N,
        trunc_counts["backfire"],
        trunc_counts["zero"],
        trunc_counts["positive"],
        trunc_counts["backfire"] / n,
    )
    logger.info(
        "Published pooled            : backfire=%d  zero=%d  positive=%d  rate=%.4f",
        published["n_backfire"],
        published["n_zero_gain"],
        published["n_positive_gain"],
        published["fraction_backfire"],
    )
    logger.info("Problems relabelled         : %d", len(relabelled))
    logger.info("Crossing backfire boundary  : %d", len(backfire_moved))
    logger.info("Written to %s", OUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
