"""GPQA Diamond dataset loading and stratified sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pilot.config import STRATIFIED_SEED

logger = logging.getLogger(__name__)

_SUBJECT_FIELD = "Subdomain"
_QUESTION_FIELD = "Question"
_CORRECT_FIELD = "Correct Answer"

_OPTION_FIELDS = (
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)

# Explicit mapping of all GPQA Diamond subdomain labels → 3 broad categories.
# Covers every subdomain observed in the gpqa_diamond split (198 problems).
_SUBDOMAIN_TO_SUBJECT: dict[str, str] = {
    # physics
    "astrophysics": "physics",
    "condensed matter physics": "physics",
    "electromagnetism and photonics": "physics",
    "high-energy particle physics": "physics",
    "optics and acoustics": "physics",
    "physics (general)": "physics",
    "quantum mechanics": "physics",
    "relativistic mechanics": "physics",
    # chemistry
    "chemistry (general)": "chemistry",
    "inorganic chemistry": "chemistry",
    "organic chemistry": "chemistry",
    # biology
    "genetics": "biology",
    "molecular biology": "biology",
}


@dataclass(frozen=True, slots=True)
class Problem:
    """A single GPQA Diamond problem."""

    id: str
    subject: str  # "physics" | "chemistry" | "biology"
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    ground_truth: str  # "A" | "B" | "C" | "D"


def _normalize_subject(raw: str) -> str:
    key = raw.strip().lower()
    subject = _SUBDOMAIN_TO_SUBJECT.get(key)
    if subject is None:
        logger.warning("Unknown GPQA subdomain %r — falling back to raw value.", raw)
        return key
    return subject


def _build_problem(idx: int, row: dict) -> Problem:
    """Build a Problem from a raw dataset row, shuffling options deterministically."""
    import random

    rng = random.Random(idx)

    correct_text = str(row[_CORRECT_FIELD])
    incorrect_texts = [str(row[f]) for f in _OPTION_FIELDS]

    options = [correct_text] + incorrect_texts
    rng.shuffle(options)

    correct_idx = options.index(correct_text)
    answer_label = "ABCD"[correct_idx]

    return Problem(
        id=f"gpqa_diamond_{idx:04d}",
        subject=_normalize_subject(str(row.get(_SUBJECT_FIELD, "unknown"))),
        question=str(row[_QUESTION_FIELD]),
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
        ground_truth=answer_label,
    )


def load_gpqa_diamond() -> list[Problem]:
    """Load the full GPQA Diamond split from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Run: pip install datasets"
        ) from exc

    import os

    token = os.environ.get("HF_TOKEN") or None
    logger.info("Loading GPQA Diamond from HuggingFace…")
    ds = load_dataset(
        "Idavidrein/gpqa",
        "gpqa_diamond",
        split="train",
        token=token,
    )
    problems = [_build_problem(i, dict(row)) for i, row in enumerate(ds)]
    logger.info("Loaded %d GPQA Diamond problems.", len(problems))
    return problems


def stratified_sample(
    problems: list[Problem],
    n: int = 50,
    seed: int = STRATIFIED_SEED,
) -> list[Problem]:
    """Stratified sample of n problems preserving subject proportions.

    Args:
        problems: Full problem list from load_gpqa_diamond().
        n: Target sample size.
        seed: RNG seed for reproducibility.

    Returns:
        List of n problems with balanced subject representation.
    """
    import random

    rng = random.Random(seed)

    by_subject: dict[str, list[Problem]] = {}
    for p in problems:
        by_subject.setdefault(p.subject, []).append(p)

    for bucket in by_subject.values():
        rng.shuffle(bucket)

    subjects = sorted(by_subject.keys())
    n_subjects = len(subjects)
    base_per_subject = n // n_subjects
    remainder = n - base_per_subject * n_subjects

    # Allocate extras to the largest buckets first
    quota: dict[str, int] = {}
    sorted_by_size = sorted(subjects, key=lambda s: -len(by_subject[s]))
    for i, s in enumerate(sorted_by_size):
        quota[s] = base_per_subject + (1 if i < remainder else 0)

    sample: list[Problem] = []
    for s in subjects:
        q = min(quota[s], len(by_subject[s]))
        if q < quota[s]:
            logger.warning(
                "Subject %r has only %d problems; requested %d.",
                s,
                len(by_subject[s]),
                quota[s],
            )
        sample.extend(by_subject[s][:q])

    rng.shuffle(sample)
    return sample
