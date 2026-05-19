"""Tests for pilot/data_loader.py — no network calls; uses synthetic problems."""

from __future__ import annotations

from pilot.config import STRATIFIED_SEED
from pilot.data_loader import Problem, stratified_sample


def _make_problems(
    n_physics: int = 40,
    n_chemistry: int = 30,
    n_biology: int = 30,
) -> list[Problem]:
    problems = []
    idx = 0
    for subject, count in [
        ("physics", n_physics),
        ("chemistry", n_chemistry),
        ("biology", n_biology),
    ]:
        for _ in range(count):
            problems.append(
                Problem(
                    id=f"gpqa_diamond_{idx:04d}",
                    subject=subject,
                    question=f"Question {idx}?",
                    option_a="A text",
                    option_b="B text",
                    option_c="C text",
                    option_d="D text",
                    ground_truth="A",
                )
            )
            idx += 1
    return problems


class TestStratifiedSample:
    def test_correct_size(self) -> None:
        problems = _make_problems()
        sample = stratified_sample(problems, n=50, seed=STRATIFIED_SEED)
        assert len(sample) == 50

    def test_same_seed_same_result(self) -> None:
        problems = _make_problems()
        s1 = stratified_sample(problems, n=50, seed=42)
        s2 = stratified_sample(problems, n=50, seed=42)
        assert [p.id for p in s1] == [p.id for p in s2]

    def test_different_seeds_different_results(self) -> None:
        problems = _make_problems()
        s1 = stratified_sample(problems, n=50, seed=42)
        s2 = stratified_sample(problems, n=50, seed=99)
        assert [p.id for p in s1] != [p.id for p in s2]

    def test_subject_proportions_within_one(self) -> None:
        """Each subject's sample count should be within ±1 of the balanced quota."""
        problems = _make_problems(n_physics=40, n_chemistry=40, n_biology=40)
        sample = stratified_sample(problems, n=50, seed=STRATIFIED_SEED)
        from collections import Counter

        counts = Counter(p.subject for p in sample)
        # 50 problems, 3 subjects → expect ~16-17 each
        for subject in ("physics", "chemistry", "biology"):
            assert (
                abs(counts[subject] - 50 // 3) <= 1
            ), f"Subject {subject!r}: {counts[subject]} (expected ~{50//3})"

    def test_all_subjects_represented(self) -> None:
        problems = _make_problems()
        sample = stratified_sample(problems, n=50, seed=STRATIFIED_SEED)
        subjects = {p.subject for p in sample}
        assert "physics" in subjects
        assert "chemistry" in subjects
        assert "biology" in subjects

    def test_all_required_fields_present(self) -> None:
        problems = _make_problems()
        sample = stratified_sample(problems, n=10, seed=STRATIFIED_SEED)
        for p in sample:
            assert p.id
            assert p.subject in ("physics", "chemistry", "biology")
            assert p.question
            assert p.option_a
            assert p.option_b
            assert p.option_c
            assert p.option_d
            assert p.ground_truth in "ABCD"

    def test_no_duplicates(self) -> None:
        problems = _make_problems()
        sample = stratified_sample(problems, n=50, seed=STRATIFIED_SEED)
        ids = [p.id for p in sample]
        assert len(ids) == len(set(ids))


class TestProblemDataclass:
    def test_immutable(self) -> None:
        import pytest

        p = Problem(
            id="test_001",
            subject="physics",
            question="Q?",
            option_a="a",
            option_b="b",
            option_c="c",
            option_d="d",
            ground_truth="A",
        )
        with pytest.raises((AttributeError, TypeError)):
            p.id = "changed"  # type: ignore[misc]
