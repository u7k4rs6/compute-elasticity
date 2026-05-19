"""Tests for pilot/diversity.py."""

from __future__ import annotations

import numpy as np
import pytest

from pilot.diversity import compute_diversity, truncate_trace


# ---------------------------------------------------------------------------
# Stub embedder for tests — no real model needed
# ---------------------------------------------------------------------------
class _FixedEmbedder:
    """Returns pre-specified embeddings for each input text."""

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self._mapping = mapping

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vecs = np.array([self._mapping[t] for t in texts], dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms
        return vecs


class _RandomEmbedder:
    """Returns random unit-normalised embeddings (seeded)."""

    def __init__(self, dim: int = 64, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._dim = dim

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vecs = self._rng.standard_normal((len(texts), self._dim))
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms
        return vecs


# ---------------------------------------------------------------------------
# compute_diversity
# ---------------------------------------------------------------------------
class TestComputeDiversity:
    def test_identical_traces_zero_distance(self) -> None:
        trace = "The answer is because of this reasoning."
        e = _FixedEmbedder({trace: np.array([1.0, 0.0, 0.0])})
        dist = compute_diversity([trace, trace, trace], e)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_traces_distance_one(self) -> None:
        e = _FixedEmbedder(
            {
                "trace_a": np.array([1.0, 0.0]),
                "trace_b": np.array([0.0, 1.0]),
            }
        )
        dist = compute_diversity(["trace_a", "trace_b"], e)
        assert dist == pytest.approx(1.0, abs=1e-6)

    def test_singleton_returns_zero(self) -> None:
        e = _FixedEmbedder({"only": np.array([1.0, 0.0])})
        assert compute_diversity(["only"], e) == 0.0

    def test_empty_list_returns_zero(self) -> None:
        e = _RandomEmbedder()
        assert compute_diversity([], e) == 0.0

    def test_random_uncorrelated_traces_above_threshold(self) -> None:
        rng = np.random.default_rng(7)
        traces = [f"trace_{i}" for i in range(8)]
        dim = 64
        vecs = rng.standard_normal((8, dim))
        mapping = {t: v for t, v in zip(traces, vecs)}
        e = _FixedEmbedder(mapping)
        dist = compute_diversity(traces, e)
        assert dist > 0.3

    def test_returns_float(self) -> None:
        e = _RandomEmbedder()
        result = compute_diversity(["a", "b"], e)
        assert isinstance(result, float)

    def test_diversity_in_zero_one_range(self) -> None:
        e = _RandomEmbedder(seed=99)
        traces = [f"t{i}" for i in range(6)]
        dist = compute_diversity(traces, e)
        assert 0.0 <= dist <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# truncate_trace
# ---------------------------------------------------------------------------
class TestTruncateTrace:
    def test_strips_answer_line(self) -> None:
        trace = "Some reasoning.\nAnswer: B"
        truncated, _ = truncate_trace(trace)
        assert "Answer: B" not in truncated

    def test_strips_lowercase_answer_line(self) -> None:
        trace = "Reasoning.\nanswer: a"
        truncated, _ = truncate_trace(trace)
        assert "answer: a" not in truncated

    def test_short_trace_not_truncated(self) -> None:
        trace = "Short trace. Answer: A"
        truncated, was_truncated = truncate_trace(trace, max_tokens=512)
        assert not was_truncated

    def test_long_trace_truncated(self) -> None:
        long_text = " ".join(["word"] * 600)
        truncated, was_truncated = truncate_trace(long_text, max_tokens=100)
        assert was_truncated
        assert len(truncated.split()) <= 100

    def test_truncation_does_not_split_mid_word(self) -> None:
        words = [f"word{i}" for i in range(600)]
        trace = " ".join(words)
        truncated, _ = truncate_trace(trace, max_tokens=100)
        for word in truncated.split():
            assert word in words

    def test_returns_tuple(self) -> None:
        result = truncate_trace("Some text.")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], bool)
