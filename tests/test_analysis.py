"""Tests for pilot/analysis.py."""

from __future__ import annotations

import numpy as np
import pytest

from pilot.analysis import (
    block_bootstrap,
    compute_auc,
    elasticity_at,
    fitted_elasticity_summary,
)
from pilot.config import N_VALUES
from pilot.fitting import (
    FitResult,
    fit_constant,
    fit_logistic,
)

C = np.array(N_VALUES, dtype=float)


def _dummy_fit(family: str, params: tuple, n_params: int) -> FitResult:
    return FitResult(
        family=family,
        params=params,
        cov=tuple(tuple(0.0 for _ in range(n_params)) for _ in range(n_params)),
        bic=0.0,
        residual_se=0.0,
        converged=True,
        n_params=n_params,
    )


# ---------------------------------------------------------------------------
# block_bootstrap
# ---------------------------------------------------------------------------
class TestBlockBootstrap:
    def test_mean_stat_correct(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        point, lo, hi = block_bootstrap(np.mean, data, n_resamples=2000, ci=0.95)
        assert point == pytest.approx(3.0, abs=1e-9)

    def test_ci_contains_true_mean_for_gaussian(self) -> None:
        rng = np.random.default_rng(42)
        true_mean = 5.0
        data = rng.normal(true_mean, 1.0, size=100)
        point, lo, hi = block_bootstrap(np.mean, data, n_resamples=2000, ci=0.95)
        assert lo < true_mean < hi

    def test_returns_tuple_of_three(self) -> None:
        result = block_bootstrap(np.mean, np.arange(10.0), n_resamples=100)
        assert len(result) == 3

    def test_ci_ordered(self) -> None:
        data = np.random.default_rng(0).normal(0, 1, 50)
        point, lo, hi = block_bootstrap(np.mean, data, n_resamples=500)
        assert lo <= point <= hi

    def test_coverage_rate_near_nominal(self) -> None:
        rng = np.random.default_rng(7)
        n_experiments = 200
        true_val = 0.0
        n_covered = 0
        for _ in range(n_experiments):
            data = rng.normal(true_val, 1.0, size=30)
            _, lo, hi = block_bootstrap(np.mean, data, n_resamples=500, ci=0.95)
            if lo <= true_val <= hi:
                n_covered += 1
        coverage = n_covered / n_experiments
        assert 0.85 <= coverage <= 0.99, f"Coverage {coverage:.2f} outside [0.85, 0.99]"


# ---------------------------------------------------------------------------
# compute_auc
# ---------------------------------------------------------------------------
class TestComputeAUC:
    def test_perfect_separation_auc_one(self) -> None:
        scores = np.array([0.1, 0.2, 0.9, 0.95])
        labels = np.array([0, 0, 1, 1])
        assert compute_auc(scores, labels) == pytest.approx(1.0)

    def test_reversed_scores_auc_zero(self) -> None:
        scores = np.array([0.9, 0.8, 0.1, 0.05])
        labels = np.array([0, 0, 1, 1])
        assert compute_auc(scores, labels) == pytest.approx(0.0)

    def test_random_scores_near_half(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 200)
        labels = rng.integers(0, 2, 200)
        auc = compute_auc(scores, labels)
        assert 0.35 <= auc <= 0.65

    def test_degenerate_no_positives(self) -> None:
        assert compute_auc(np.array([0.5, 0.6]), np.array([0, 0])) == pytest.approx(0.5)

    def test_degenerate_no_negatives(self) -> None:
        assert compute_auc(np.array([0.5, 0.6]), np.array([1, 1])) == pytest.approx(0.5)

    def test_returns_float(self) -> None:
        result = compute_auc(np.array([0.5]), np.array([1]))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# elasticity_at
# ---------------------------------------------------------------------------
class TestElasticityAt:
    def test_constant_zero_elasticity(self) -> None:
        fit = _dummy_fit("constant", (0.5,), 1)
        assert elasticity_at(8.0, fit) == pytest.approx(0.0)

    def test_logistic_elasticity_positive_near_inflection(self) -> None:
        L, k, c0 = 0.7, 0.15, 8.0
        fit = _dummy_fit("logistic", (L, k, c0), 3)
        elast = elasticity_at(c0, fit)
        assert elast > 0.0

    def test_logistic_elasticity_symmetric(self) -> None:
        L, k, c0 = 0.7, 0.15, 16.0
        fit = _dummy_fit("logistic", (L, k, c0), 3)
        e_left = elasticity_at(c0 - 5, fit)
        e_right = elasticity_at(c0 + 5, fit)
        assert abs(e_left - e_right) < 1e-6

    def test_gompertz_elasticity_positive(self) -> None:
        fit = _dummy_fit("gompertz", (0.7, 3.0, 0.1), 3)
        assert elasticity_at(8.0, fit) > 0.0

    def test_shifted_logistic_elasticity_positive(self) -> None:
        fit = _dummy_fit("shifted_logistic", (0.7, 0.15, 8.0, 0.1), 4)
        assert elasticity_at(8.0, fit) > 0.0

    def test_unimodal_negative_after_peak(self) -> None:
        c_star = 8.0
        fit = _dummy_fit("unimodal", (0.3, c_star, 10.0, 0.2), 4)
        assert elasticity_at(c_star + 20.0, fit) < 0.0

    def test_unimodal_zero_at_peak(self) -> None:
        c_star = 8.0
        fit = _dummy_fit("unimodal", (0.3, c_star, 10.0, 0.2), 4)
        assert abs(elasticity_at(c_star, fit)) < 1e-9

    def test_unknown_family_raises(self) -> None:
        fit = _dummy_fit("nonexistent", (0.5,), 1)
        with pytest.raises(ValueError):
            elasticity_at(8.0, fit)


# ---------------------------------------------------------------------------
# fitted_elasticity_summary
# ---------------------------------------------------------------------------
class TestFittedElasticitySummary:
    def test_constant_family_zero(self) -> None:
        R = np.full_like(C, 0.4)
        fit = fit_constant(C, R)
        summary = fitted_elasticity_summary(fit, c_range=(8.0, 64.0))
        assert summary == pytest.approx(0.0, abs=1e-9)

    def test_logistic_positive_summary(self) -> None:
        from pilot.fitting import _logistic

        R = _logistic(C, 0.7, 0.15, 8.0)
        fit = fit_logistic(C, R)
        summary = fitted_elasticity_summary(fit, c_range=(8.0, 64.0))
        assert summary > 0.0

    def test_returns_float(self) -> None:
        R = np.full_like(C, 0.5)
        fit = fit_constant(C, R)
        result = fitted_elasticity_summary(fit)
        assert isinstance(result, float)
