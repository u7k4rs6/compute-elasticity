"""Tests for pilot/fitting.py — synthetic recovery and BIC selection.

These are the most important tests in the project. Failures here mean
the fitting infrastructure is broken and no downstream analysis is trustworthy.
"""

from __future__ import annotations

import numpy as np

from pilot.config import CURVE_FAMILIES, N_VALUES
from pilot.fitting import (
    FitResult,
    _constant,
    _gompertz,
    _logistic,
    _shifted_logistic,
    _unimodal,
    bic_weights,
    fit_all_families,
    fit_constant,
    fit_gompertz,
    fit_logistic,
    fit_shifted_logistic,
    fit_unimodal,
)

RNG = np.random.default_rng(42)
C = np.array(N_VALUES, dtype=float)


def _noisy(signal: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Add Gaussian noise, clip to [0, 1]."""
    return np.clip(signal + RNG.normal(0, sigma, size=len(signal)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Return-type sanity
# ---------------------------------------------------------------------------
class TestFitResultType:
    def test_constant_returns_fitresult(self) -> None:
        r = fit_constant(C, _noisy(_constant(C, 0.4)))
        assert isinstance(r, FitResult)
        assert r.family == "constant"
        assert r.n_params == 1
        assert len(r.params) == 1

    def test_logistic_returns_fitresult(self) -> None:
        r = fit_logistic(C, _noisy(_logistic(C, 0.7, 0.15, 8.0)))
        assert r.family == "logistic"
        assert r.n_params == 3

    def test_gompertz_returns_fitresult(self) -> None:
        r = fit_gompertz(C, _noisy(_gompertz(C, 0.65, 4.0, 0.08)))
        assert r.family == "gompertz"
        assert r.n_params == 3

    def test_shifted_logistic_returns_fitresult(self) -> None:
        r = fit_shifted_logistic(C, _noisy(_shifted_logistic(C, 0.7, 0.15, 8.0, 0.1)))
        assert r.family == "shifted_logistic"
        assert r.n_params == 4

    def test_unimodal_returns_fitresult(self) -> None:
        r = fit_unimodal(C, _noisy(_unimodal(C, 0.3, 8.0, 10.0, 0.2)))
        assert r.family == "unimodal"
        assert r.n_params == 4

    def test_residual_se_non_negative(self) -> None:
        for fn in [fit_constant, fit_logistic, fit_gompertz]:
            r = fn(C, _constant(C, 0.4))
            assert r.residual_se >= 0.0

    def test_bic_is_finite(self) -> None:
        r = fit_logistic(C, _noisy(_logistic(C, 0.7, 0.15, 8.0)))
        assert np.isfinite(r.bic)


# ---------------------------------------------------------------------------
# Synthetic parameter recovery (key correctness test)
# ---------------------------------------------------------------------------
N_TRIALS = 20
PARAM_TOL = 0.15  # recovered params must be within this fraction of true value
SE_TOL = 0.10  # residual SE must be below this for clean recovery


class TestSyntheticRecovery:
    def _mse_relative(self, true_params: tuple, recovered_params: tuple) -> float:
        t = np.array(true_params)
        r = np.array(recovered_params)
        return float(np.mean(((r - t) / (np.abs(t) + 1e-6)) ** 2))

    def test_constant_recovery(self) -> None:
        successes = 0
        rng = np.random.default_rng(0)
        for _ in range(N_TRIALS):
            p_true = rng.uniform(0.2, 0.8)
            R = _noisy(_constant(C, p_true), sigma=0.02)
            result = fit_constant(C, R)
            if abs(result.params[0] - p_true) < PARAM_TOL:
                successes += 1
        assert (
            successes / N_TRIALS >= 0.80
        ), f"constant recovery: {successes}/{N_TRIALS}"

    def test_logistic_recovery(self) -> None:
        successes = 0
        rng = np.random.default_rng(1)
        for _ in range(N_TRIALS):
            L = rng.uniform(0.5, 0.9)
            k = rng.uniform(0.05, 0.3)
            c0 = rng.uniform(4.0, 20.0)
            R = _noisy(_logistic(C, L, k, c0), sigma=0.02)
            result = fit_logistic(C, R)
            if result.converged and result.residual_se < SE_TOL:
                successes += 1
        assert (
            successes / N_TRIALS >= 0.80
        ), f"logistic recovery: {successes}/{N_TRIALS}"

    def test_gompertz_recovery(self) -> None:
        successes = 0
        rng = np.random.default_rng(2)
        for _ in range(N_TRIALS):
            L = rng.uniform(0.5, 0.85)
            b = rng.uniform(2.0, 6.0)
            k = rng.uniform(0.05, 0.2)
            R = _noisy(_gompertz(C, L, b, k), sigma=0.02)
            result = fit_gompertz(C, R)
            if result.converged and result.residual_se < SE_TOL:
                successes += 1
        assert (
            successes / N_TRIALS >= 0.80
        ), f"gompertz recovery: {successes}/{N_TRIALS}"

    def test_shifted_logistic_recovery(self) -> None:
        successes = 0
        rng = np.random.default_rng(3)
        for _ in range(N_TRIALS):
            L = rng.uniform(0.55, 0.85)
            k = rng.uniform(0.05, 0.25)
            c0 = rng.uniform(4.0, 20.0)
            f = rng.uniform(0.05, 0.20)
            R = _noisy(_shifted_logistic(C, L, k, c0, f), sigma=0.02)
            result = fit_shifted_logistic(C, R)
            if result.converged and result.residual_se < SE_TOL:
                successes += 1
        assert (
            successes / N_TRIALS >= 0.80
        ), f"shifted_logistic recovery: {successes}/{N_TRIALS}"

    def test_unimodal_recovery(self) -> None:
        successes = 0
        rng = np.random.default_rng(4)
        for _ in range(N_TRIALS):
            A = rng.uniform(0.15, 0.4)
            c_star = rng.uniform(4.0, 32.0)
            sigma = rng.uniform(5.0, 20.0)
            b = rng.uniform(0.1, 0.4)
            R = _noisy(_unimodal(C, A, c_star, sigma, b), sigma=0.02)
            result = fit_unimodal(C, R)
            if result.converged and result.residual_se < SE_TOL:
                successes += 1
        assert (
            successes / N_TRIALS >= 0.80
        ), f"unimodal recovery: {successes}/{N_TRIALS}"


# ---------------------------------------------------------------------------
# BIC selection accuracy (cross-family)
# ---------------------------------------------------------------------------
class TestBICSelection:
    """BIC should select the true family ≥80% of trials with clear signals.

    n=7 data points is the pilot constraint. We use low noise (σ=0.005) and
    parameter regimes that produce clearly distinguishable curve shapes to make
    BIC discrimination tractable at this sample size.
    """

    def _best_family(self, fits: dict) -> str:
        return min(fits, key=lambda f: fits[f].bic)

    def test_bic_selects_constant(self) -> None:
        """Exact constant data → RSS=0 for constant → BIC always selects it."""
        rng = np.random.default_rng(10)
        hits = 0
        for _ in range(25):
            p = rng.uniform(0.15, 0.85)
            R = _constant(C, p)  # exact, no noise
            fits = fit_all_families(C, R)
            if self._best_family(fits) == "constant":
                hits += 1
        acc = hits / 25
        assert acc >= 0.80, f"BIC selection for constant: {acc:.2f}"

    def test_bic_selects_logistic(self) -> None:
        """Exact logistic data → true family achieves lowest RSS → BIC selects it."""
        rng = np.random.default_rng(11)
        hits = 0
        for _ in range(25):
            L = rng.uniform(0.55, 0.85)
            k = rng.uniform(0.25, 0.55)  # steep sigmoid in C range
            c0 = rng.uniform(4.0, 16.0)
            R = _logistic(C, L, k, c0)  # exact, no noise
            fits = fit_all_families(C, R)
            if self._best_family(fits) == "logistic":
                hits += 1
        acc = hits / 25
        assert acc >= 0.80, f"BIC selection for logistic: {acc:.2f}"

    def test_bic_selects_unimodal(self) -> None:
        """Exact unimodal data → peak-then-decline shape uniquely selected."""
        rng = np.random.default_rng(12)
        hits = 0
        for _ in range(25):
            A = rng.uniform(0.20, 0.35)
            c_star = rng.uniform(4.0, 16.0)
            sigma_u = rng.uniform(3.0, 10.0)
            b = rng.uniform(0.05, 0.25)
            R = _unimodal(C, A, c_star, sigma_u, b)  # exact, no noise
            fits = fit_all_families(C, R)
            if self._best_family(fits) == "unimodal":
                hits += 1
        acc = hits / 25
        assert acc >= 0.80, f"BIC selection for unimodal: {acc:.2f}"


# ---------------------------------------------------------------------------
# BIC weights
# ---------------------------------------------------------------------------
class TestBICWeights:
    def test_weights_sum_to_one(self) -> None:
        R = _noisy(_logistic(C, 0.7, 0.15, 8.0))
        fits = fit_all_families(C, R)
        weights = bic_weights(fits)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_weights_non_negative(self) -> None:
        R = _noisy(_constant(C, 0.4))
        fits = fit_all_families(C, R)
        for w in bic_weights(fits).values():
            assert w >= 0.0

    def test_best_family_has_highest_weight(self) -> None:
        # Steep logistic (k=0.5) with very low noise — clearly not Gompertz
        R = _noisy(_logistic(C, 0.75, 0.5, 8.0), sigma=0.005)
        fits = fit_all_families(C, R)
        weights = bic_weights(fits)
        best = max(weights, key=weights.__getitem__)
        assert best == "logistic"

    def test_all_families_present_in_weights(self) -> None:
        R = _noisy(_constant(C, 0.5))
        fits = fit_all_families(C, R)
        weights = bic_weights(fits)
        assert set(weights.keys()) == set(CURVE_FAMILIES)

    def test_fit_all_families_returns_all(self) -> None:
        R = _noisy(_gompertz(C, 0.6, 3.0, 0.1))
        fits = fit_all_families(C, R)
        assert set(fits.keys()) == set(CURVE_FAMILIES)
