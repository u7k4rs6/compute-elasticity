"""Statistical analysis utilities: bootstrap, AUC, elasticity computation."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from pilot.fitting import FitResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------
def block_bootstrap(
    stat_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a scalar statistic.

    Args:
        stat_fn: Function mapping a 1-D data array to a scalar.
        data: Observed data (1-D array).
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95).
        seed: RNG seed.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    point = stat_fn(data)
    boot_stats = np.array(
        [
            stat_fn(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_resamples)
        ]
    )
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# AUC (binary classification)
# ---------------------------------------------------------------------------
def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC of `scores` predicting binary `labels`.

    Uses the Wilcoxon-Mann-Whitney statistic formulation: no sklearn required.

    Args:
        scores: Continuous predictor values.
        labels: Binary labels (0 or 1).

    Returns:
        AUC in [0, 1]. Returns 0.5 on degenerate inputs.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_correct = sum(1 for p in pos for n in neg if p > n)
    n_tie = sum(1 for p in pos for n in neg if p == n)
    total = len(pos) * len(neg)
    return (n_correct + 0.5 * n_tie) / total


# ---------------------------------------------------------------------------
# Analytical elasticity: dR/dc per family
# ---------------------------------------------------------------------------
def elasticity_at(c: float, fit_result: FitResult) -> float:
    """Return dR/dc analytically at compute budget c.

    Uses closed-form derivatives for each curve family.
    """
    family = fit_result.family
    p = fit_result.params

    if family == "constant":
        return 0.0

    if family == "logistic":
        L, k, c0 = p[0], p[1], p[2]
        z = np.exp(-k * (c - c0))
        sig = 1.0 / (1.0 + z)
        return float(L * k * sig * (1.0 - sig))

    if family == "gompertz":
        L, b, k = p[0], p[1], p[2]
        exp_kc = np.exp(-k * c)
        return float(L * b * k * exp_kc * np.exp(-b * exp_kc))

    if family == "shifted_logistic":
        L, k, c0, f = p[0], p[1], p[2], p[3]
        z = np.exp(-k * (c - c0))
        sig = 1.0 / (1.0 + z)
        return float((L - f) * k * sig * (1.0 - sig))

    if family == "unimodal":
        A, c_star, sigma, _b = p[0], p[1], p[2], p[3]
        diff = c - c_star
        gauss = np.exp(-(diff**2) / (2.0 * sigma**2))
        return float(-A * diff / (sigma**2) * gauss)

    raise ValueError(f"Unknown family: {family!r}")


# ---------------------------------------------------------------------------
# Fitted elasticity summary over a compute range
# ---------------------------------------------------------------------------
def fitted_elasticity_summary(
    fit_result: FitResult,
    c_range: tuple[float, float] = (8.0, 64.0),
    n_points: int = 100,
) -> float:
    """Mean of |dR/dc| over c_range, sampled at n_points.

    Args:
        fit_result: Fit from pilot.fitting.
        c_range: (c_min, c_max) range for integration.
        n_points: Number of evaluation points.

    Returns:
        Mean absolute elasticity over the range.
    """
    c_vals = np.linspace(c_range[0], c_range[1], n_points)
    elasticities = np.array([elasticity_at(c, fit_result) for c in c_vals])
    return float(np.mean(np.abs(elasticities)))
