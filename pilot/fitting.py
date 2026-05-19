"""Curve-family fitting for R(c) = accuracy as a function of compute budget.

Five families are fit via MLE (scipy.optimize.curve_fit). BIC weights are
used for soft model selection. All families operate on c = number of samples N.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit

from pilot.config import CURVE_FAMILIES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FitResult:
    """Outcome of fitting one curve family to (c, R) data."""

    family: str
    params: tuple[float, ...]
    cov: tuple[tuple[float, ...], ...]
    bic: float
    residual_se: float  # RMS residual in accuracy units
    converged: bool
    n_params: int


# ---------------------------------------------------------------------------
# Curve functions
# ---------------------------------------------------------------------------
def _constant(c: np.ndarray, p: float) -> np.ndarray:
    return np.full_like(c, p, dtype=float)


def _logistic(c: np.ndarray, L: float, k: float, c0: float) -> np.ndarray:
    z = np.clip(-k * (c - c0), -500.0, 500.0)
    return L / (1.0 + np.exp(z))


def _gompertz(c: np.ndarray, L: float, b: float, k: float) -> np.ndarray:
    inner = np.clip(-k * c, -500.0, 500.0)
    return L * np.exp(-b * np.exp(inner))


def _shifted_logistic(
    c: np.ndarray, L: float, k: float, c0: float, f: float
) -> np.ndarray:
    z = np.clip(-k * (c - c0), -500.0, 500.0)
    return f + (L - f) / (1.0 + np.exp(z))


def _unimodal(
    c: np.ndarray, A: float, c_star: float, sigma: float, b: float
) -> np.ndarray:
    return A * np.exp(-((c - c_star) ** 2) / (2.0 * sigma**2)) + b


# ---------------------------------------------------------------------------
# BIC helper
# ---------------------------------------------------------------------------
def _compute_bic(k: int, n: int, rss: float) -> float:
    """Gaussian-MLE BIC: k·ln(n) + n·ln(2π·RSS/n) + n."""
    rss_safe = max(rss, 1e-12)
    return k * np.log(n) + n * np.log(2 * np.pi * rss_safe / n) + n


def _make_fit_result(
    family: str,
    fn: Callable,
    c: np.ndarray,
    R: np.ndarray,
    bounds: tuple,
    p0: list[float],
) -> FitResult:
    """Run curve_fit and package results into FitResult."""
    n = len(c)
    k = len(p0)
    converged = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                fn, c, R, p0=p0, bounds=bounds, maxfev=10_000, method="trf"
            )
        fitted = fn(c, *popt)
        rss = float(np.sum((R - fitted) ** 2))
    except (RuntimeError, ValueError) as exc:
        logger.debug("curve_fit failed for %s: %s", family, exc)
        converged = False
        popt = np.array(p0, dtype=float)
        fitted = fn(c, *popt)
        rss = float(np.sum((R - fitted) ** 2))
        pcov = np.full((k, k), np.inf)

    bic = _compute_bic(k=k, n=n, rss=rss)
    residual_se = float(np.sqrt(rss / n))
    params_t = tuple(float(x) for x in popt)
    cov_t = tuple(tuple(float(x) for x in row) for row in pcov)
    return FitResult(
        family=family,
        params=params_t,
        cov=cov_t,
        bic=bic,
        residual_se=residual_se,
        converged=converged,
        n_params=k,
    )


# ---------------------------------------------------------------------------
# Public fit functions
# ---------------------------------------------------------------------------
def fit_constant(c: np.ndarray, R: np.ndarray) -> FitResult:
    """Fit constant R(c) = p."""
    p0 = [float(np.mean(R))]
    bounds = ([0.0], [1.0])
    return _make_fit_result("constant", _constant, c, R, bounds, p0)


def fit_logistic(c: np.ndarray, R: np.ndarray) -> FitResult:
    """Fit logistic R(c) = L / (1 + exp(-k(c - c0)))."""
    p0 = [max(float(np.max(R)), 0.5), 0.1, float(np.median(c))]
    bounds = ([0.0, 1e-4, 0.0], [1.0, 10.0, 1e4])
    return _make_fit_result("logistic", _logistic, c, R, bounds, p0)


def fit_gompertz(c: np.ndarray, R: np.ndarray) -> FitResult:
    """Fit Gompertz R(c) = L · exp(-b · exp(-k·c))."""
    p0 = [max(float(np.max(R)), 0.5), 3.0, 0.1]
    bounds = ([0.0, 1e-4, 1e-4], [1.0, 100.0, 10.0])
    return _make_fit_result("gompertz", _gompertz, c, R, bounds, p0)


def fit_shifted_logistic(c: np.ndarray, R: np.ndarray) -> FitResult:
    """Fit shifted logistic R(c) = f + (L-f)/(1+exp(-k(c-c0))) with floor f."""
    p0 = [max(float(np.max(R)), 0.5), 0.1, float(np.median(c)), float(np.min(R))]
    bounds = ([0.0, 1e-4, 0.0, 0.0], [1.0, 10.0, 1e4, 0.25])
    return _make_fit_result("shifted_logistic", _shifted_logistic, c, R, bounds, p0)


def fit_unimodal(c: np.ndarray, R: np.ndarray) -> FitResult:
    """Fit unimodal R(c) = A·exp(-((c-c*)²/2σ²)) + b."""
    peak_idx = int(np.argmax(R))
    p0 = [
        max(float(R[peak_idx] - np.min(R)), 0.1),
        float(c[peak_idx]),
        max(float(np.std(c)), 1.0),
        float(np.min(R)),
    ]
    bounds = ([0.0, 0.0, 1e-4, 0.0], [1.0, 1e4, 1e4, 1.0])
    return _make_fit_result("unimodal", _unimodal, c, R, bounds, p0)


# ---------------------------------------------------------------------------
# Batch fit + BIC weights
# ---------------------------------------------------------------------------
def fit_all_families(c: np.ndarray, R: np.ndarray) -> dict[str, FitResult]:
    """Fit all 5 families; return dict keyed by family name."""
    fitters = {
        "constant": fit_constant,
        "logistic": fit_logistic,
        "gompertz": fit_gompertz,
        "shifted_logistic": fit_shifted_logistic,
        "unimodal": fit_unimodal,
    }
    results: dict[str, FitResult] = {}
    for name in CURVE_FAMILIES:
        results[name] = fitters[name](c, R)
    return results


def bic_weights(fit_results: dict[str, FitResult]) -> dict[str, float]:
    """Compute BIC weights: w_i = exp(-ΔBIC_i/2) / Σ_j exp(-ΔBIC_j/2)."""
    bics = np.array([fit_results[f].bic for f in CURVE_FAMILIES])
    delta = bics - bics.min()
    raw = np.exp(-delta / 2.0)
    total = raw.sum()
    if total == 0:
        weights = np.ones(len(CURVE_FAMILIES)) / len(CURVE_FAMILIES)
    else:
        weights = raw / total
    return {f: float(w) for f, w in zip(CURVE_FAMILIES, weights)}
