"""H1-H6 falsification tests.

Reads outputs/fits/ and outputs/diversity/ to evaluate each hypothesis
against HYPOTHESIS_THRESHOLDS. Skipped in Phase 1 (no pilot data yet);
runs for real in Phase 9.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="no pilot data yet — run after Phase 8")
def test_h1_fittable_above_noise() -> None:
    """H1: R(c) curves fittable above noise (median residual SE < 0.10)."""


@pytest.mark.skip(reason="no pilot data yet — run after Phase 8")
def test_h2_mode_diversity() -> None:
    """H2: Multiple curve families win (≥2 families with mean BIC weight ≥0.10)."""


@pytest.mark.skip(reason="no pilot data yet — run after Phase 8")
def test_h3_embedding_diversity_predicts_elasticity() -> None:
    """H3: Embedding diversity (N=4) > entropy (N=1) for elasticity prediction."""


@pytest.mark.skip(reason="no pilot data yet — run after Phase 8")
def test_h5_nontrivial_unimodal_subset() -> None:
    """H5: Non-trivial unimodal subset (≥5% of problems where unimodal wins)."""


@pytest.mark.skip(reason="no pilot data yet — run after Phase 8")
def test_h6_curve_params_stable_across_temperature() -> None:
    """H6: Curve params stable across T ∈ {0.3, 0.7, 1.0} (≥70% CI overlap)."""
