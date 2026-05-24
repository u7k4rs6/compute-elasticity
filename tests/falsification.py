"""H1–H6 falsification tests.

Reads outputs/hypothesis_results.json produced by scripts/run_analysis.py
and asserts each hypothesis verdict against the pre-registered thresholds.

A pytest failure = hypothesis falsified (or analysis not yet run).
H4 is deferred; its test asserts the DEFERRED marker is present.

Usage:
    pytest tests/falsification.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_RESULTS_PATH = Path("outputs/hypothesis_results.json")


class TestHypotheses:
    """Confirmatory H1–H6 falsification suite (Phase 9)."""

    @pytest.fixture(scope="class")
    def results(self) -> dict:
        """Load hypothesis_results.json produced by run_analysis.py."""
        if not _RESULTS_PATH.exists():
            pytest.skip(
                f"{_RESULTS_PATH} not found — run scripts/run_analysis.py first"
            )
        with _RESULTS_PATH.open() as f:
            return json.load(f)

    def test_h1_fittable_above_noise(self, results: dict) -> None:
        """H1: Median residual SE < 0.10 across 47 main-pilot problems."""
        h1 = results["H1"]
        measured = h1["measured"]
        verdict = h1["verdict"]
        assert verdict == "PASS", (
            f"H1 FAILED: median residual SE = {measured:.4f}, "
            f"threshold = {h1['threshold_pass']} (verdict: {verdict})"
        )

    def test_h2_mode_diversity(self, results: dict) -> None:
        """H2: Multiple curve families win (criterion a OR b)."""
        h2 = results["H2"]
        verdict = h2["overall_verdict"]
        ca = h2["criterion_a"]
        cb = h2["criterion_b"]
        assert verdict == "PASS", (
            f"H2 FAILED: criterion_a={ca['verdict']} "
            f"({ca['n_families']} families >= 0.10), "
            f"criterion_b={cb['verdict']} "
            f"({cb['fraction_2plus_close']:.1%} problems with >=2 close families)"
        )

    def test_h3_diversity_predicts_elasticity(self, results: dict) -> None:
        """H3: AUC(diversity) >= 0.60 AND delta-AUC(diversity-entropy) >= 0.03."""
        h3 = results["H3"]
        verdict = h3["verdict"]
        assert verdict == "PASS", (
            f"H3 {verdict}: AUC(diversity)={h3['auc_diversity']:.3f}, "
            f"AUC(entropy)={h3['auc_entropy']:.3f}, "
            f"delta={h3['delta_auc_entropy']:+.3f} "
            f"(need AUC>=0.60 and delta>=0.03)"
        )

    def test_h4_deferred(self, results: dict) -> None:
        """H4: Domain distribution shift — deferred to full study."""
        assert (
            results["H4"]["verdict"] == "DEFERRED"
        ), "H4 verdict should be DEFERRED (domain test not run in pilot)"

    def test_h5_nontrivial_unimodal_subset(self, results: dict) -> None:
        """H5: Non-trivial unimodal subset (>= 5% of problems)."""
        h5 = results["H5"]
        verdict = h5["verdict"]
        assert verdict == "PASS", (
            f"H5 FAILED: {h5['unimodal_win_count']}/{h5['n_total']} unimodal wins "
            f"({h5['unimodal_fraction']:.1%}), threshold >= 5%"
        )

    def test_h6_curve_params_stable_across_temperature(self, results: dict) -> None:
        """H6: Bootstrap-CI overlap rate >= 0.70 across T in {0.3, 0.7, 1.0}."""
        h6 = results["H6"]
        verdict = h6["verdict"]
        assert verdict == "PASS", (
            f"H6 {verdict}: mean CI overlap rate = {h6['mean_ci_overlap_rate']:.3f}, "
            f"threshold >= {h6['threshold_pass']} "
            f"(per-pair: {h6['per_pair_overlap']})"
        )
