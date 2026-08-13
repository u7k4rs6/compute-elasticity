"""PH1-PH4 falsification tests for the self-consistency backfire study.

Reads outputs/gate_model2/confirmatory_results.json produced by
scripts/run_phase13_analysis.py and asserts each pre-registered hypothesis on
the 151-problem confirmatory split, for both models. This file is the source
for Table 1 of the paper.

Hypotheses and thresholds are locked at git tag backfire-prereg-v1.0
(preregistration_backfire.md) and must not be changed here. Each test asserts
both the measured value against its threshold and the stored verdict, so a
verdict that disagrees with its own numbers fails rather than passing quietly.

Scope: this suite asserts stored values against pre-registered thresholds. It
does not recompute the analysis, so it guards against regeneration drift and
makes the confirmatory claim runnable; it does not independently verify that
the analysis is correct. That is the same standard as tests/falsification.py.

A pytest failure = a pre-registered hypothesis is falsified.

Usage:
    pytest tests/falsification_backfire.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_RESULTS_PATH = Path("outputs/gate_model2/confirmatory_results.json")

# Locked at tag backfire-prereg-v1.0. Do not edit.
_PH1_MIN_BACKFIRE_RATE = 0.33
_PH2_MAX_CAPTURE = 0.10
_PH3_MAX_TOP_BIN_ACC = 0.70
_PH4_MAX_CAPTURE = 0.10

_MODELS: tuple[tuple[str, str], ...] = (
    ("model_1", "Qwen2.5-7B"),
    ("model_2", "Llama-3-8B"),
)


class TestBackfireHypotheses:
    """Confirmatory PH1-PH4 falsification suite (n=151, both models)."""

    @pytest.fixture(scope="class")
    def results(self) -> dict:
        """Load confirmatory_results.json, failing loudly if it is absent.

        Deliberately fails rather than skips: a missing results file means the
        confirmatory analysis has not been run, and a skipped test in a
        pre-registered suite is too easily read as a pass.

        Returns:
            The parsed confirmatory results.
        """
        if not _RESULTS_PATH.exists():
            pytest.fail(
                f"{_RESULTS_PATH} not found. Run scripts/run_phase13_analysis.py "
                "to produce it. Refusing to skip: an absent results file must "
                "not be mistaken for a passing hypothesis."
            )
        with _RESULTS_PATH.open() as f:
            return json.load(f)

    def test_ph1_backfire_prevalence(self, results: dict) -> None:
        """PH1: backfire rate >= 33% of confirmatory problems for BOTH models."""
        ph1 = results["PH1"]
        assert ph1["threshold"] == _PH1_MIN_BACKFIRE_RATE, (
            f"PH1 threshold drifted from the pre-registered value: "
            f"{ph1['threshold']} != {_PH1_MIN_BACKFIRE_RATE}"
        )
        for key, name in _MODELS:
            rate = ph1[key]["value"]
            assert rate >= _PH1_MIN_BACKFIRE_RATE, (
                f"PH1 FALSIFIED for {name}: backfire rate = {rate:.4f}, "
                f"threshold >= {_PH1_MIN_BACKFIRE_RATE}"
            )
        assert ph1["pass"] is True, (
            f"PH1 stored verdict is {ph1['pass']} despite both rates clearing "
            f"the threshold: {[ph1[k]['value'] for k, _ in _MODELS]}"
        )

    def test_ph2_agreement_gate_fails(self, results: dict) -> None:
        """PH2: agreement gate (k=8, tau=0.75) captures <= 10% of oracle ceiling."""
        ph2 = results["PH2"]
        assert ph2["threshold"] == _PH2_MAX_CAPTURE, (
            f"PH2 threshold drifted from the pre-registered value: "
            f"{ph2['threshold']} != {_PH2_MAX_CAPTURE}"
        )
        for key, name in _MODELS:
            capture = ph2[key]["value"]
            assert capture <= _PH2_MAX_CAPTURE, (
                f"PH2 FALSIFIED for {name}: ceiling captured = {capture:.4f}, "
                f"threshold <= {_PH2_MAX_CAPTURE}"
            )
        assert ph2["pass"] is True, (
            f"PH2 stored verdict is {ph2['pass']} despite both captures clearing "
            f"the threshold: {[ph2[k]['value'] for k, _ in _MODELS]}"
        )

    def test_ph3_confidence_does_not_track_correctness(self, results: dict) -> None:
        """PH3: top-confidence bin accuracy <= 70% for BOTH models."""
        ph3 = results["PH3"]
        assert ph3["threshold"] == _PH3_MAX_TOP_BIN_ACC, (
            f"PH3 threshold drifted from the pre-registered value: "
            f"{ph3['threshold']} != {_PH3_MAX_TOP_BIN_ACC}"
        )
        for key, name in _MODELS:
            acc = ph3[key]["value"]
            assert acc <= _PH3_MAX_TOP_BIN_ACC, (
                f"PH3 FALSIFIED for {name}: top-agreement-bin accuracy = "
                f"{acc:.4f}, threshold <= {_PH3_MAX_TOP_BIN_ACC}"
            )
        assert ph3["pass"] is True, (
            f"PH3 stored verdict is {ph3['pass']} despite both bins clearing "
            f"the threshold: {[ph3[k]['value'] for k, _ in _MODELS]}"
        )

    def test_ph4_entropy_gate_fails(self, results: dict) -> None:
        """PH4: entropy gate captures <= 10% of oracle ceiling for BOTH models.

        PH4 is contingent on logprob availability. The pre-registration states
        that if logprobs are unavailable for a model, PH4 is not evaluated for
        it and that fact is reported, so this test asserts availability
        explicitly rather than letting an unevaluated model pass silently.
        """
        ph4 = results["PH4"]
        assert ph4["threshold"] == _PH4_MAX_CAPTURE, (
            f"PH4 threshold drifted from the pre-registered value: "
            f"{ph4['threshold']} != {_PH4_MAX_CAPTURE}"
        )
        for key, name in _MODELS:
            model = ph4[key]
            assert model["logprobs_available"] is True, (
                f"PH4 not evaluated for {name}: logprobs unavailable. The "
                "pre-registration requires this be reported, not passed over."
            )
            capture = model["best_ceiling_captured"]
            assert capture <= _PH4_MAX_CAPTURE, (
                f"PH4 FALSIFIED for {name}: ceiling captured = {capture:.4f}, "
                f"threshold <= {_PH4_MAX_CAPTURE}"
            )
        assert ph4["pass"] is True, (
            f"PH4 stored verdict is {ph4['pass']} despite both captures clearing "
            f"the threshold: "
            f"{[ph4[k]['best_ceiling_captured'] for k, _ in _MODELS]}"
        )
