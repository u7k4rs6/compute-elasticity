"""Tests for pilot/validation.py — planted anomalies must all be caught."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pilot.config import SCHEMA_VERSION
from pilot.validation import ValidationReport, scan_outputs, validate_sample


def _valid_sample(**overrides) -> dict:
    """Build a minimal valid sample dict."""
    base = {
        "schema_version": SCHEMA_VERSION,
        "problem_id": "gpqa_diamond_0001",
        "subject": "physics",
        "ground_truth": "A",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "provider": "together_ai",
        "temperature": 0.7,
        "sample_idx": 0,
        "n_total_in_batch": 64,
        "seed_hex": "a1b2c3d4",
        "prompt_template_hash": "sha256:abc123",
        "full_response": "Thinking... Answer: A",
        "extracted_answer": "A",
        "extraction_pass": 1,
        "correct": True,
        "input_tokens": 150,
        "output_tokens": 300,
        "latency_ms": 5000,
        "timestamp": "2026-05-20T14:30:00Z",
        "api_metadata": {},
    }
    base.update(overrides)
    return base


class TestValidateSample:
    def test_valid_sample_no_issues(self) -> None:
        assert validate_sample(_valid_sample()) == []

    def test_missing_required_field(self) -> None:
        sample = _valid_sample()
        del sample["problem_id"]
        issues = validate_sample(sample)
        assert any("missing_field:problem_id" in i for i in issues)

    def test_invalid_subject(self) -> None:
        issues = validate_sample(_valid_sample(subject="mathematics"))
        assert any("invalid_subject" in i for i in issues)

    def test_invalid_ground_truth(self) -> None:
        issues = validate_sample(_valid_sample(ground_truth="E"))
        assert any("invalid_ground_truth" in i for i in issues)

    def test_invalid_extracted_answer(self) -> None:
        issues = validate_sample(_valid_sample(extracted_answer="X"))
        assert any("invalid_extracted_answer" in i for i in issues)

    def test_correct_not_bool(self) -> None:
        issues = validate_sample(_valid_sample(correct=1))
        assert any("correct_not_bool" in i for i in issues)

    def test_negative_input_tokens(self) -> None:
        issues = validate_sample(_valid_sample(input_tokens=-5))
        assert any("invalid_input_tokens" in i for i in issues)

    def test_negative_output_tokens(self) -> None:
        issues = validate_sample(_valid_sample(output_tokens=-1))
        assert any("invalid_output_tokens" in i for i in issues)

    def test_negative_latency(self) -> None:
        issues = validate_sample(_valid_sample(latency_ms=-100))
        assert any("invalid_latency_ms" in i for i in issues)

    def test_suspicious_latency_flagged(self) -> None:
        issues = validate_sample(_valid_sample(latency_ms=400_000))
        assert any("suspicious_latency_ms" in i for i in issues)

    def test_wrong_schema_version(self) -> None:
        issues = validate_sample(_valid_sample(schema_version="v1.0"))
        assert any("schema_version" in i for i in issues)

    def test_multiple_missing_fields(self) -> None:
        sample = _valid_sample()
        del sample["problem_id"]
        del sample["ground_truth"]
        del sample["correct"]
        issues = validate_sample(sample)
        assert len(issues) >= 3

    def test_returns_list(self) -> None:
        assert isinstance(validate_sample(_valid_sample()), list)


class TestScanOutputs:
    def _write_jsonl(self, directory: Path, filename: str, samples: list[dict]) -> Path:
        path = directory / filename
        path.write_text("\n".join(json.dumps(s) for s in samples))
        return path

    def test_clean_directory_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            self._write_jsonl(d, "problem_001.jsonl", [_valid_sample()])
            report = scan_outputs(d)
            assert report.is_clean
            assert report.n_samples == 1

    def test_detects_missing_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            bad = _valid_sample()
            del bad["problem_id"]
            self._write_jsonl(d, "bad.jsonl", [bad])
            report = scan_outputs(d)
            assert report.n_invalid > 0

    def test_detects_duplicate_seed_hex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            s1 = _valid_sample(seed_hex="dup_seed")
            s2 = _valid_sample(seed_hex="dup_seed")
            self._write_jsonl(d, "dup.jsonl", [s1, s2])
            report = scan_outputs(d)
            assert len(report.duplicate_seeds) > 0

    def test_detects_negative_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            self._write_jsonl(d, "neg.jsonl", [_valid_sample(input_tokens=-1)])
            report = scan_outputs(d)
            assert report.n_invalid > 0

    def test_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = scan_outputs(Path(tmpdir))
            assert report.n_files == 0
            assert report.is_clean

    def test_counts_samples_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            samples = [
                _valid_sample(sample_idx=i, seed_hex=f"seed_{i:04d}") for i in range(5)
            ]
            self._write_jsonl(d, "multi.jsonl", samples)
            report = scan_outputs(d)
            assert report.n_samples == 5

    def test_returns_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = scan_outputs(Path(tmpdir))
            assert isinstance(report, ValidationReport)
