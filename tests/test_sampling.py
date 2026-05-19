"""Tests for pilot/sampling.py — mocked HTTP layer, no live API calls."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pilot.data_loader import Problem
from pilot.sampling import (
    APIError,
    Completion,
    ConfigError,
    RetryableAPIError,
    TogetherClient,
    _load_existing_seeds,
    _with_retry,
    sample_problem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_problem(idx: int = 0) -> Problem:
    return Problem(
        id=f"gpqa_diamond_{idx:04d}",
        subject="physics",
        question="What is 2+2?",
        option_a="3",
        option_b="4",
        option_c="5",
        option_d="6",
        ground_truth="B",
    )


def _make_completion(
    text: str = "Answer: B", provider: str = "together_ai"
) -> Completion:
    return Completion(
        text=text,
        input_tokens=100,
        output_tokens=50,
        latency_ms=1200.0,
        provider=provider,
        model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        raw={},
    )


class _MockClient:
    """Synchronous mock that returns a fixed Completion."""

    def __init__(self, response: Completion | None = None, raises=None) -> None:
        self._response = response or _make_completion()
        self._raises = raises
        self.call_count = 0

    async def complete(self, prompt, temperature, seed_hex) -> Completion:
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        return self._response


# ---------------------------------------------------------------------------
# _with_retry
# ---------------------------------------------------------------------------
class TestWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        calls = []

        async def fn():
            calls.append(1)
            return _make_completion()

        result = await _with_retry(fn, max_retries=3)
        assert len(calls) == 1
        assert isinstance(result, Completion)

    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self) -> None:
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 3:
                raise RetryableAPIError("transient")
            return _make_completion()

        with patch("pilot.sampling.asyncio.sleep", new_callable=AsyncMock):
            result = await _with_retry(fn, max_retries=5)
        assert len(calls) == 3
        assert isinstance(result, Completion)

    @pytest.mark.asyncio
    async def test_raises_api_error_after_exhaustion(self) -> None:
        async def fn():
            raise RetryableAPIError("always fails")

        with patch("pilot.sampling.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(APIError):
                await _with_retry(fn, max_retries=2)

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        calls = []

        async def fn():
            calls.append(1)
            raise APIError("fatal")

        with pytest.raises(APIError):
            await _with_retry(fn, max_retries=5)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# _load_existing_seeds
# ---------------------------------------------------------------------------
class TestLoadExistingSeeds:
    def test_nonexistent_file_returns_empty(self) -> None:
        assert _load_existing_seeds(Path("/tmp/nonexistent_xyz.jsonl")) == set()

    def test_reads_seeds_from_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write(json.dumps({"seed_hex": "abc123"}) + "\n")
            f.write(json.dumps({"seed_hex": "def456"}) + "\n")
            fname = f.name
        seeds = _load_existing_seeds(Path(fname))
        assert seeds == {"abc123", "def456"}

    def test_skips_malformed_lines(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write("not json\n")
            f.write(json.dumps({"seed_hex": "good"}) + "\n")
            fname = f.name
        seeds = _load_existing_seeds(Path(fname))
        assert "good" in seeds


# ---------------------------------------------------------------------------
# sample_problem — idempotency and schema
# ---------------------------------------------------------------------------
class TestSampleProblem:
    @pytest.mark.asyncio
    async def test_generates_expected_number_of_samples(self) -> None:
        client = _MockClient(_make_completion("Answer: B"))
        problem = _make_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p0000.jsonl"
            samples = await sample_problem(
                problem, n_total=4, client=client, output_path=out
            )
            assert len(samples) == 4
            assert client.call_count == 4

    @pytest.mark.asyncio
    async def test_idempotent_second_run_is_noop(self) -> None:
        client = _MockClient(_make_completion("Answer: B"))
        problem = _make_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p0000.jsonl"
            await sample_problem(problem, n_total=4, client=client, output_path=out)
            first_count = client.call_count
            await sample_problem(problem, n_total=4, client=client, output_path=out)
            assert client.call_count == first_count  # no new calls

    @pytest.mark.asyncio
    async def test_output_file_created(self) -> None:
        client = _MockClient()
        problem = _make_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p0000.jsonl"
            await sample_problem(problem, n_total=2, client=client, output_path=out)
            assert out.exists()

    @pytest.mark.asyncio
    async def test_output_jsonl_valid_schema(self) -> None:
        client = _MockClient(_make_completion("Answer: B"))
        problem = _make_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p0000.jsonl"
            await sample_problem(problem, n_total=3, client=client, output_path=out)
            lines = [
                json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()
            ]
            assert len(lines) == 3
            for line in lines:
                assert "schema_version" in line
                assert "problem_id" in line
                assert "seed_hex" in line
                assert "full_response" in line

    @pytest.mark.asyncio
    async def test_api_error_skips_sample_does_not_fabricate(self) -> None:
        client = _MockClient(raises=APIError("fatal"))
        problem = _make_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p0000.jsonl"
            samples = await sample_problem(
                problem, n_total=3, client=client, output_path=out
            )
            assert len(samples) == 0  # no fabricated samples
            assert not out.exists() or out.stat().st_size == 0

    @pytest.mark.asyncio
    async def test_correct_field_set_properly(self) -> None:
        client = _MockClient(_make_completion("Answer: B"))  # problem GT is B
        problem = _make_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p0000.jsonl"
            samples = await sample_problem(
                problem, n_total=1, client=client, output_path=out
            )
            assert samples[0].correct is True

    @pytest.mark.asyncio
    async def test_incorrect_answer_marked_false(self) -> None:
        client = _MockClient(_make_completion("Answer: A"))  # wrong, GT is B
        problem = _make_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p0000.jsonl"
            samples = await sample_problem(
                problem, n_total=1, client=client, output_path=out
            )
            assert samples[0].correct is False


# ---------------------------------------------------------------------------
# ConfigError
# ---------------------------------------------------------------------------
class TestConfigError:
    def test_together_client_raises_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        with pytest.raises(ConfigError):
            TogetherClient(api_key=None)

    def test_together_client_accepts_explicit_key(self) -> None:
        client = TogetherClient(api_key="test_key")
        assert client._key == "test_key"
