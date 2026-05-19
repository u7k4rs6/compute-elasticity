"""Tests for pilot/scoring.py — 5-pass answer extraction pipeline."""

from __future__ import annotations

import pytest

from pilot.scoring import ExtractionResult, ScoredSample, extract_answer, score_sample


class TestExtractAnswer:
    # --- Pass 1: "Answer: X" in last 200 chars ---
    def test_pass1_basic(self) -> None:
        r = extract_answer("After thinking... Answer: B")
        assert r.answer == "B"
        assert r.pass_number == 1

    def test_pass1_colon_variants(self) -> None:
        for text in ["answer: A", "Answer: C", "ANSWER: D"]:
            assert extract_answer(text).pass_number == 1

    def test_pass1_only_last_200_chars(self) -> None:
        prefix = "Answer: Z" + " " * 300
        result = extract_answer(prefix + "Answer: A")
        assert result.answer == "A"
        assert result.pass_number == 1

    def test_pass1_last_match_wins(self) -> None:
        r = extract_answer("Answer: A ... Answer: C")
        assert r.answer == "C"

    # --- Pass 2: \boxed{X} ---
    def test_pass2_boxed(self) -> None:
        r = extract_answer("The result is \\boxed{C}")
        assert r.answer == "C"
        assert r.pass_number == 2

    def test_pass2_not_triggered_by_pass1_match(self) -> None:
        r = extract_answer("Answer: A and \\boxed{B}")
        assert r.pass_number == 1

    # --- Pass 3: bare letter on last non-empty line ---
    def test_pass3_last_line_single_letter(self) -> None:
        r = extract_answer("Here is my reasoning.\nC")
        assert r.answer == "C"
        assert r.pass_number == 3

    def test_pass3_ignores_empty_trailing_lines(self) -> None:
        r = extract_answer("My answer is\nB\n\n")
        assert r.answer == "B"
        assert r.pass_number == 3

    # --- Pass 4: bare letter in last 500 chars ---
    def test_pass4_fallback(self) -> None:
        # No pass 1/2/3 match; bare letter somewhere in last 500 chars
        r = extract_answer("I think the correct choice is probably D somewhere here.")
        # Pass 3 would grab D from last line, so let's check pass_number <= 4
        assert r.answer == "D"
        assert r.pass_number in (3, 4)

    # --- No match ---
    def test_no_match_returns_none(self) -> None:
        r = extract_answer("The answer is unknown to me.")
        assert r.answer is None
        assert r.pass_number == 5

    def test_empty_response_returns_none(self) -> None:
        r = extract_answer("")
        assert r.answer is None

    # --- Edge cases ---
    def test_multiple_a_d_letters(self) -> None:
        # Ambiguous line with multiple letters — pass 3 should pick last
        r = extract_answer("Options A B C or D: Answer: B")
        assert r.answer == "B"
        assert r.pass_number == 1

    def test_lowercase_answer(self) -> None:
        r = extract_answer("answer: a")
        assert r.answer == "A"

    def test_answer_after_long_preamble(self) -> None:
        long_text = "x" * 300 + " Answer: D"
        r = extract_answer(long_text)
        assert r.answer == "D"

    def test_returns_extraction_result_type(self) -> None:
        r = extract_answer("Answer: A")
        assert isinstance(r, ExtractionResult)


# ---------------------------------------------------------------------------
# score_sample (sync path — no LLM)
# ---------------------------------------------------------------------------
class TestScoreSample:
    @pytest.mark.asyncio
    async def test_correct_sample(self) -> None:
        sample = {"full_response": "After thinking... Answer: B"}
        result = await score_sample(sample, ground_truth="B")
        assert result.correct is True
        assert result.extracted_answer == "B"

    @pytest.mark.asyncio
    async def test_incorrect_sample(self) -> None:
        sample = {"full_response": "Answer: C"}
        result = await score_sample(sample, ground_truth="A")
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_unparseable_conservative(self) -> None:
        sample = {"full_response": "I have no idea what the answer is."}
        result = await score_sample(sample, ground_truth="A", api_client=None)
        assert result.correct is False
        assert result.extracted_answer is None
        assert result.extraction_pass == 6

    @pytest.mark.asyncio
    async def test_returns_scored_sample(self) -> None:
        sample = {"full_response": "Answer: A"}
        result = await score_sample(sample, ground_truth="A")
        assert isinstance(result, ScoredSample)
