"""Tests for pilot/prompts.py."""

from __future__ import annotations

import hashlib

from pilot.prompts import (
    MAIN_PROMPT_TEMPLATE,
    PASS5_SCORER_HASH,
    PASS5_SCORER_TEMPLATE,
    PROMPT_TEMPLATE_HASH,
    render_pass5_prompt,
    render_prompt,
)

_EXPECTED_PROMPT_HASH = (
    "e3544f731c3b30d49f373585e192da39347a272fe68fd9d309e8aafc763b73c1"
)
_EXPECTED_SCORER_HASH = (
    "0ca0b0f97745fcb58756e9b0a42b4c2e2ff298eb92c6e3e35096567c34e0f303"
)


def test_prompt_hash_matches_expected() -> None:
    assert PROMPT_TEMPLATE_HASH == _EXPECTED_PROMPT_HASH


def test_scorer_hash_matches_expected() -> None:
    assert PASS5_SCORER_HASH == _EXPECTED_SCORER_HASH


def test_prompt_hash_is_sha256_of_template() -> None:
    computed = hashlib.sha256(MAIN_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    assert computed == PROMPT_TEMPLATE_HASH


def test_scorer_hash_is_sha256_of_template() -> None:
    computed = hashlib.sha256(PASS5_SCORER_TEMPLATE.encode("utf-8")).hexdigest()
    assert computed == PASS5_SCORER_HASH


def test_render_prompt_contains_question() -> None:
    rendered = render_prompt(
        question_text="What is the speed of light?",
        option_a="3e8 m/s",
        option_b="3e6 m/s",
        option_c="3e10 m/s",
        option_d="1e8 m/s",
    )
    assert "What is the speed of light?" in rendered
    assert "A) 3e8 m/s" in rendered
    assert "B) 3e6 m/s" in rendered
    assert "C) 3e10 m/s" in rendered
    assert "D) 1e8 m/s" in rendered
    assert 'Answer: X" where X is one of A, B, C, or D' in rendered


def test_render_prompt_ends_with_newline() -> None:
    rendered = render_prompt("Q?", "a", "b", "c", "d")
    assert rendered.endswith("\n")


def test_render_pass5_contains_all_fields() -> None:
    rendered = render_pass5_prompt(
        question_text="What is 2+2?",
        option_a="3",
        option_b="4",
        option_c="5",
        option_d="6",
        ground_truth="B",
        full_response="After thinking... Answer: B",
    )
    assert "What is 2+2?" in rendered
    assert "Correct answer: B" in rendered
    assert "After thinking... Answer: B" in rendered
    assert "CORRECT" in rendered
    assert "INCORRECT" in rendered
    assert "TRULY_UNPARSEABLE" in rendered


def test_render_prompt_no_extra_braces() -> None:
    rendered = render_prompt("Q?", "a", "b", "c", "d")
    assert "{" not in rendered
    assert "}" not in rendered
