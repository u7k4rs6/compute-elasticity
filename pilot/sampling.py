"""Async API clients for Together AI and DeepInfra, plus idempotent sampler.

No live calls are made from this module during tests — mock the HTTP layer.
All actual sampling happens via scripts/run_*.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from pilot.config import (
    DEEPINFRA_API_BASE,
    MAX_RETRIES,
    MODEL,
    SCHEMA_VERSION,
    TEMPERATURE_MAIN,
    TOGETHER_API_BASE,
)
from pilot.prompts import PROMPT_TEMPLATE_HASH
from pilot.scoring import extract_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------
class PilotError(Exception):
    """Base for all pilot-specific errors."""


class ConfigError(PilotError):
    """Missing or invalid configuration (e.g. absent env var)."""


class APIError(PilotError):
    """Non-retryable API error."""


class RetryableAPIError(PilotError):
    """Transient API error — caller should retry with backoff."""


# ---------------------------------------------------------------------------
# Completion result
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Completion:
    """Raw completion from an API provider."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str
    model: str
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------
class APIClient(ABC):
    """Abstract async inference client."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float,
        seed_hex: str,
    ) -> Completion:
        """Send a prompt and return a Completion.

        Implementations must NOT retry internally — retries are handled
        by sample_problem().
        """


# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------
async def _with_retry(
    coro_fn,
    max_retries: int = MAX_RETRIES,
) -> Completion:
    """Call coro_fn() with exponential back-off + jitter on RetryableAPIError."""
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except RetryableAPIError as exc:
            if attempt == max_retries:
                raise APIError(f"Exhausted {max_retries} retries: {exc}") from exc
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "Attempt %d failed (%s); retrying in %.1fs", attempt + 1, exc, delay
            )
            await asyncio.sleep(delay)
    raise APIError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Together AI client
# ---------------------------------------------------------------------------
class TogetherClient(APIClient):
    """Async client for Together AI (OpenAI-compatible chat API)."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("TOGETHER_API_KEY")
        if not key:
            raise ConfigError(
                "TOGETHER_API_KEY env var is not set. "
                "Add it to your .env file (see .env.example)."
            )
        self._key = key
        self._base = TOGETHER_API_BASE

    async def complete(
        self,
        prompt: str,
        temperature: float = TEMPERATURE_MAIN,
        seed_hex: str = "00000000",
    ) -> Completion:
        async def _call() -> Completion:
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self._base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": 2048,
                        "seed": int(seed_hex[:8], 16) if seed_hex else None,
                    },
                )
            latency_ms = (time.monotonic() - t0) * 1000

            if resp.status_code in (429, 500, 502, 503, 504):
                raise RetryableAPIError(f"HTTP {resp.status_code}")
            if resp.status_code != 200:
                raise APIError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return Completion(
                text=choice["message"]["content"],
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                latency_ms=latency_ms,
                provider="together_ai",
                model=MODEL,
                raw=data,
            )

        return await _with_retry(_call)


# ---------------------------------------------------------------------------
# DeepInfra client
# ---------------------------------------------------------------------------
class DeepInfraClient(APIClient):
    """Async client for DeepInfra (OpenAI-compatible chat API)."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("DEEPINFRA_API_KEY")
        if not key:
            raise ConfigError(
                "DEEPINFRA_API_KEY env var is not set. "
                "Add it to your .env file (see .env.example)."
            )
        self._key = key
        self._base = DEEPINFRA_API_BASE

    async def complete(
        self,
        prompt: str,
        temperature: float = TEMPERATURE_MAIN,
        seed_hex: str = "00000000",
    ) -> Completion:
        async def _call() -> Completion:
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self._base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": 2048,
                    },
                )
            latency_ms = (time.monotonic() - t0) * 1000

            if resp.status_code in (429, 500, 502, 503, 504):
                raise RetryableAPIError(f"HTTP {resp.status_code}")
            if resp.status_code != 200:
                raise APIError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return Completion(
                text=choice["message"]["content"],
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                latency_ms=latency_ms,
                provider="deepinfra",
                model=MODEL,
                raw=data,
            )

        return await _with_retry(_call)


# ---------------------------------------------------------------------------
# Idempotent problem sampler
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Sample:
    """A completed, scored sample ready for serialisation."""

    problem_id: str
    subject: str
    ground_truth: str
    temperature: float
    sample_idx: int
    n_total_in_batch: int
    seed_hex: str
    full_response: str
    extracted_answer: str | None
    extraction_pass: int
    correct: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str
    timestamp: str
    prompt_template_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "problem_id": self.problem_id,
            "subject": self.subject,
            "ground_truth": self.ground_truth,
            "model": MODEL,
            "provider": self.provider,
            "temperature": self.temperature,
            "sample_idx": self.sample_idx,
            "n_total_in_batch": self.n_total_in_batch,
            "seed_hex": self.seed_hex,
            "prompt_template_hash": f"sha256:{self.prompt_template_hash}",
            "full_response": self.full_response,
            "extracted_answer": self.extracted_answer or "UNPARSEABLE",
            "extraction_pass": self.extraction_pass,
            "correct": self.correct,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "api_metadata": {},
        }


def _load_existing_seeds(path: Path) -> set[str]:
    """Return set of seed_hex values already written to an output JSONL."""
    if not path.exists():
        return set()
    seeds: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            seeds.add(json.loads(line)["seed_hex"])
        except (json.JSONDecodeError, KeyError):
            pass
    return seeds


async def sample_problem(
    problem: Any,  # pilot.data_loader.Problem
    n_total: int,
    client: APIClient,
    output_path: Path,
    temperature: float = TEMPERATURE_MAIN,
) -> list[Sample]:
    """Idempotently generate n_total samples for a problem.

    Reads existing output_path to avoid re-generating already-collected samples.
    Appends new samples as JSONL lines (one per sample).

    Args:
        problem: A Problem dataclass instance.
        n_total: Number of samples to collect (e.g. 64).
        client: Async APIClient (Together or DeepInfra).
        output_path: Path to the .jsonl output file.
        temperature: Sampling temperature.

    Returns:
        List of Sample objects generated in this call (not including pre-existing ones).
    """
    from pilot.prompts import render_prompt

    existing_seeds = _load_existing_seeds(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = render_prompt(
        question_text=problem.question,
        option_a=problem.option_a,
        option_b=problem.option_b,
        option_c=problem.option_c,
        option_d=problem.option_d,
    )

    import datetime

    new_samples: list[Sample] = []

    with output_path.open("a") as fh:
        for idx in range(n_total):
            # Generate a deterministic seed per problem/idx/temperature to ensure
            # idempotency: same seed → same result on re-run
            seed_hex = (
                f"{hash((problem.id, idx, temperature)) & 0xFFFFFFFFFFFFFFFF:016x}"
            )

            if seed_hex in existing_seeds:
                continue  # already collected — idempotent skip

            try:
                completion = await client.complete(
                    prompt=prompt,
                    temperature=temperature,
                    seed_hex=seed_hex,
                )
            except APIError as exc:
                logger.error(
                    "API error for %s idx=%d: %s — skipping (never fabricate samples)",
                    problem.id,
                    idx,
                    exc,
                )
                continue

            extraction = extract_answer(completion.text)
            is_correct = (
                extraction.answer == problem.ground_truth
                if extraction.answer is not None
                else False
            )

            sample = Sample(
                problem_id=problem.id,
                subject=problem.subject,
                ground_truth=problem.ground_truth,
                temperature=temperature,
                sample_idx=idx,
                n_total_in_batch=n_total,
                seed_hex=seed_hex,
                full_response=completion.text,
                extracted_answer=extraction.answer,
                extraction_pass=extraction.pass_number,
                correct=is_correct,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                latency_ms=completion.latency_ms,
                provider=completion.provider,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                prompt_template_hash=PROMPT_TEMPLATE_HASH,
            )
            fh.write(json.dumps(sample.to_dict()) + "\n")
            fh.flush()
            new_samples.append(sample)
            existing_seeds.add(seed_hex)

    return new_samples
