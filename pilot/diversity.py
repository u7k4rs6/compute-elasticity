"""Embedding-based diversity for reasoning traces.

Diversity = mean pairwise cosine distance over a set of CoT traces.
Used to compute the H3 predictor feature at N=4.
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

_ANSWER_LINE_RE = re.compile(r"(?i)^\s*answer\s*[:\s]\s*[A-D]\s*$")


def truncate_trace(trace: str, max_tokens: int = 512) -> tuple[str, bool]:
    """Strip the final 'Answer: X' line and truncate at last sentence boundary.

    Returns (truncated_text, was_truncated).
    Truncation never splits mid-word; it snaps to the last period/newline.
    """
    lines = trace.splitlines()
    # Drop trailing answer line
    if lines and _ANSWER_LINE_RE.match(lines[-1]):
        lines = lines[:-1]
    text = "\n".join(lines).rstrip()

    # Approximate token count as whitespace-split words (fast proxy)
    words = text.split()
    if len(words) <= max_tokens:
        return text, False

    truncated_words = words[:max_tokens]
    truncated = " ".join(truncated_words)

    # Snap back to the last sentence-ending punctuation to avoid mid-sentence cuts
    for sep in (".", "!", "?", "\n"):
        idx = truncated.rfind(sep)
        if idx > len(truncated) // 2:
            truncated = truncated[: idx + 1]
            break

    return truncated, True


def compute_diversity(traces: list[str], embedder: object) -> float:
    """Return mean pairwise cosine distance over the set of traces.

    Args:
        traces: List of reasoning trace strings.
        embedder: Any object with an `encode(texts) -> np.ndarray` method
                  (e.g. a sentence_transformers.SentenceTransformer instance).

    Returns:
        Mean pairwise cosine distance in [0, 1]. Returns 0.0 for singleton sets.
    """
    if len(traces) < 2:
        return 0.0

    truncated = [truncate_trace(t)[0] for t in traces]
    embeddings: np.ndarray = embedder.encode(truncated, normalize_embeddings=True)

    # Cosine distance = 1 - cosine_similarity; with normalised vectors: sim = dot
    n = len(embeddings)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            total_dist += 1.0 - sim
            count += 1

    return total_dist / count if count > 0 else 0.0
