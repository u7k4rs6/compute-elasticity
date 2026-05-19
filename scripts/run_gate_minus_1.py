"""Phase 4 — Gate -1: Embedder validation.

Validates that bge-small-en-v1.5 meaningfully distinguishes reasoning traces
from different problems vs. within the same problem.

Steps:
  1. Select 3 gate problems (one per subject) from the locked 50.
  2. Sample N=4 completions per problem (idempotent).
  3. Generate 15 trace pairs (8 within-problem, 7 between-problem; seed=42).
  4. Interactive labeling loop — label each pair s/d/a/q.
  5. Compute embedding distances; run Mann-Whitney U (one-sided).
  6. PASS if mean(diff) > mean(same) AND p < 0.10 AND gap > 0.01.
  7. Write outputs/gate_minus_1_labels.json.

Usage:
    source .venv/bin/activate
    python scripts/run_gate_minus_1.py

Non-interactive (uses auto-labels from pair construction; for smoke-testing):
    GATE_NONINTERACTIVE=1 python scripts/run_gate_minus_1.py
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_N_SAMPLES: int = 4
_N_PAIRS: int = 15
_N_WITHIN: int = 8
_N_BETWEEN: int = 7
_PAIR_SEED: int = 42
_GATE_PASS_P: float = 0.10
_GATE_PASS_MIN_GAP: float = 0.01  # minimum cosine-distance gap to confirm visibility


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _load_locked_ids() -> list[str]:
    ids_path = ROOT / "data" / "problem_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(f"Locked problem IDs not found: {ids_path}")
    return json.loads(ids_path.read_text())


def _select_gate_problems(problems: list[Any]) -> list[Any]:
    """Return one problem per subject (physics/chemistry/biology), first by sorted ID."""
    locked = set(_load_locked_ids())
    by_subject: dict[str, list[Any]] = {}
    for p in problems:
        if p.id in locked:
            by_subject.setdefault(p.subject, []).append(p)

    gate: list[Any] = []
    for subject in ("physics", "chemistry", "biology"):
        bucket = sorted(by_subject.get(subject, []), key=lambda p: p.id)
        if not bucket:
            raise RuntimeError(f"No locked problems found for subject: {subject}")
        gate.append(bucket[0])
    return gate


def _generate_pairs(
    problem_ids: list[str],
    n_samples: int,
    n_within: int,
    n_between: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically generate within- and between-problem trace pairs."""
    rng = random.Random(seed)

    within: list[dict[str, Any]] = []
    for pid in problem_ids:
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                within.append(
                    {
                        "problem_a": pid,
                        "trace_idx_a": i,
                        "problem_b": pid,
                        "trace_idx_b": j,
                        "auto_label": "same",
                    }
                )

    between: list[dict[str, Any]] = []
    for k in range(len(problem_ids)):
        for ll in range(k + 1, len(problem_ids)):
            for i in range(n_samples):
                for j in range(n_samples):
                    between.append(
                        {
                            "problem_a": problem_ids[k],
                            "trace_idx_a": i,
                            "problem_b": problem_ids[ll],
                            "trace_idx_b": j,
                            "auto_label": "different",
                        }
                    )

    rng.shuffle(within)
    rng.shuffle(between)

    selected = within[:n_within] + between[:n_between]
    rng.shuffle(selected)

    for idx, pair in enumerate(selected):
        pair["pair_id"] = idx
    return selected


def _load_traces(samples_dir: Path, problem_id: str, n_samples: int) -> list[str]:
    """Read up to n_samples full_response strings from a problem's JSONL file."""
    path = samples_dir / f"{problem_id}.jsonl"
    if not path.exists():
        return []
    traces: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            traces.append(obj["full_response"])
        except (json.JSONDecodeError, KeyError):
            continue
        if len(traces) >= n_samples:
            break
    return traces


def _interactive_label(
    pair: dict[str, Any],
    trace_a: str,
    trace_b: str,
    pair_num: int,
    n_total: int,
) -> str:
    """Display pair previews and return human label: 's', 'd', 'a', or 'q'."""
    from pilot.diversity import truncate_trace

    preview_a = truncate_trace(trace_a)[0][:400].replace("\n", " ")
    preview_b = truncate_trace(trace_b)[0][:400].replace("\n", " ")

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Pair {pair_num}/{n_total}  |  auto_label={pair['auto_label']}")
    print(f"  A [{pair['problem_a']} trace {pair['trace_idx_a']}]:")
    print(f"    {preview_a}")
    print(f"  B [{pair['problem_b']} trace {pair['trace_idx_b']}]:")
    print(f"    {preview_b}")
    print("  → [s]ame  [d]ifferent  [a]bort pair  [q]uit")

    while True:
        try:
            key = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "q"
        if key in ("s", "d", "a", "q"):
            return key
        print("  Please enter s, d, a, or q.")


async def run_gate(interactive: bool = True) -> bool:
    """Run Gate -1 embedder validation; return True if gate passes."""
    import numpy as np
    from scipy import stats
    from sentence_transformers import SentenceTransformer

    from pilot.config import (
        EMBEDDING_MODEL,
        MODEL,
        OUTPUTS_DIR,
        SAMPLES_DIR,
        SCHEMA_VERSION,
        TOGETHER_INPUT_PRICE_PER_TOKEN,
        TOGETHER_OUTPUT_PRICE_PER_TOKEN,
    )
    from pilot.data_loader import load_gpqa_diamond
    from pilot.diversity import truncate_trace
    from pilot.sampling import TogetherClient, sample_problem

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- 1. Select gate problems ---
    logger.info("Loading GPQA Diamond and selecting gate problems…")
    all_problems = load_gpqa_diamond()
    gate_problems = _select_gate_problems(all_problems)
    gate_ids = [p.id for p in gate_problems]
    logger.info(
        "Gate problems: %s",
        ", ".join(f"{p.id} ({p.subject})" for p in gate_problems),
    )

    # --- 2. Sample N=4 completions per problem ---
    try:
        client = TogetherClient()
    except Exception as exc:
        logger.error("Failed to initialise TogetherClient: %s", exc)
        return False

    total_input_tokens = 0
    total_output_tokens = 0
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for problem in gate_problems:
        output_path = SAMPLES_DIR / f"{problem.id}.jsonl"
        logger.info("Sampling %d completions for %s…", _N_SAMPLES, problem.id)
        new_samples = await sample_problem(
            problem=problem,
            n_total=_N_SAMPLES,
            client=client,
            output_path=output_path,
            temperature=0.7,
        )
        for s in new_samples:
            total_input_tokens += s.input_tokens
            total_output_tokens += s.output_tokens
        logger.info("%d new samples for %s.", len(new_samples), problem.id)

    total_cost = (
        total_input_tokens * TOGETHER_INPUT_PRICE_PER_TOKEN
        + total_output_tokens * TOGETHER_OUTPUT_PRICE_PER_TOKEN
    )
    logger.info(
        "Sampling done. New tokens: %d in / %d out. Cost: $%.4f.",
        total_input_tokens,
        total_output_tokens,
        total_cost,
    )

    # --- 3. Load traces from disk ---
    traces: dict[str, list[str]] = {}
    for pid in gate_ids:
        loaded = _load_traces(SAMPLES_DIR, pid, _N_SAMPLES)
        if len(loaded) < _N_SAMPLES:
            logger.error(
                "Problem %s: only %d/%d traces loaded.",
                pid,
                len(loaded),
                _N_SAMPLES,
            )
            return False
        traces[pid] = loaded

    # --- 4. Generate pairs ---
    pairs = _generate_pairs(gate_ids, _N_SAMPLES, _N_WITHIN, _N_BETWEEN, _PAIR_SEED)

    # --- 5. Label pairs ---
    labeled: list[dict[str, Any]] = []
    quit_early = False

    if interactive:
        print(
            "\n" + "=" * 70 + "\nGate -1: Embedder Validation — Interactive Labeling\n"
            "Label each pair: [s]ame reasoning / [d]ifferent reasoning\n" + "=" * 70
        )
        for num, pair in enumerate(pairs, start=1):
            trace_a = traces[pair["problem_a"]][pair["trace_idx_a"]]
            trace_b = traces[pair["problem_b"]][pair["trace_idx_b"]]
            label = _interactive_label(pair, trace_a, trace_b, num, len(pairs))
            if label == "q":
                quit_early = True
                break
            if label == "a":
                continue
            pair["human_label"] = label
            labeled.append(pair)
    else:
        # Non-interactive: derive label from auto_label ("same"→"s", "different"→"d")
        for pair in pairs:
            pair["human_label"] = pair["auto_label"][0]
            labeled.append(pair)

    if len(labeled) < 5:
        logger.error(
            "Too few labeled pairs (%d < 5) to run the gate check.", len(labeled)
        )
        return False

    # --- 6. Embed traces and compute pairwise distances ---
    logger.info("Loading embedder %s…", EMBEDDING_MODEL)
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:
        logger.error("Failed to load embedder: %s", exc)
        logger.error(
            "Gate -1 FAIL: swap to jinaai/jina-embeddings-v3 per preregistration §8."
        )
        return False

    # Collect unique (problem, idx) keys needed
    needed: set[tuple[str, int]] = set()
    for pair in labeled:
        needed.add((pair["problem_a"], pair["trace_idx_a"]))
        needed.add((pair["problem_b"], pair["trace_idx_b"]))

    keys = list(needed)
    texts = [truncate_trace(traces[pid][idx])[0] for pid, idx in keys]
    emb_matrix = embedder.encode(texts, normalize_embeddings=True)
    emb_map = {k: emb_matrix[i] for i, k in enumerate(keys)}

    for pair in labeled:
        ea = emb_map[(pair["problem_a"], pair["trace_idx_a"])]
        eb = emb_map[(pair["problem_b"], pair["trace_idx_b"])]
        pair["embedding_distance"] = round(float(1.0 - np.dot(ea, eb)), 6)

    # --- 7. Mann-Whitney U (one-sided: different > same) ---
    same_dists = [p["embedding_distance"] for p in labeled if p["human_label"] == "s"]
    diff_dists = [p["embedding_distance"] for p in labeled if p["human_label"] == "d"]

    if not same_dists or not diff_dists:
        logger.error(
            "Need both labels; got %d same, %d different.",
            len(same_dists),
            len(diff_dists),
        )
        return False

    mean_same = float(np.mean(same_dists))
    mean_diff = float(np.mean(diff_dists))
    gap = mean_diff - mean_same
    u_stat, p_val = stats.mannwhitneyu(diff_dists, same_dists, alternative="greater")

    gate_pass = (
        mean_diff > mean_same
        and float(p_val) < _GATE_PASS_P
        and gap > _GATE_PASS_MIN_GAP
    )

    # --- 8. Human-readable summary ---
    sep = "=" * 70
    print(f"\n{sep}")
    print("Gate -1 Result")
    print(sep)
    print(f"  Gate problems   : {', '.join(gate_ids)}")
    print(
        f"  Pairs labeled   : {len(labeled)}  (same={len(same_dists)}, diff={len(diff_dists)})"
    )
    print(f"  Mean dist same  : {mean_same:.4f}")
    print(f"  Mean dist diff  : {mean_diff:.4f}")
    print(f"  Gap (diff-same) : {gap:.4f}  (threshold: >{_GATE_PASS_MIN_GAP})")
    print(f"  Mann-Whitney U  : {u_stat:.1f}")
    print(f"  p (one-sided)   : {p_val:.4f}  (threshold: <{_GATE_PASS_P})")
    print(f"  Sampling cost   : ${total_cost:.4f}")
    print(f"  Verdict         : {'PASS' if gate_pass else 'FAIL'}")
    print(sep)

    if not gate_pass:
        if mean_diff <= mean_same:
            logger.error(
                "mean_dist(diff)=%.4f ≤ mean_dist(same)=%.4f", mean_diff, mean_same
            )
        if float(p_val) >= _GATE_PASS_P:
            logger.error("p=%.4f ≥ threshold %.2f", p_val, _GATE_PASS_P)
        if gap <= _GATE_PASS_MIN_GAP:
            logger.error("gap=%.4f ≤ min_gap %.2f", gap, _GATE_PASS_MIN_GAP)
        logger.error(
            "If gate still fails after embedder swap, kill H3 and re-tag "
            "as pre-pilot-v6.0.2-H3-killed per preregistration §8."
        )

    # --- 9. Write output JSON ---
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "gate_minus_1_labels.json"
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": MODEL,
        "embedder": EMBEDDING_MODEL,
        "n_samples_per_problem": _N_SAMPLES,
        "gate_problems": gate_ids,
        "quit_early": quit_early,
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "problem_a": p["problem_a"],
                "trace_idx_a": p["trace_idx_a"],
                "problem_b": p["problem_b"],
                "trace_idx_b": p["trace_idx_b"],
                "auto_label": p["auto_label"],
                "human_label": p["human_label"],
                "embedding_distance": p["embedding_distance"],
            }
            for p in labeled
        ],
        "stats": {
            "n_labeled": len(labeled),
            "n_same": len(same_dists),
            "n_different": len(diff_dists),
            "mean_dist_same": round(mean_same, 6),
            "mean_dist_different": round(mean_diff, 6),
            "gap": round(gap, 6),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": round(float(p_val), 6),
            "gate_pass": gate_pass,
        },
        "total_cost_usd": round(total_cost, 6),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", out_path)

    return gate_pass


def main() -> None:
    """Entry point."""
    _load_env()
    interactive = os.getenv("GATE_NONINTERACTIVE") != "1"
    ok = asyncio.run(run_gate(interactive=interactive))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
