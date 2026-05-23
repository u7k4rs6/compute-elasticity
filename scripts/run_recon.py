"""Phase 5 — Day 0 reconnaissance.

Runs 1 sample per problem across 10 problems (3 Gate-1 reuses + 7 new),
scores each with the 5-pass extractor, and determines which capability band
the model falls into.

Band thresholds (N=1 accuracy over 10 problems):
  Green:  0.25 ≤ acc ≤ 0.45  → proceed normally
  Yellow: 0.15 ≤ acc < 0.25 or 0.45 < acc ≤ 0.55 → proceed with flag
  Red:    acc < 0.15 or acc > 0.55 → pivot (see PRD §5 stop conditions)

The 10 recon problems become the fixed Phase 7 temperature side-test subset.

Usage:
    source .venv/bin/activate
    python scripts/run_recon.py
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_N_RECON: int = 10
_N_GATE: int = 3
_N_NEW: int = 7
_TEMPERATURE: float = 0.7

_GREEN_LOW: float = 0.25
_GREEN_HIGH: float = 0.45
_YELLOW_LOW: float = 0.15
_YELLOW_HIGH: float = 0.55


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


def _load_gate_problem_ids() -> list[str]:
    """Read gate_problems list from outputs/gate_minus_1_labels.json."""
    labels_path = ROOT / "outputs" / "gate_minus_1_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Gate -1 labels not found: {labels_path}\n"
            "Run scripts/run_gate_minus_1.py first."
        )
    data = json.loads(labels_path.read_text())
    return data["gate_problems"]


def _select_new_recon_problems(
    all_problems: list[Any],
    gate_ids: list[str],
    locked_ids: list[str],
) -> list[Any]:
    """Return first 7 locked problems not already in Gate -1, sorted by ID."""
    locked = set(locked_ids)
    gate = set(gate_ids)
    remaining = [p for p in all_problems if p.id in locked and p.id not in gate]
    remaining.sort(key=lambda p: p.id)
    return remaining[:_N_NEW]


def _read_first_sample(samples_dir: Path, problem_id: str) -> dict[str, Any] | None:
    """Return the first valid JSON object from a problem's JSONL file."""
    path = samples_dir / f"{problem_id}.jsonl"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _band(accuracy: float) -> tuple[str, str]:
    """Return (band, action) for a given N=1 accuracy."""
    if _GREEN_LOW <= accuracy <= _GREEN_HIGH:
        return "green", "proceed"
    if _YELLOW_LOW <= accuracy < _GREEN_LOW or _GREEN_HIGH < accuracy <= _YELLOW_HIGH:
        return "yellow", "proceed_with_flag"
    return "red", "pivot"


async def run_recon() -> bool:
    """Run Day 0 recon; return True if band is green or yellow (i.e. not red)."""
    from pilot.config import (
        MODEL,
        OUTPUTS_DIR,
        SAMPLES_DIR,
        SCHEMA_VERSION,
        TOGETHER_INPUT_PRICE_PER_TOKEN,
        TOGETHER_OUTPUT_PRICE_PER_TOKEN,
    )
    from pilot.data_loader import load_gpqa_diamond
    from pilot.sampling import TogetherClient, sample_problem
    from pilot.scoring import extract_answer, pass5_score

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- 1. Load locked IDs ---
    locked_ids = _load_locked_ids()
    logger.info("Loaded %d locked problem IDs.", len(locked_ids))

    # --- 2. Load Gate -1 problem IDs from labels file ---
    gate_ids = _load_gate_problem_ids()
    logger.info("Gate -1 problems: %s", ", ".join(gate_ids))

    # --- 3. Select 7 new problems ---
    logger.info("Loading GPQA Diamond…")
    all_problems = load_gpqa_diamond()
    problem_map = {p.id: p for p in all_problems}

    new_problems = _select_new_recon_problems(all_problems, gate_ids, locked_ids)
    if len(new_problems) < _N_NEW:
        logger.error(
            "Only %d new recon problems available (need %d).",
            len(new_problems),
            _N_NEW,
        )
        return False

    gate_problems = [problem_map[pid] for pid in gate_ids]
    recon_problems = gate_problems + new_problems
    logger.info(
        "Recon set: %s",
        ", ".join(f"{p.id} ({p.subject})" for p in recon_problems),
    )

    # --- 4. Ensure 1 sample per problem at T=0.7 ---
    try:
        client = TogetherClient()
    except Exception as exc:
        logger.error("Failed to initialise TogetherClient: %s", exc)
        return False

    total_input_tokens = 0
    total_output_tokens = 0
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for problem in recon_problems:
        output_path = SAMPLES_DIR / f"{problem.id}.jsonl"
        new_samples = await sample_problem(
            problem=problem,
            n_total=1,
            client=client,
            output_path=output_path,
            temperature=_TEMPERATURE,
        )
        for s in new_samples:
            total_input_tokens += s.input_tokens
            total_output_tokens += s.output_tokens
        if new_samples:
            logger.info("Sampled 1 new completion for %s.", problem.id)
        else:
            logger.info("Reusing existing sample for %s.", problem.id)

    total_cost = (
        total_input_tokens * TOGETHER_INPUT_PRICE_PER_TOKEN
        + total_output_tokens * TOGETHER_OUTPUT_PRICE_PER_TOKEN
    )
    logger.info(
        "Sampling done. %d in / %d out tokens. Cost: $%.4f.",
        total_input_tokens,
        total_output_tokens,
        total_cost,
    )

    # --- 5 & 6. Score each problem ---
    per_problem: list[dict[str, Any]] = []
    n_correct = 0
    by_subject: dict[str, dict[str, int]] = {}

    for problem in recon_problems:
        sample = _read_first_sample(SAMPLES_DIR, problem.id)
        if sample is None:
            logger.error("No sample found for %s — scoring as incorrect.", problem.id)
            extracted_answer: str | None = None
            extraction_pass = 6
            correct = False
        else:
            full_response: str = sample.get("full_response", "")
            result = extract_answer(full_response)
            extracted_answer = result.answer
            extraction_pass = result.pass_number

            if extracted_answer is not None:
                correct = extracted_answer == problem.ground_truth
            else:
                # Pass 5: LLM scorer
                logger.info("Pass 5 needed for %s — invoking LLM scorer.", problem.id)
                correct = await pass5_score(
                    question_text=problem.question,
                    option_a=problem.option_a,
                    option_b=problem.option_b,
                    option_c=problem.option_c,
                    option_d=problem.option_d,
                    ground_truth=problem.ground_truth,
                    full_response=full_response,
                    api_client=client,
                )
                extraction_pass = 5

        if correct:
            n_correct += 1

        subj = problem.subject
        if subj not in by_subject:
            by_subject[subj] = {"n": 0, "correct": 0}
        by_subject[subj]["n"] += 1
        if correct:
            by_subject[subj]["correct"] += 1

        is_gate = problem.id in set(gate_ids)
        per_problem.append(
            {
                "problem_id": problem.id,
                "subject": subj,
                "ground_truth": problem.ground_truth,
                "extracted_answer": extracted_answer or "UNPARSEABLE",
                "extraction_pass": extraction_pass,
                "correct": correct,
                "reused_from_gate_1": is_gate,
            }
        )
        logger.info(
            "  %s  (%s)  gt=%s  ans=%s  pass=%d  %s",
            problem.id,
            subj,
            problem.ground_truth,
            extracted_answer or "?",
            extraction_pass,
            "CORRECT" if correct else "INCORRECT",
        )

    # --- 7. Band determination ---
    accuracy = n_correct / _N_RECON
    band, action = _band(accuracy)

    # --- 8. Print summary ---
    sep = "=" * 70
    print(f"\n{sep}")
    print("Day 0 Reconnaissance — Summary")
    print(sep)
    print(
        f"  {'Problem':<28}  {'Subject':<12}  {'GT':<4}  {'Ans':<4}  {'Pass':<5}  Result"
    )
    print(f"  {'-'*28}  {'-'*12}  {'-'*4}  {'-'*4}  {'-'*5}  ------")
    for pp in per_problem:
        flag = "*" if pp["reused_from_gate_1"] else " "
        print(
            f"  {flag}{pp['problem_id']:<27}  {pp['subject']:<12}  "
            f"{pp['ground_truth']:<4}  {pp['extracted_answer']:<4}  "
            f"{pp['extraction_pass']:<5}  {'CORRECT' if pp['correct'] else 'incorrect'}"
        )
    print("  (* = reused from Gate -1)")
    print()
    print("  Subject breakdown:")
    for subj, counts in sorted(by_subject.items()):
        print(f"    {subj:<12}: {counts['correct']}/{counts['n']} correct")
    print()
    print(f"  Overall accuracy : {n_correct}/{_N_RECON} = {accuracy:.2%}")
    print(
        f"  Band thresholds  : "
        f"Red <{_YELLOW_LOW:.0%} | Yellow {_YELLOW_LOW:.0%}–{_GREEN_LOW:.0%} | "
        f"Green {_GREEN_LOW:.0%}–{_GREEN_HIGH:.0%} | "
        f"Yellow {_GREEN_HIGH:.0%}–{_YELLOW_HIGH:.0%} | Red >{_YELLOW_HIGH:.0%}"
    )
    print(f"  Band             : {band.upper()} → {action}")
    print(f"  Sampling cost    : ${total_cost:.4f}")
    print(sep)

    if band == "red":
        logger.error(
            "RED band: accuracy %.2f is outside Yellow range. "
            "Pivot benchmark/model per PRD §5.",
            accuracy,
        )
    elif band == "yellow":
        logger.warning(
            "YELLOW band: accuracy %.2f is outside Green range. "
            "Proceeding with flag — document in writeup.",
            accuracy,
        )
    else:
        logger.info("GREEN band: accuracy %.2f. Proceeding to Phase 6.", accuracy)

    # --- 9. Write output JSON ---
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "recon_results.json"
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "day_0_recon",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": MODEL,
        "temperature": _TEMPERATURE,
        "n_samples_per_problem": 1,
        "gate_1_problems_reused": gate_ids,
        "new_recon_problems": [p.id for p in new_problems],
        "per_problem": per_problem,
        "summary": {
            "n_problems": _N_RECON,
            "n_correct": n_correct,
            "accuracy": round(accuracy, 6),
            "by_subject": by_subject,
            "band": band,
            "band_action": action,
        },
        "total_cost_usd": round(total_cost, 6),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", out_path)

    return band != "red"


def main() -> None:
    """Entry point."""
    _load_env()
    ok = asyncio.run(run_recon())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
