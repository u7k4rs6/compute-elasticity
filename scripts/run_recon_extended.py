"""Phase 5 extended — full 47-problem reconnaissance.

The original 10-problem recon (run_recon.py) over-represented chemistry/biology
and returned a Yellow-band result (50%). This extended pass samples all 47
main-pilot problems at N=1 to get a representative, physics-inclusive estimate.

run_recon.py stays untouched in git history as evidence of the biased
measurement that prompted this extended pass.

Band thresholds (N=1 accuracy over 47 problems):
  Green:       0.25 ≤ acc ≤ 0.45  → proceed with locked grid {1,2,4,8,16,32,64}
  Yellow-low:  0.15 ≤ acc < 0.25  → proceed with flag (floor risk)
  Yellow-high: 0.45 < acc ≤ 0.55  → proceed with flag (consider N=128 extension)
  Red:         acc < 0.15 or acc > 0.55 → pivot

Usage:
    source .venv/bin/activate
    python scripts/run_recon_extended.py
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

_N_PROBLEMS: int = 47
_TEMPERATURE: float = 0.7
_QUICK_RECON_ACCURACY: float = 0.50  # original 10-problem result

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
    return json.loads(labels_path.read_text())["gate_problems"]


def _select_main_pilot_problems(
    all_problems: list[Any],
    gate_ids: list[str],
    locked_ids: list[str],
) -> list[Any]:
    """Return the 47 main-pilot problems: locked 50 minus 3 Gate-1, sorted by ID."""
    locked = set(locked_ids)
    gate = set(gate_ids)
    problems = [p for p in all_problems if p.id in locked and p.id not in gate]
    problems.sort(key=lambda p: p.id)
    return problems


def _has_sample_at_temperature(
    samples_dir: Path, problem_id: str, temperature: float
) -> bool:
    """Return True if a sample at the given temperature already exists on disk."""
    path = samples_dir / f"{problem_id}.jsonl"
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if abs(obj.get("temperature", -1) - temperature) < 1e-6:
                return True
        except json.JSONDecodeError:
            continue
    return False


def _read_first_sample_at_temperature(
    samples_dir: Path, problem_id: str, temperature: float
) -> dict[str, Any] | None:
    """Return the first sample dict matching the given temperature."""
    path = samples_dir / f"{problem_id}.jsonl"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if abs(obj.get("temperature", -1) - temperature) < 1e-6:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _band(accuracy: float) -> tuple[str, str]:
    """Return (band, action) string pair for a given accuracy."""
    if _GREEN_LOW <= accuracy <= _GREEN_HIGH:
        return "green", "proceed"
    if _YELLOW_LOW <= accuracy < _GREEN_LOW:
        return "yellow_low", "proceed_with_flag"
    if _GREEN_HIGH < accuracy <= _YELLOW_HIGH:
        return "yellow_high", "proceed_with_flag"
    return "red", "pivot"


async def run_recon_extended() -> bool:
    """Run extended recon over all 47 main-pilot problems; return True if not red."""
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

    # --- 1 & 2. Load IDs ---
    locked_ids = _load_locked_ids()
    gate_ids = _load_gate_problem_ids()
    logger.info("Locked IDs: %d. Gate-1 IDs: %s.", len(locked_ids), ", ".join(gate_ids))

    # --- 3. Select 47 main-pilot problems ---
    logger.info("Loading GPQA Diamond…")
    all_problems = load_gpqa_diamond()
    problems = _select_main_pilot_problems(all_problems, gate_ids, locked_ids)
    if len(problems) != _N_PROBLEMS:
        logger.error(
            "Expected %d problems, got %d. Check locked IDs and gate IDs.",
            _N_PROBLEMS,
            len(problems),
        )
        return False
    logger.info(
        "Selected %d main-pilot problems. Subject counts: %s",
        len(problems),
        {
            s: sum(1 for p in problems if p.subject == s)
            for s in ("physics", "chemistry", "biology")
        },
    )

    # --- 4. Ensure 1 sample per problem at T=0.7 ---
    try:
        client = TogetherClient()
    except Exception as exc:
        logger.error("Failed to initialise TogetherClient: %s", exc)
        return False

    new_input_tokens = 0
    new_output_tokens = 0
    n_reused = 0
    n_newly_sampled = 0
    reused_flags: dict[str, bool] = {}

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for problem in problems:
        already_exists = _has_sample_at_temperature(
            SAMPLES_DIR, problem.id, _TEMPERATURE
        )
        output_path = SAMPLES_DIR / f"{problem.id}.jsonl"
        new_samples = await sample_problem(
            problem=problem,
            n_total=1,
            client=client,
            output_path=output_path,
            temperature=_TEMPERATURE,
        )
        for s in new_samples:
            new_input_tokens += s.input_tokens
            new_output_tokens += s.output_tokens

        if already_exists or not new_samples:
            reused_flags[problem.id] = True
            n_reused += 1
            logger.info("Reused existing sample for %s.", problem.id)
        else:
            reused_flags[problem.id] = False
            n_newly_sampled += 1
            logger.info("Sampled new completion for %s.", problem.id)

    new_cost = (
        new_input_tokens * TOGETHER_INPUT_PRICE_PER_TOKEN
        + new_output_tokens * TOGETHER_OUTPUT_PRICE_PER_TOKEN
    )
    logger.info(
        "Sampling done. Reused: %d, New: %d. Cost: $%.4f (%d in / %d out tokens).",
        n_reused,
        n_newly_sampled,
        new_cost,
        new_input_tokens,
        new_output_tokens,
    )

    # --- 5. Score each problem ---
    per_problem: list[dict[str, Any]] = []
    n_correct = 0
    by_subject: dict[str, dict[str, int]] = {}

    for problem in problems:
        sample = _read_first_sample_at_temperature(
            SAMPLES_DIR, problem.id, _TEMPERATURE
        )
        if sample is None:
            logger.error("No sample found for %s — marking incorrect.", problem.id)
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

        per_problem.append(
            {
                "problem_id": problem.id,
                "subject": subj,
                "ground_truth": problem.ground_truth,
                "extracted_answer": extracted_answer or "UNPARSEABLE",
                "extraction_pass": extraction_pass,
                "correct": correct,
                "reused": reused_flags.get(problem.id, True),
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

    # --- 6 & 7. Compute accuracy and band ---
    accuracy = n_correct / _N_PROBLEMS
    band, action = _band(accuracy)

    by_subject_full = {
        s: {
            "n": by_subject.get(s, {}).get("n", 0),
            "correct": by_subject.get(s, {}).get("correct", 0),
            "accuracy": round(
                by_subject.get(s, {}).get("correct", 0)
                / by_subject.get(s, {}).get("n", 1),
                6,
            ),
        }
        for s in ("physics", "chemistry", "biology")
    }

    delta = accuracy - _QUICK_RECON_ACCURACY

    # --- 8. Print summary ---
    sep = "=" * 72
    print(f"\n{sep}")
    print("Extended Recon — All 47 Main-Pilot Problems")
    print(sep)

    for subject in ("physics", "chemistry", "biology"):
        subject_probs = [pp for pp in per_problem if pp["subject"] == subject]
        if not subject_probs:
            continue
        print(f"\n  [{subject.upper()}]")
        print(
            f"  {'Problem':<28}  {'GT':<4}  {'Ans':<4}  {'Pass':<5}  {'Reused':<7}  Result"
        )
        print(f"  {'-'*28}  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*7}  ------")
        for pp in subject_probs:
            print(
                f"  {pp['problem_id']:<28}  {pp['ground_truth']:<4}  "
                f"{pp['extracted_answer']:<4}  {pp['extraction_pass']:<5}  "
                f"{'yes' if pp['reused'] else 'no':<7}  "
                f"{'CORRECT' if pp['correct'] else 'incorrect'}"
            )

    print(f"\n{sep}")
    print("  Subject breakdown:")
    for subj in ("physics", "chemistry", "biology"):
        d = by_subject_full[subj]
        pct = d["correct"] / d["n"] * 100 if d["n"] else 0
        print(f"    {subj:<12}: {d['correct']:>2}/{d['n']:<2} = {pct:.1f}%")

    print()
    print("  Subject distribution (47 problems): ", end="")
    print(
        "  ".join(
            f"{s}={by_subject_full[s]['n']}"
            for s in ("physics", "chemistry", "biology")
        )
    )
    print()
    print(f"  Overall accuracy : {n_correct}/{_N_PROBLEMS} = {accuracy:.1%}")
    print(
        f"  Band thresholds  : "
        f"Red <{_YELLOW_LOW:.0%} | Yellow-low {_YELLOW_LOW:.0%}–{_GREEN_LOW:.0%} | "
        f"Green {_GREEN_LOW:.0%}–{_GREEN_HIGH:.0%} | "
        f"Yellow-high {_GREEN_HIGH:.0%}–{_YELLOW_HIGH:.0%} | Red >{_YELLOW_HIGH:.0%}"
    )
    print(f"  Band             : {band.upper()} → {action}")
    print()
    sign = "+" if delta >= 0 else ""
    print(f"  Original 10-problem recon : {_QUICK_RECON_ACCURACY:.0%} (Yellow)")
    print(
        f"  Extended 47-problem recon : {accuracy:.0%} ({band.upper()})  "
        f"[delta {sign}{delta:.0%}]"
    )
    print()
    print(f"  Samples reused / new      : {n_reused} reused / {n_newly_sampled} new")
    print(f"  New sampling cost         : ${new_cost:.4f}")
    print(sep)

    if band == "red":
        logger.error(
            "RED band: accuracy %.1f%% outside Yellow range. Pivot per PRD §5.",
            accuracy * 100,
        )
    elif band == "yellow_low":
        logger.warning(
            "YELLOW-LOW band: %.1f%% — floor risk. Proceed with flag.", accuracy * 100
        )
    elif band == "yellow_high":
        logger.warning(
            "YELLOW-HIGH band: %.1f%% — consider N=128 extension. Proceed with flag.",
            accuracy * 100,
        )
    else:
        logger.info("GREEN band: %.1f%%. Proceed to Phase 6.", accuracy * 100)

    # --- 9. Write output JSON ---
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "recon_extended_results.json"
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "day_0_recon_extended",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": MODEL,
        "temperature": _TEMPERATURE,
        "n_problems": _N_PROBLEMS,
        "n_reused_from_prior": n_reused,
        "n_newly_sampled": n_newly_sampled,
        "per_problem": per_problem,
        "summary": {
            "n_correct": n_correct,
            "accuracy": round(accuracy, 6),
            "by_subject": by_subject_full,
            "band": band,
            "band_action": action,
            "comparison_to_quick_recon": {
                "quick_accuracy": _QUICK_RECON_ACCURACY,
                "extended_accuracy": round(accuracy, 6),
                "delta": round(delta, 6),
            },
        },
        "new_sampling_cost_usd": round(new_cost, 6),
        "new_input_tokens": new_input_tokens,
        "new_output_tokens": new_output_tokens,
    }
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", out_path)

    return band != "red"


def main() -> None:
    """Entry point."""
    _load_env()
    ok = asyncio.run(run_recon_extended())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
