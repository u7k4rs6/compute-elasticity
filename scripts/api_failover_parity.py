"""Phase 3 — API failover parity check.

Sends identical 5 problems × N=4 completions to Together AI and DeepInfra,
then compares response distributions to confirm both providers are usable
as primary/fallback.

Fail criterion: if one provider systematically returns empty responses or
error rates differ by >20 percentage points, the check fails and you should
pick a different fallback provider.

Usage:
    source .venv/bin/activate
    python scripts/api_failover_parity.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


# 5 fixed GPQA-style questions NOT in the 50 sample (indices chosen outside
# the locked set; these are used only for parity checking, not for data).
_PARITY_PROMPTS = [
    (
        "You are a helpful AI assistant solving a graduate-level multiple-choice question.\n\n"
        "Question: Which of the following correctly describes a key property of quantum entanglement?\n\n"
        "Options:\n"
        "A) Entangled particles can communicate faster than light\n"
        "B) Measurement of one particle instantaneously determines the state of its partner\n"
        "C) Entanglement only occurs in macroscopic systems\n"
        "D) Entangled particles must be in the same location\n\n"
        'Think through this step by step, then provide your final answer in the format "Answer: X".\n'
    ),
    (
        "You are a helpful AI assistant solving a graduate-level multiple-choice question.\n\n"
        "Question: What is the primary mechanism by which CRISPR-Cas9 introduces targeted mutations?\n\n"
        "Options:\n"
        "A) RNA interference silencing\n"
        "B) RNA-guided DNA cleavage followed by error-prone repair\n"
        "C) Transposon insertion at guide RNA binding sites\n"
        "D) Methylation-directed recombination\n\n"
        'Think through this step by step, then provide your final answer in the format "Answer: X".\n'
    ),
    (
        "You are a helpful AI assistant solving a graduate-level multiple-choice question.\n\n"
        "Question: In organic chemistry, which reaction involves the addition of a carbene to an alkene?\n\n"
        "Options:\n"
        "A) Diels-Alder cycloaddition\n"
        "B) Cyclopropanation\n"
        "C) Grignard addition\n"
        "D) Aldol condensation\n\n"
        'Think through this step by step, then provide your final answer in the format "Answer: X".\n'
    ),
    (
        "You are a helpful AI assistant solving a graduate-level multiple-choice question.\n\n"
        "Question: Which statement about black hole thermodynamics is correct?\n\n"
        "Options:\n"
        "A) Black holes violate the second law of thermodynamics\n"
        "B) A black hole's entropy is proportional to its event horizon area\n"
        "C) Black holes have zero temperature by definition\n"
        "D) Black hole entropy is proportional to its volume\n\n"
        'Think through this step by step, then provide your final answer in the format "Answer: X".\n'
    ),
    (
        "You are a helpful AI assistant solving a graduate-level multiple-choice question.\n\n"
        "Question: What distinguishes competitive inhibition from non-competitive inhibition of enzymes?\n\n"
        "Options:\n"
        "A) Competitive inhibitors bind the active site; non-competitive inhibitors bind an allosteric site\n"
        "B) Only non-competitive inhibition is reversible\n"
        "C) Competitive inhibitors change the enzyme's Vmax\n"
        "D) Non-competitive inhibition increases the apparent Km\n\n"
        'Think through this step by step, then provide your final answer in the format "Answer: X".\n'
    ),
]

N_PER_PROBLEM = 4


async def _collect(client, provider_name: str, prompts: list[str]) -> list[dict]:
    """Collect N_PER_PROBLEM completions per prompt from one provider."""
    results = []
    for p_idx, prompt in enumerate(prompts):
        for s_idx in range(N_PER_PROBLEM):
            seed_hex = f"{(p_idx * 100 + s_idx):016x}"
            try:
                c = await client.complete(
                    prompt=prompt,
                    temperature=0.7,
                    seed_hex=seed_hex,
                )
                results.append(
                    {
                        "provider": provider_name,
                        "prompt_idx": p_idx,
                        "sample_idx": s_idx,
                        "ok": True,
                        "length": len(c.text),
                        "output_tokens": c.output_tokens,
                        "latency_ms": c.latency_ms,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "provider": provider_name,
                        "prompt_idx": p_idx,
                        "sample_idx": s_idx,
                        "ok": False,
                        "error": str(exc),
                    }
                )
    return results


def _report(results: list[dict], provider: str) -> tuple[float, float]:
    """Return (success_rate, mean_latency_ms) for a provider."""
    mine = [r for r in results if r["provider"] == provider]
    ok = [r for r in mine if r["ok"]]
    success_rate = len(ok) / len(mine) if mine else 0.0
    mean_latency = sum(r["latency_ms"] for r in ok) / len(ok) if ok else float("nan")
    return success_rate, mean_latency


async def run_parity() -> bool:
    """Run parity check; return True if both providers pass."""
    from pilot.sampling import ConfigError, DeepInfraClient, TogetherClient

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    together_ok = True
    deepinfra_ok = True

    try:
        together = TogetherClient()
    except ConfigError as exc:
        logger.error("Together AI not configured: %s", exc)
        together_ok = False
        together = None

    try:
        deepinfra = DeepInfraClient()
    except ConfigError as exc:
        logger.error("DeepInfra not configured: %s", exc)
        deepinfra_ok = False
        deepinfra = None

    if not together_ok and not deepinfra_ok:
        return False

    results: list[dict] = []

    if together is not None:
        logger.info(
            "Collecting %d completions from Together AI…",
            len(_PARITY_PROMPTS) * N_PER_PROBLEM,
        )
        results += await _collect(together, "together_ai", _PARITY_PROMPTS)

    if deepinfra is not None:
        logger.info(
            "Collecting %d completions from DeepInfra…",
            len(_PARITY_PROMPTS) * N_PER_PROBLEM,
        )
        results += await _collect(deepinfra, "deepinfra", _PARITY_PROMPTS)

    # Print report
    print("\n=== Parity Report ===")
    for provider in ("together_ai", "deepinfra"):
        sr, ml = _report(results, provider)
        status = "OK" if sr >= 0.8 else "FAIL"
        print(
            f"  {provider:20s}  success={sr:.0%}  mean_latency={ml:.0f}ms  [{status}]"
        )

    # Check divergence
    ta_sr, _ = _report(results, "together_ai")
    di_sr, _ = _report(results, "deepinfra")

    all_ok = True

    if together is not None and ta_sr < 0.8:
        logger.error(
            "Together AI success rate %.0f%% < 80%% — primary provider failing",
            ta_sr * 100,
        )
        all_ok = False

    if deepinfra is not None and di_sr < 0.8:
        logger.error(
            "DeepInfra success rate %.0f%% < 80%% — fallback provider failing",
            di_sr * 100,
        )
        all_ok = False

    if together is not None and deepinfra is not None and abs(ta_sr - di_sr) > 0.2:
        logger.error(
            "Provider success rates diverge by >20pp (%.0f%% vs %.0f%%) — check fallback",
            ta_sr * 100,
            di_sr * 100,
        )
        all_ok = False

    print(json.dumps({"results_summary": results[:4]}, indent=2))
    return all_ok


def main() -> None:
    _load_env()
    ok = asyncio.run(run_parity())
    if ok:
        print("\nParity check PASSED.")
        sys.exit(0)
    else:
        print("\nParity check FAILED — see logs. Consider switching fallback provider.")
        sys.exit(1)


if __name__ == "__main__":
    main()
