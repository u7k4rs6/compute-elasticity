[![arXiv](https://img.shields.io/badge/arXiv-2608.11403-b31b1b.svg)](https://arxiv.org/abs/2608.11403)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2608.11403-blue.svg)](https://doi.org/10.48550/arXiv.2608.11403)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

# When Self-Consistency Backfires

**Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs**

Utkarsh Bahuguna, Scaler School of Technology

Accepted at the **COLM 2026 Workshop on Efficient Reasoning**.
Paper: [arXiv:2608.11403](https://arxiv.org/abs/2608.11403)

---

## What this is

Self-consistency samples N chains of thought and returns the plurality answer. It is widely treated as a low-risk way to spend inference-time compute: sample more, vote, do at least as well.

On a hard benchmark it is not. Across all 198 GPQA Diamond problems, majority voting at N=64 **reduces** per-problem accuracy relative to a single sample on **56.6%** of problems for Qwen2.5-7B and **65.7%** for Llama-3-8B. Aggregate accuracy barely moves (Qwen 0.342 to 0.369), which is exactly what hides the per-problem harm underneath.

This repository contains the full pipeline, the pre-registration, the analysis, and the falsification suites.

## Headline results

| | Qwen2.5-7B | Llama-3-8B |
|---|---|---|
| Backfire rate (pooled, n=198) | 56.6% [49.5, 63.6] | 65.7% [59.1, 71.7] |
| MV accuracy, N=1 → N=64 | 0.342 → 0.369 | 0.273 → 0.313 |
| Grid oracle upper bound | 0.482 | 0.439 |
| Agreement gate (k=8, τ=0.75) | 0.368 | 0.312 |
| Entropy gate (confirmatory 151) | 0.318 | 0.306 |

Two cheap verifier-free gates were tested to see whether the oracle headroom is reachable without ground truth. Neither moves accuracy more than 0.002 from fixed-budget voting at N=64. The mechanism is direct: confidence does not track correctness on these problems. In Qwen's highest-agreement bin the plurality answer is correct 52.5% of the time, and Llama's highest-agreement bin is *less* accurate than its lowest.

## Pre-registered confirmatory results

Hypotheses and thresholds were locked and git-tagged before any confirmatory analysis, at tag `backfire-prereg-v1.0`. 47 problems are exploratory (hypotheses were generated from them); the remaining 151 are the confirmatory test set. PASS/FAIL is decided on the confirmatory set only.

| H | Prediction (both models) | Qwen2.5-7B | Llama-3-8B | Result |
|---|---|---|---|---|
| PH1 | backfire rate ≥ 33% | 60.3% [53.0, 68.2] | 65.6% [58.3, 73.5] | **PASS** |
| PH2 | agree gate capture ≤ 10% | 0.8% | −1.6% | **PASS** |
| PH3 | top-agree-bin acc. ≤ 70% | 51.2% (n=43) | 14.3% (n=21) | **PASS** |
| PH4 | entropy gate capture ≤ 10% | 0.5% | 0.9% | **PASS** |

Captures in this table are measured against the binary {N=1, N=64} oracle relative to an MV_acc(64) baseline. The grid-oracle captures quoted in the paper's Section 4.3 use a more generous ceiling and are reported separately as post-hoc analysis.

## Setup

Both models served on Together AI:

- `Qwen2.5-7B-Instruct-Turbo`
- `Meta-Llama-3-8B-Instruct-Lite`

N=64 samples per problem, temperature 0.7, single locked prompt template verified byte-identical across all runs by SHA-256. Five-pass answer extraction; parse rate 99.5% (Qwen) and 98.6% (Llama). Confidence intervals are problem-level bootstrap, 1000 iterations, seed 42.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in API keys in .env

pytest tests/ -v --tb=short
```

Activate the virtualenv before running anything. `pytest-asyncio` is installed there and not in system Python, so running `python3 -m pytest` outside the venv produces 15 collection failures reading `async def functions are not natively supported`. Those are not broken tests.

## Reproducibility

- **Pre-registration:** [`preregistration_backfire.md`](preregistration_backfire.md), locked before data collection, timestamped at tag `backfire-prereg-v1.0`.
- **Confirmatory verdicts:** PH1–PH4 PASS/FAIL is computed by `scripts/run_phase13_analysis.py` and stored in `outputs/gate_model2/confirmatory_results.json`. This is the source for Table 1.
- **Falsification suite:** `pytest tests/falsification_backfire.py -v` reads that file and asserts each verdict, per model, along with the stored thresholds themselves, so a regenerated result with an edited threshold fails rather than silently validating against a moved line. A pytest failure means a pre-registered hypothesis is falsified. Note this asserts stored values against thresholds; it does not recompute the analysis.
- **Truncation invariance:** `scripts/verify_truncation_invariance.py`. Llama has exactly 64 stored samples per problem; for Qwen the 47 exploratory problems carry 65 to 72 from earlier sampling runs. Reclassifying every problem from its first 64 samples leaves the backfire count unchanged at 112 of 198, with no problem crossing the boundary.
- **Pilot suite:** `pytest tests/falsification.py -v` asserts the pilot's H1–H6 against `outputs/hypothesis_results.json`. It predates this paper and does not test PH1–PH4. Note it exits non-zero by design: H3 was genuinely falsified in the pilot, and the suite reports that as a test failure.

## Known limitations

Stated in full in the paper. The short version:

- Llama's single-sample accuracy (0.273) sits just above the 0.25 random baseline, so Qwen is the primary demonstration and Llama corroborates the direction.
- One benchmark, two small non-reasoning models. Whether backfire persists for reasoning-native models is the central open question.
- The entropy gate's threshold was selected in-sample on the confirmatory 151, so its reported capture is optimistic.
- The pipeline retained only a mean entropy scalar rather than per-token log-probability arrays, which rules out final-answer margin analysis.

## Project history

This repository began as a pilot study on **compute elasticity** (repo formerly `compute-elasticity`), fitting accuracy-versus-compute curves for a single model on a 47-problem subset. That pilot returned GO, and its findings redirected the work: the falsification of its H3, which found that single-sample token entropy outperformed 4-sample embedding diversity as a predictor, is what motivated testing entropy as a routing gate here.

The pilot's writeup is retained at [`PILOT_WRITEUP.md`](PILOT_WRITEUP.md) and its phase tags remain in the repository:

`phase-0-complete` → `phase-1-complete` → `pre-pilot-v6.0` → `pre-pilot-v6.0.1-single-provider` → `pre-pilot-v6.0.2-turbo-variant` → `phase-4-complete` → ... → `phase-9-complete`

`pre-pilot-v6.0` is the **pilot's** pre-registration and is not the pre-registration for this paper. The two are distinct commits: `pre-pilot-v6.0` (2026-05-19) and `backfire-prereg-v1.0` (2026-06-05).

## Citation

```bibtex
@misc{bahuguna2026backfires,
  title         = {When Self-Consistency Backfires: Majority Vote Hurts the
                   Majority of Hard Science Problems for Small LLMs},
  author        = {Utkarsh Bahuguna},
  year          = {2026},
  eprint        = {2608.11403},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2608.11403},
  note          = {Accepted at the COLM 2026 Workshop on Efficient Reasoning}
}
```

## License

Apache 2.0, see [`LICENSE`](LICENSE).
