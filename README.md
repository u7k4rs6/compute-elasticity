# Compute Elasticity in LLM Reasoning

A pilot study measuring how **Qwen2.5-7B-Instruct-Turbo** accuracy on **GPQA Diamond** responds to inference-time compute via best-of-N sampling at N ∈ {1, 2, 4, 8, 16, 32, 64}.

See `PRD.md` for the full specification and `CLAUDE.md` for repository conventions.

## Quick start

```bash
cp .env.example .env
# Fill in API keys in .env

pip install -r requirements.txt
pytest tests/ -v --tb=short
```

## Phase status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Bootstrap | ✓ |
| 1 | Pure-Python modules + unit tests | ✓ |
| 2 | Pre-registration lock-in | ✓ |
| 3 | API smoke tests | ✓ |
| 4 | Gate -1: Embedder validation | ✓ |
| 5 | Day 0 reconnaissance | ✓ |
| 6 | Main pilot sampling | ✓ |
| 7 | Temperature side test | ✓ |
| 8 | Fitting | ✓ |
| 9 | Pass 1: confirmatory analysis | ✓ |
| 10 | Pass 2: exploratory + go/no-go | ✓ |

## Pilot Results

**Overall verdict: GO** — 4 PASS, 1 FAIL, 1 DEFERRED across H1–H6.

Full writeup: [`PILOT_WRITEUP.md`](PILOT_WRITEUP.md). Full results: [`outputs/hypothesis_results.json`](outputs/hypothesis_results.json)

| H | Statement | Verdict | Measured |
|---|-----------|---------|----------|
| H1 | R(c) curves fittable above noise | **PASS** | Median residual SE = 0.0107 (threshold < 0.10) |
| H2 | Multiple curve families win — mode diversity | **PASS** | 3 families with mean BIC weight ≥ 0.10; 34% of problems have ≥2 close families |
| H3 | Embedding diversity > entropy for elasticity prediction | **FAIL** | AUC(diversity) = 0.524, AUC(entropy) = 0.650; entropy is the stronger predictor |
| H4 | Curve distribution differs by domain | DEFERRED | Deferred to full multi-model study |
| H5 | Non-trivial unimodal subset (≥5% of problems) | **PASS** | 21/47 problems (44.7%) — note: only 1 has a genuine interior peak; see [`outputs/unimodal_sensitivity.json`](outputs/unimodal_sensitivity.json) |
| H6 | Curve params stable across temperature | **PASS** | Mean 95% bootstrap CI overlap = 1.000 across T ∈ {0.3, 0.7, 1.0} |

**Headline findings:**

- **H3 reframing.** Embedding diversity at N=4 does not outperform per-token entropy at N=1 as an elasticity predictor. Entropy (AUC=0.650) beats diversity (AUC=0.524) by a margin of 0.127, the reverse of H3's direction. This is a clean falsification: the single-sample entropy baseline is a better signal than the 4-sample embedding spread. This becomes a key finding for the full study's feature design.

- **H6 temperature invariance.** 95% bootstrap CIs for fitted curve parameters overlap perfectly (rate = 1.000) across all three temperature pairs for all 10 side-test problems. Curve shape is effectively temperature-invariant over {0.3, 0.7, 1.0} for this model and benchmark.

- **H5 interpretation.** Of the 21 unimodal BIC winners, only 1 problem (gpqa_diamond_0026, c* ≈ 11) shows a genuine interior accuracy peak. The remaining 20 are fitting artifacts — either decaying (peak below the data grid, c* < 4) or saturating (peak beyond grid, c* > 56). See [`outputs/unimodal_sensitivity.json`](outputs/unimodal_sensitivity.json) for the full regime breakdown.

**Total API spend: ~$0.60** of the $15.00 budget. The original PRD estimate ($10–15) overstated actual usage by ~25× — outputs were shorter than expected, and the Pass-5 LLM scorer never had to fire because regex extraction held above 99%.

## Reproducibility

- **Pre-registration:** [`preregistration.md`](preregistration.md) — hypothesis thresholds and methodology locked before data collection, timestamped at tag `pre-pilot-v6.0`.
- **Falsification suite:** `pytest tests/falsification.py -v` — reads `outputs/hypothesis_results.json` and asserts each verdict. A pytest failure = hypothesis falsified.
- **Phase tags** (in order):
  `phase-0-complete` → `phase-1-complete` → `pre-pilot-v6.0` → `pre-pilot-v6.0.1-single-provider` → `pre-pilot-v6.0.2-turbo-variant` → `phase-4-complete` → `phase-5-complete` → `phase-6-complete` → `phase-7-complete` → `phase-8-complete` → `phase-9-complete`

  > Phase 2 corresponds to the three `pre-pilot-v6.0.*` tags. Phase 3 (API smoke tests) was committed inline and has no dedicated tag.

## License

Apache 2.0 — see `LICENSE`.
