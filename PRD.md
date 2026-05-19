# Compute Elasticity in LLM Reasoning — Project Requirements Document

**Version:** v6.0 (Claude Code edition)
**Derived from:** v5.5 research PRD (no methodology changes; restructured for agentic execution)
**Status:** Locked. Pre-pilot.
**Primary reader:** Claude Code
**Secondary reader:** Human developer reviewing PRs
**Companion file:** `CLAUDE.md` (repository conventions, code style, anti-patterns)

---

## 0. Quick reference

Build a Python pilot that measures how Qwen2.5-7B-Instruct's accuracy on GPQA Diamond responds to inference-time compute via best-of-N sampling at N ∈ {1, 2, 4, 8, 16, 32, 64}. Fit 5 parametric curve families per problem with BIC-weighted soft model selection. Test 6 pre-registered hypotheses (H1–H6) against locked thresholds. Produce a go/no-go decision for a larger multi-model study.

The work splits into **11 phases**. Phases 0–3 are pure code (no API spend, ~zero risk). Phase 4 onward incurs API cost — total expected pilot cost is ~$15.

**Stack:** Python 3.10+, `scipy.optimize`, `sentence-transformers`, async `httpx`, `pytest`, Together AI API (single provider; DeepInfra deferred to full study).

**Hard rules** (full list in `CLAUDE.md`):
- Locked prompt template: never modify after Day 0 (Phase 5)
- Locked hypotheses H1–H6: never add new confirmatory tests mid-pilot
- Two analysis passes maximum: Pass 1 confirmatory, Pass 2 exploratory, no Pass 3
- All scope expansion temptations → `TODO_full_study.md`, never implemented in pilot

---

## 1. Goal hierarchy

Three contribution tiers. The paper survives if only Primary lands.

| Tier | Contribution | What "landing" means |
|---|---|---|
| **Primary** | Characterization | H1 + H2 pass: parametric framework works, mode diversity exists |
| **Secondary** | Predictability | H3 passes: embedding diversity predicts elasticity |
| **Tertiary** | Allocator (full study, NOT pilot) | Pareto-frontier improvement on cost-vs-accuracy |

Never claim uniform dominance. Conditional claims only.

---

## 2. Scope

### In scope (pilot)
- **Model:** `Qwen/Qwen2.5-7B-Instruct` only
- **Benchmark:** GPQA Diamond, 50 problems stratified by subject (47 main + 3 Gate-1)
- **Strategy:** best-of-N sampling only
- **Compute axis:** N ∈ {1, 2, 4, 8, 16, 32, 64}
- **Temperature:** 0.7 (main), {0.3, 0.7, 1.0} (side test)
- **Primary predictor feature:** CoT embedding diversity at N=4 (`bge-small-en-v1.5`)
- **Baseline predictor:** per-step entropy at N=1
- **Fitting:** MLE per problem, 5 families, BIC weights
- **Hypotheses:** H1, H2, H3, H5, H6 in pilot; H4 deferred

### Out of scope (do NOT build)
- Allocator / Pareto routing (full study)
- Tree search, reflection, verifier-guided decoding, self-consistency vote weighting
- Additional models / benchmarks / strategies
- Hierarchical Bayesian fitting (MLE only)
- Any predictor feature beyond embedding diversity + entropy baseline
- Pass 3 of analysis
- Prompt engineering beyond §6.1 locked template (sole exception: Phase 3 if extraction <95%)

Everything tempting → `TODO_full_study.md`.

---

## 3. Locked hypotheses

| H | Statement | Operational test | Pass | Fail |
|---|---|---|---|---|
| **H1** | R(c) curves fittable above noise | Median residual SE across 47 problems, in accuracy units | <0.10 | ≥0.15 |
| **H2** | Multiple curve families win — mode diversity | (a) ≥2 families with mean BIC weight ≥0.10 across 47 problems, OR (b) ≥30% problems have ≥2 families within 2 BIC of best | Either holds | Neither |
| **H3** | Embedding diversity (N=4) > entropy (N=1) for elasticity prediction | AUC predicting above-median fitted elasticity over c ∈ [8, 64] | AUC ≥0.60 AND ΔAUC ≥0.03 | AUC <0.55 OR ΔAUC ≤0 |
| **H4** | Curve dist. differs by domain | Deferred to full study | — | — |
| **H5** | Non-trivial unimodal subset (degradation regime) | Fraction of problems where unimodal wins BIC selection | ≥5% | 0% |
| **H6** | Curve params stable across temperature | Bootstrap-CI overlap rate for fitted params across T ∈ {0.3, 0.7, 1.0} on side-test subset | ≥70% | <50% |

Thresholds hardcoded in `pilot/config.py` as `HYPOTHESIS_THRESHOLDS`. Never modify mid-pilot.

---

## 4. Locked specifications

### 4.1 Prompt template

```
You are a helpful AI assistant solving a graduate-level multiple-choice question.

Question: {question_text}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Think through this step by step, then provide your final answer in the format
"Answer: X" where X is one of A, B, C, or D.
```

`pilot/prompts.py` must:
1. Store this template verbatim including trailing newline rules.
2. Compute and expose `PROMPT_TEMPLATE_HASH` as `sha256(template.encode("utf-8")).hexdigest()`.
3. Refuse to render the prompt if the live hash doesn't match the stored constant.

### 4.2 Pass 5 LLM-scorer prompt

```
You are evaluating a student's answer to a multiple-choice question.

Question: {question_text}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Correct answer: {ground_truth}

Student's response:
{full_response}

Did the student arrive at the correct answer? Respond with exactly one word:
- "CORRECT" if the student's final reasoning concludes with the correct answer
- "INCORRECT" if the student's final reasoning concludes with a wrong answer
- "TRULY_UNPARSEABLE" if no final answer can be determined from the response
```

Scoring rule: 1 if `CORRECT`, else 0. Hash logged identically to main prompt.

### 4.3 5-pass scoring pipeline

```python
# pilot/scoring.py — apply passes in order, return on first match
PASS_1 = r"(?i)answer[:\s]+([A-D])\b"           # last 200 chars of response
PASS_2 = r"\\boxed\{([A-D])\}"                  # anywhere in response
PASS_3 = r"\b([A-D])\b"                         # last non-empty line
PASS_4 = r"\b([A-D])\b"                         # last 500 chars (loose)
PASS_5 = "<LLM scorer; only invoked if 1-4 fail>"
```

All 5 fail → score conservatively as incorrect.

### 4.4 The 5 curve families

All families fit on `c` = compute budget (in samples N) → `R` ∈ [0, 1] = accuracy.

| Family | Form | Free params | Bounds |
|---|---|---|---|
| `constant` | `R(c) = p` | `p` | `p ∈ [0, 1]` |
| `logistic` | `R(c) = L / (1 + exp(-k(c - c0)))` | `L, k, c0` | `L ∈ [0, 1]`, `k > 0`, `c0 > 0` |
| `gompertz` | `R(c) = L · exp(-b · exp(-k·c))` | `L, b, k` | `L ∈ [0, 1]`, `b > 0`, `k > 0` |
| `shifted_logistic` | logistic with `c0` large + small floor `f` | `L, k, c0, f` | `L ∈ [0, 1]`, `k > 0`, `c0 > 0`, `f ∈ [0, 0.25]` |
| `unimodal` | Gaussian-like peak: `R(c) = A · exp(-((c - c*)² / 2σ²)) + b` | `A, c*, σ, b` | `A ∈ [0, 1]`, `c* > 0`, `σ > 0`, `b ∈ [0, 1]` |

Estimator: `scipy.optimize.curve_fit` with bounded parameters, sensible initial guesses (see `pilot/fitting.py` docstrings).

BIC: `BIC = k·log(n) - 2·log(L)` where `n = 7` (number of N values).
BIC weights: `w_i = exp(-ΔBIC_i / 2) / Σ_j exp(-ΔBIC_j / 2)`.

### 4.5 Sample JSONL schema

One sample per line in `outputs/samples/<problem_id>.jsonl`:

```json
{
  "schema_version": "v6.0-pilot",
  "problem_id": "gpqa_diamond_<idx>",
  "subject": "physics|chemistry|biology",
  "ground_truth": "A",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "provider": "together_ai",
  "temperature": 0.7,
  "sample_idx": 0,
  "n_total_in_batch": 64,
  "seed_hex": "a1b2c3d4e5f6g7h8",
  "prompt_template_hash": "sha256:<hex>",
  "full_response": "<model output>",
  "extracted_answer": "A|B|C|D|UNPARSEABLE",
  "extraction_pass": 1,
  "correct": true,
  "input_tokens": 1234,
  "output_tokens": 2345,
  "latency_ms": 12345,
  "timestamp": "2026-05-20T14:30:00Z",
  "api_metadata": {}
}
```

`pilot/validation.py` enforces schema on read.

---

## 5. Phase plan

Each phase has: **Goal**, **Inputs**, **Deliverables**, **Acceptance test**, **Stop conditions**.

Acceptance tests are pytest commands. Phase N cannot start until Phase N-1's acceptance test passes. Tag git on every phase completion: `git tag phase-N-complete`.

---

### Phase 0 — Bootstrap

**Goal:** Empty but runnable repo skeleton.

**Deliverables:**
- Directory layout per §7
- `requirements.txt` with pinned major versions
- `.env.example` listing required env vars (no real keys)
- `.gitignore` excluding `.env`, `outputs/`, `__pycache__`, `.pytest_cache`
- `README.md` (brief) and `LICENSE` (Apache 2.0)
- `pyproject.toml` with `ruff` and `black` config
- `pytest.ini` or `pyproject.toml [tool.pytest.ini_options]`

**Acceptance test:**
```bash
pip install -r requirements.txt
pytest --collect-only   # zero tests is fine, must not error
ruff check .             # must pass on empty modules
```

**Stop conditions:** none (this phase cannot fail meaningfully).

---

### Phase 1 — Pure-Python modules + unit tests (no network)

**Goal:** All `pilot/` modules implemented and unit-tested on synthetic data. No API calls in this phase.

**Deliverables (one per task):**

#### T1.1 — `pilot/config.py`
Constants: model name, provider URLs, N values, temperature, paths, `HYPOTHESIS_THRESHOLDS` dict. All locked values in one place.

#### T1.2 — `pilot/prompts.py`
- `MAIN_PROMPT_TEMPLATE`, `PASS5_SCORER_TEMPLATE` (verbatim from §4.1, §4.2)
- `PROMPT_TEMPLATE_HASH`, `PASS5_SCORER_HASH` constants
- `render_prompt(question, options)`, `render_pass5_prompt(...)` functions
- Hash-mismatch raises `RuntimeError`

**Test:** `tests/test_prompts.py` — verifies hashes match expected values, render functions produce expected strings on fixture inputs.

#### T1.3 — `pilot/fitting.py`
Implements 5 families with signatures:
```python
def fit_constant(c: np.ndarray, R: np.ndarray) -> FitResult: ...
def fit_logistic(c: np.ndarray, R: np.ndarray) -> FitResult: ...
def fit_gompertz(c: np.ndarray, R: np.ndarray) -> FitResult: ...
def fit_shifted_logistic(c: np.ndarray, R: np.ndarray) -> FitResult: ...
def fit_unimodal(c: np.ndarray, R: np.ndarray) -> FitResult: ...

def fit_all_families(c, R) -> Dict[str, FitResult]: ...
def bic_weights(fit_results: Dict[str, FitResult]) -> Dict[str, float]: ...
```

`FitResult` dataclass: `params`, `cov`, `bic`, `residual_se`, `converged: bool`.

**Test:** `tests/test_fitting.py` — synthetic recovery. For each family, generate noisy data from known params, fit, assert recovered params within tolerance (e.g., MSE < 0.01 on params). Cross-fit each family to data from other families and assert BIC selects the correct one ≥80% of trials.

#### T1.4 — `pilot/scoring.py`
- `extract_answer(response: str) -> tuple[str | None, int]` — returns `(answer, pass_number)` or `(None, 5)` after all regex passes
- `pass5_score(question, options, ground_truth, full_response, api_client) -> bool` — invokes LLM scorer
- `score_sample(sample_dict, ground_truth, api_client=None) -> ScoredSample`

**Test:** `tests/test_scoring.py` — handcrafted cases:
- `"...Answer: B"` → pass 1, "B"
- `"...\\boxed{C}..."` → pass 2, "C"
- `"...the answer is A.\n"` ending → pass 3
- Edge cases: multiple A-D letters, no answer, ambiguous endings

#### T1.5 — `pilot/diversity.py`
- `compute_diversity(traces: list[str], embedder) -> float` — mean pairwise cosine distance
- `truncate_trace(trace: str, max_tokens: int = 512) -> tuple[str, bool]` — strip `Answer: X` line, truncate at last sentence boundary; returns `(truncated_text, was_truncated)`

**Test:** `tests/test_diversity.py`
- Identical traces → distance ≈ 0
- Random uncorrelated traces → distance > 0.3
- Truncation never splits mid-word

#### T1.6 — `pilot/data_loader.py`
- `load_gpqa_diamond() -> list[Problem]`
- `stratified_sample(problems, n=50, seed=42) -> list[Problem]`
- `Problem` dataclass with `id, subject, question, options, ground_truth`

**Test:** `tests/test_data_loader.py` — stratified sample preserves subject proportions within ±1 problem; same seed → same sample; problems have all required fields.

#### T1.7 — `pilot/analysis.py`
- `block_bootstrap(stat_fn, data, n_resamples=1000, ci=0.95) -> tuple[float, float, float]`
- `compute_auc(scores, labels) -> float`
- `elasticity_at(c, fit_result) -> float` — dR/dc evaluated analytically per family
- `fitted_elasticity_summary(fit_result, c_range=(8, 64)) -> float` — mean elasticity over range

**Test:** `tests/test_analysis.py` — bootstrap CI on known-distribution data covers true value at correct rate; AUC on linearly separable synthetic data → 1.0; AUC on random → ~0.5.

#### T1.8 — `pilot/validation.py`
- `validate_sample(line: dict) -> list[str]` — returns list of issues, empty if valid
- `scan_outputs(dir: Path) -> ValidationReport` — full directory scan: duplicates, token mismatches, missing fields, suspicious latencies

**Test:** `tests/test_validation.py` — planted anomalies (duplicate seed_hex, missing fields, negative tokens) all caught.

#### T1.9 — `pilot/sampling.py` (no live calls yet — interface only)
- `class APIClient` with `async def complete(prompt, temperature, seed_hex) -> Completion`
- `class TogetherClient(APIClient)` — implementation
- `async def sample_problem(problem, n_total, client, output_path) -> list[Sample]` — idempotent (reads existing file, only generates new samples)
- Backoff: exponential, max 5 retries, jitter

**Test:** `tests/test_sampling.py` — mock HTTP layer; verify retry logic, idempotency (run twice, second run is no-op), schema correctness of written lines.

#### T1.10 — `tests/falsification.py` (skeleton)
Pytest-style suite reading `outputs/fits/` and `outputs/diversity/`, executing each hypothesis test against `HYPOTHESIS_THRESHOLDS`. Skeleton can run against synthetic outputs in Phase 1; runs for real in Phase 9.

**Acceptance test for Phase 1:**
```bash
pytest tests/ -v --tb=short
# Must show 100% pass on all tests above. falsification.py marks all as skipped
# (no real data yet) but must import cleanly.
ruff check pilot/ tests/
black --check pilot/ tests/
```

**Stop conditions:**
- Any curve family fails synthetic recovery → debug the fit before moving on; this is the single most important pre-pilot signal.
- BIC selection accuracy <80% in cross-family test → review BIC formula; do not proceed.

---

### Phase 2 — Pre-registration lock-in

**Goal:** Freeze the methodology in a git tag before any data is touched.

**Deliverables:**
- `preregistration.md` — distilled from PRD §1–§4 of this doc, plus §6 stop conditions. Includes the 50 stratified problem IDs as JSON.
- Hash log: prompt template hash, scorer prompt hash, problem-id list hash (all in `preregistration.md`).
- Pre-flight checklist (this PRD §8) ticked off.
- Git tag: `pre-pilot-v6.0`. **This tag is the timestamp proof for the paper.** Push to GitHub.

**Acceptance test:**
```bash
git tag --list pre-pilot-v6.0
# Tag must exist and be signed (gpg signing optional but recommended)
git show pre-pilot-v6.0 --stat
# Must show preregistration.md as part of the tagged commit
```

**Stop conditions:** if the pre-flight checklist has any unchecked box, do not tag.

---

### Phase 3 — API smoke tests

**Goal:** Verify API access works end-to-end at minimum cost (~$0.50) before committing to the main pilot.

**Deliverables:**
- `scripts/smoke_test.py` — sends 5 simple prompts to Together AI, prints latency and token counts.
- `scripts/extraction_rate_check.py` — runs 20 actual Qwen2.5-7B completions on GPQA-diamond-flavored questions (not from the 50 sample), applies passes 1–4, reports unparseable rate.

**Acceptance test:**
```bash
python scripts/smoke_test.py            # exits 0, prints sane latencies
python scripts/extraction_rate_check.py # extraction rate ≥95% on passes 1-4
```

**Stop conditions:**
- Extraction rate <95%: this is the ONLY phase where prompt iteration is permitted. Iterate the template until ≥95%. Document iteration in `preregistration.md` *before* the tag is finalized, or re-tag as `pre-pilot-v6.0.1` with explicit changelog. If after 3 prompt iterations the rate is still <95%, raise to the user.

---

### Phase 4 — Gate -1: Embedder validation

**Goal:** Verify that `bge-small-en-v1.5` embedding distance discriminates between different reasoning strategies on real Qwen output, before depending on it for H3.

**Inputs:** 3 GPQA Diamond problems (one per subject) drawn from the 50 sample. N=4 samples each = 12 reasoning traces total.

**Deliverables:**
- `scripts/run_gate_minus_1.py` — samples 4 completions per problem, presents trace pairs for manual labeling, writes labels to `outputs/gate_minus_1_labels.json`, computes distances and Mann-Whitney U.
- Manual labels for 15 trace pairs: `same-strategy` / `different-strategy` / `ambiguous` (defined in `docs/gate_minus_1_labeling_guide.md`).

**Acceptance test:**
- Mean embedding distance for `different-strategy` pairs > `same-strategy` pairs.
- Mann-Whitney U p-value < 0.10.
- Visible ranking gap (no overlap in middle 50% of distributions).

**Stop conditions:**
- Gate fails: swap to `jinaai/jina-embeddings-v3`, re-run. If still fails, kill H3 in `preregistration.md` (annotate, retag as `pre-pilot-v6.0.2-H3-killed`), shrink to Primary tier only.

---

### Phase 5 — Day 0 reconnaissance

**Goal:** Verify Qwen2.5-7B + GPQA Diamond is in the right capability range (target N=1 accuracy 0.25–0.45). Below floor: too hard; above ceiling: too easy.

**Inputs:** 10 problems (the 3 from Gate -1 + 7 additional from stratified pool). N=1 sample per problem at T=0.7. These 10 problems will also serve as the temperature side-test subset (Phase 7).

**Deliverables:**
- `scripts/run_recon.py` — samples N=1 on 10 problems, computes accuracy, prints band.

**Acceptance bands:**
- **Green (0.25–0.45):** proceed to main pilot with N ∈ {1, 2, 4, 8, 16, 32, 64}.
- **Yellow (0.15–0.25 or 0.45–0.55):** proceed; if near ceiling add N=128, otherwise note floor risk.
- **Red (<0.15 or >0.55):** pivot per fallback (MATH-500 level-5 for easier, MMLU-Pro reasoning for harder, or Qwen2.5-32B-Instruct as alternative model). Document in `preregistration.md` and re-tag.

---

### Phase 6 — Main pilot sampling

**Goal:** Collect the full grid: 47 problems × N ∈ {1, 2, 4, 8, 16, 32, 64} at T=0.7. ~21,000 model calls peak.

**Deliverables:**
- `scripts/run_main_pilot.py` — async sampler, max 32 concurrent calls, idempotent. Writes append-only to `outputs/samples/<problem_id>.jsonl`. Git-checkpoints every 5 completed problems.
- `outputs/samples/` populated.

**Acceptance test:**
```bash
python scripts/run_main_pilot.py        # exits 0
python scripts/validate_data.py outputs/samples/  # zero validation errors
# Verify all 47 problems have N=64 samples (max budget filled)
```

**Stop conditions:**
- Provider outage: retry with backoff (max 5 attempts); on persistent failure, halt and resume next day.
- Cost overrun beyond $12: halt and notify user.

---

### Phase 7 — Temperature side test

**Goal:** Test H6 — curve-shape stability across temperatures.

**Inputs:** 10 problems (the recon subset) × N=16 × T ∈ {0.3, 0.7, 1.0} = 480 additional calls. Reuse the T=0.7 samples already collected in Phase 6 for the T=0.7 slice.

**Deliverables:**
- `scripts/run_side_test.py` — fills the T=0.3 and T=1.0 cells. Writes to `outputs/samples/` with `temperature` field distinguishing the slices.

**Acceptance test:**
```bash
python scripts/validate_data.py outputs/samples/
# All 10 side-test problems should have N=16 samples at each of 3 temperatures.
```

---

### Phase 8 — Fitting

**Goal:** Apply the 5-family library to every problem; compute BIC weights, best-fit family, fitted elasticity.

**Deliverables:**
- `scripts/run_fitting.py` — reads `outputs/samples/`, computes per-problem accuracy at each N, fits all 5 families, writes per-problem result to `outputs/fits/<problem_id>.json`.
- `outputs/fits/` populated with one JSON per problem containing: `{family: {params, cov, bic, weight, residual_se, converged}}` plus `best_fit_family` and `mean_elasticity_8_64`.
- Also: `scripts/run_diversity.py` — computes embedding diversity at N=4 for each problem, writes to `outputs/diversity/<problem_id>.json`.

**Acceptance test:**
```bash
python scripts/run_fitting.py
python scripts/run_diversity.py
# Manual sanity check: open 6 random fit JSONs and visually inspect a fitted-curve plot
# (saved alongside fits) for each. Fits should look reasonable, not pathological.
```

---

### Phase 9 — Pass 1: confirmatory analysis

**Goal:** Run pre-registered H1–H6 tests against locked thresholds. Single summary table + 6 hypothesis-specific plots.

**Deliverables:**
- `scripts/run_analysis.py` — runs `tests/falsification.py` against `outputs/fits/` and `outputs/diversity/`, outputs `outputs/hypothesis_summary.json` and `outputs/plots/pass1_*.png`.
- `tests/falsification.py` — pytest suite that reads outputs and asserts each hypothesis against thresholds. Exit code 0 if all hypotheses resolved (pass or fail per spec); non-zero on unexpected errors.

**Acceptance test:**
```bash
pytest tests/falsification.py -v
# Each H1-H6 marked as PASSED or FAILED (not errored) with the actual measured stat
# alongside the threshold.
```

**Stop conditions:** see §6 (stop-condition matrix).

---

### Phase 10 — Pass 2: exploratory + go/no-go writeup

**Goal:** One exploratory pass (anything beyond H1–H6), plus a pilot writeup with the full-study decision.

**Deliverables:**
- `outputs/plots/pass2_*.png` — every plot tagged `[EXPLORATORY]` in caption.
- `PILOT_WRITEUP.md` — sections: (1) summary, (2) H1–H6 results, (3) exploratory findings, (4) limitations & confounders, (5) go/no-go decision, (6) full-study TODO updates.

**Acceptance test:** human review (this is the user's decision point).

**Hard cap:** no Pass 3. If you find yourself wanting one, write it as a full-study TODO.

---

## 6. Stop-condition matrix

| Failure | Phase detected | Response |
|---|---|---|
| Gate -1 (embedder) fails | 4 | Swap embedder → if still fails, kill H3, shrink to Primary, retag |
| Recon Red band | 5 | Pivot domain or model per fallback list, retag |
| Extraction rate <95% after iteration | 3 | Raise to user; do not proceed |
| H1 only falsified | 9 | Bump N→128 on subset, retest. Still fails → framework dead, pivot project |
| H2 only falsified | 9 | Reframe as "universal scaling" paper |
| H3 only falsified | 9 | Predictor track dead. Paper = Primary only, workshop-tier |
| H5 only falsified | 9 | Note in limitations; no degradation regime at this scope |
| H6 falsified | 9 | Reframe elasticity as (model, problem, prompt) property |
| H1+H2 both falsified | 9 | Framework dead at this scope. Pivot project — don't paper over |
| All pass but effects tiny | 9 | Frame as "universal scaling"; do NOT expand scope to chase effects |
| Cost >$25 mid-pilot | 6 | Halt, notify user |

Every stop-condition trigger requires updating `preregistration.md` with what happened and re-tagging (`pre-pilot-v6.0.X-<reason>`). No quiet continuation.

---

## 7. Repository layout

```
compute_elasticity_pilot/
├── README.md
├── PRD.md                       # THIS FILE
├── CLAUDE.md                    # repository conventions for Claude Code
├── preregistration.md           # locked before Phase 3, git-tagged
├── PILOT_WRITEUP.md             # Phase 10 deliverable
├── TODO_full_study.md           # all scope-creep temptations land here
├── requirements.txt
├── pyproject.toml               # ruff, black, pytest config
├── .env.example
├── .gitignore                   # excludes .env, outputs/, __pycache__
├── LICENSE                      # Apache 2.0
│
├── pilot/
│   ├── __init__.py
│   ├── config.py                # constants, HYPOTHESIS_THRESHOLDS
│   ├── prompts.py               # locked templates + hashes
│   ├── data_loader.py
│   ├── sampling.py              # async, idempotent
│   ├── scoring.py               # 5-pass + LLM scorer
│   ├── fitting.py               # 5 families + BIC
│   ├── diversity.py             # embeddings + truncation
│   ├── analysis.py              # bootstrap, AUC, elasticity
│   ├── validation.py            # data QA
│   └── plotting.py              # 6 locked figures
│
├── tests/
│   ├── test_prompts.py
│   ├── test_fitting.py          # synthetic recovery
│   ├── test_scoring.py
│   ├── test_diversity.py
│   ├── test_data_loader.py
│   ├── test_analysis.py
│   ├── test_validation.py
│   ├── test_sampling.py         # mocked HTTP
│   └── falsification.py         # H1-H6 reads outputs/
│
├── scripts/
│   ├── smoke_test.py
│   ├── extraction_rate_check.py
│   ├── run_gate_minus_1.py
│   ├── run_recon.py
│   ├── run_main_pilot.py
│   ├── run_side_test.py
│   ├── run_fitting.py
│   ├── run_diversity.py
│   ├── run_analysis.py
│   └── validate_data.py
│
├── data/                        # gpqa cached via datasets lib (gitignored except metadata)
│
├── outputs/                     # gitignored
│   ├── samples/                 # one .jsonl per problem
│   ├── fits/                    # one .json per problem
│   ├── diversity/
│   ├── flags/
│   └── plots/
│
└── docs/
    └── gate_minus_1_labeling_guide.md
```

---

## 8. Pre-flight checklist (Phase 2 gate)

All boxes must be ticked before `git tag pre-pilot-v6.0`:

- [ ] Together AI account funded, key in `.env`
- [ ] HuggingFace account, `huggingface_hub login` complete
- [ ] GitHub repo created public, Apache 2.0
- [ ] `preregistration.md` written and committed
- [ ] 50 stratified problem IDs locked in `preregistration.md` (JSON block)
- [ ] Subject distribution logged
- [ ] Prompt hash + scorer hash logged in `preregistration.md`
- [ ] All Phase 1 unit tests passing (`pytest tests/ -v` clean)
- [ ] `ruff check .` clean
- [ ] `black --check .` clean
- [ ] Anti-patterns in `CLAUDE.md` re-read

---

## 9. Budget guardrails

| Phase | Best | Realistic | Worst |
|---|---|---|---|
| 3 (smoke tests) | $0.10 | $0.50 | $1 |
| 4 (Gate -1) | $0.10 | $0.20 | $0.50 |
| 5 (recon) | $0.10 | $0.20 | $0.50 |
| 6 (main pilot) | $4 | $6 | $8 |
| 6 (Pass 5 scoring overhead) | $0.50 | $1 | $2 |
| 7 (side test) | $0.50 | $1 | $2 |
| Buffer | $0.50 | $1 | $2 |
| **Pilot total** | **~$6** | **~$10** | **~$16** |

`pilot/config.py` exposes a `COST_HARD_CAP = 12.0` constant. Sampling scripts must track per-call cost (input_tokens × input_price + output_tokens × output_price) and halt if cumulative cost exceeds 80% of cap ($9.60).

---

## 10. Beyond the pilot

If Phase 10 returns "go", the full study expands along:

- Additional models: Claude Haiku 4.5, GPT-5-mini, Gemini 2.5 Flash, Qwen2.5-32B
- Additional domains: LiveCodeBench, MATH-500, MMLU-Pro reasoning
- Additional strategies: tree search, verifier-guided decoding, reflection
- Hierarchical Bayesian fitting (partial pooling across problems within domain)
- Synthetic procedural benchmarks (depth-controlled arithmetic etc.)
- Adaptive allocator (Tertiary tier)
- Prompt-stability hypothesis (H6b)

Full-study budget: ~$95 realistic, ~$245 worst case.

Target venue: ICLR 2027 main track (submission ~October 2026), fallback NeurIPS 2026 workshops.

---

## 11. References

| Resource | URL |
|---|---|
| GPQA dataset | https://huggingface.co/datasets/Idavidrein/gpqa |
| Qwen2.5-7B-Instruct | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| Together AI docs | https://docs.together.ai |
| BGE small embedder | https://huggingface.co/BAAI/bge-small-en-v1.5 |
| sentence-transformers | https://www.sbert.net |
| scipy optimize | https://docs.scipy.org/doc/scipy/reference/optimize.html |

Foundational inference-time scaling literature: Wang et al. 2022 (self-consistency), Lightman et al. 2023 (PRMs), Cobbe et al. 2021 (best-of-N), Yao et al. 2023 (ToT), Shinn et al. 2023 (Reflexion), Brown et al. 2024 (Large Language Monkeys), Snell et al. 2024 (optimal test-time compute), OpenAI 2024 (o1 system card).

---

## 12. Glossary

| Term | Definition |
|---|---|
| **R(c)** | Expected accuracy as a function of compute budget c |
| **Elasticity E(c)** | dR/dc — marginal accuracy per unit additional compute |
| **Dominant mode** | A curve family that wins BIC selection on a non-trivial fraction of problems |
| **BIC weight** | Normalized exp(-ΔBIC/2) — soft model selection |
| **Gate -1** | Embedder validation pre-step (Phase 4) |
| **Recon gate** | Day-0 N=1 accuracy check (Phase 5) |
| **Pass 1 / Pass 2** | Pre-registered confirmatory vs. labeled exploratory |
| **Pre-registration** | Locked hypotheses + thresholds + stop conditions, git-tagged before sampling |

---

## End note

This PRD is the canonical project specification. Any deviation from a locked element (prompt, threshold, hypothesis, scope) requires explicit user approval AND a re-tag of `preregistration.md`. There is no other legitimate path.

The next legitimate trigger for revising this PRD is **pilot data**. Not further critique, not framing refinements, not exploratory thoughts. Data.
Start Phase 0.