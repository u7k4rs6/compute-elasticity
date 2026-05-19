## v6.0.2 — model variant amendment (2026-05-19)
- Switched model string from Qwen/Qwen2.5-7B-Instruct to
  Qwen/Qwen2.5-7B-Instruct-Turbo.
- Reason: Qwen2.5-7B-Instruct is not a serverless endpoint on Together AI.
  The Turbo variant is FP8-quantized but otherwise equivalent (same 7.61B
  params, same instruction tuning, same context window).
- Methodology impact: outputs may differ slightly from canonical bf16 weights.
  This is a confounder to note in the writeup, not a falsification of H1-H6.
  The pilot still measures compute elasticity for the model that is actually
  deployed at scale on this provider — which is arguably more representative
  than the canonical weights anyway.
- All hypotheses, thresholds, prompts, schemas unchanged.
- Reproducibility note: requires Python ≥3.10. On Python 3.14 specifically,
  requires dill ≥0.4, datasets ≥4.0, multiprocess ≥0.70.19 — older dill versions
  hit a pickle._batch_setitems signature mismatch on Python 3.14 that prevents
  datasets from loading GPQA Diamond. Tested working on Python 3.14 with these
  minimums.

---

## v6.0.1 — single-provider amendment (2026-05-19)
- DeepInfra fallback deferred to full study.
- Rationale: pilot fits within Together AI budget ($15 bonus credit available);
  fallback redundancy not justified at pilot scale.
- Affected: removes Phase 3 parity check; sample schema provider field always "together_ai".
- Methodology unchanged: same model, same prompts, same hypotheses, same thresholds.

---

# Pre-registration: Compute Elasticity in LLM Reasoning

**Schema version:** v6.0-pilot  
**Locked:** 2026-05-19  
**Git tag:** pre-pilot-v6.0  
**Status:** LOCKED — deviations require re-tagging with explicit changelog

---

## 1. Research Question

Does Qwen2.5-7B-Instruct-Turbo accuracy on GPQA Diamond follow a monotonically increasing
curve as inference-time compute grows via best-of-N sampling, or does it exhibit
saturation, plateau, or even unimodal degradation?

---

## 2. Model and Benchmark

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct-Turbo (FP8; see v6.0.2 amendment) |
| Benchmark | GPQA Diamond |
| Metric | Pass@1 accuracy (best of N samples scored independently) |
| N values | 1, 2, 4, 8, 16, 32, 64 |
| Temperature (main) | 0.7 |
| Temperature (side test) | 0.3, 0.7, 1.0 |
| Providers | Together AI (primary), DeepInfra (fallback) |

---

## 3. Sample Selection

**Method:** Stratified random sample of 50 GPQA Diamond problems, preserving subject
proportions. Seed: 42. Code: `pilot/data_loader.stratified_sample(problems, n=50, seed=42)`.

**Subject distribution:** biology: 16, chemistry: 17, physics: 17

**Problem IDs (JSON):**
```json
[
  "gpqa_diamond_0008",
  "gpqa_diamond_0052",
  "gpqa_diamond_0130",
  "gpqa_diamond_0027",
  "gpqa_diamond_0195",
  "gpqa_diamond_0029",
  "gpqa_diamond_0047",
  "gpqa_diamond_0101",
  "gpqa_diamond_0043",
  "gpqa_diamond_0075",
  "gpqa_diamond_0167",
  "gpqa_diamond_0080",
  "gpqa_diamond_0077",
  "gpqa_diamond_0164",
  "gpqa_diamond_0079",
  "gpqa_diamond_0082",
  "gpqa_diamond_0098",
  "gpqa_diamond_0046",
  "gpqa_diamond_0131",
  "gpqa_diamond_0030",
  "gpqa_diamond_0127",
  "gpqa_diamond_0051",
  "gpqa_diamond_0065",
  "gpqa_diamond_0074",
  "gpqa_diamond_0154",
  "gpqa_diamond_0012",
  "gpqa_diamond_0087",
  "gpqa_diamond_0151",
  "gpqa_diamond_0032",
  "gpqa_diamond_0173",
  "gpqa_diamond_0040",
  "gpqa_diamond_0159",
  "gpqa_diamond_0070",
  "gpqa_diamond_0096",
  "gpqa_diamond_0178",
  "gpqa_diamond_0007",
  "gpqa_diamond_0171",
  "gpqa_diamond_0050",
  "gpqa_diamond_0160",
  "gpqa_diamond_0107",
  "gpqa_diamond_0134",
  "gpqa_diamond_0106",
  "gpqa_diamond_0090",
  "gpqa_diamond_0010",
  "gpqa_diamond_0092",
  "gpqa_diamond_0156",
  "gpqa_diamond_0026",
  "gpqa_diamond_0176",
  "gpqa_diamond_0165",
  "gpqa_diamond_0121"
]
```

**Problem-ID list SHA-256:** `4f27d5db7ca80decb271af7b69c28199d9f4454101605b3b669cdbb8b81fde8e`

---

## 4. Prompt Template

**Locked template hash (SHA-256):** `e3544f731c3b30d49f373585e192da39347a272fe68fd9d309e8aafc763b73c1`

Template text (canonical, stored in `pilot/prompts.py::MAIN_PROMPT_TEMPLATE`):

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

---

## 5. Answer Extraction Pipeline (5 passes)

All passes are regex-based (1–4); Pass 5 is LLM-assisted fallback only.

| Pass | Scope | Pattern |
|---|---|---|
| 1 | Last 200 chars | `(?i)answer[:\s]+([A-D])\b` — last match wins |
| 2 | Full response | `\\boxed{([A-D])}` — last match wins |
| 3 | Last non-empty line | `\b([A-D])\b` — last match wins |
| 4 | Last 500 chars | `\b([A-D])\b` — last match wins |
| 5 | LLM scorer | Pass-5 scorer prompt (hash below) |
| 6 | Conservative | All passes fail + no api_client → incorrect |

**Pass-5 scorer hash (SHA-256):** `0ca0b0f97745fcb58756e9b0a42b4c2e2ff298eb92c6e3e35096567c34e0f303`

---

## 6. Curve Families

Five parametric families fitted via MLE (scipy.optimize.curve_fit) with BIC-weighted
soft model selection. BIC formula (Gaussian MLE):

```
BIC(k, n, RSS) = k·log(n) + n·log(2π·RSS/n) + n
```

| Family | Parameters | Notes |
|---|---|---|
| constant | R₀ | Baseline: no elasticity |
| logistic | L, k, c₀ | Monotone S-curve |
| gompertz | L, b, k | Asymmetric S-curve |
| shifted_logistic | L, k, c₀, f | Logistic with non-zero floor |
| unimodal | L, k, c₀, σ | Gaussian bump — tests H5 |

BIC weights: `w_f = exp(-0.5·ΔBIC_f) / Σ exp(-0.5·ΔBIC_i)`

---

## 7. Pre-registered Hypotheses

All thresholds are locked in `pilot/config.py::HYPOTHESIS_THRESHOLDS`. Changes require
re-tagging.

### H1 — Curve fit quality
Metric: `median_residual_se` across all 50 problems.  
**Pass:** median_residual_SE ≤ 0.10  
**Fail:** median_residual_SE ≥ 0.15 (indeterminate if between)

### H2a — Model uncertainty
Metric: number of curve families with mean BIC weight ≥ 0.10.  
**Pass:** ≥ 2 families each with mean BIC weight ≥ 0.10

### H2b — Competing families
Metric: fraction of problems where runner-up family is within 2 BIC units.  
**Pass:** fraction ≥ 0.30

### H3 — Diversity-accuracy correlation
Metric: AUC of embedding diversity predicting accuracy gain from N=8 to N=64.  
**Pass:** AUC ≥ 0.60 AND ΔAUC vs. entropy ≥ 0.03  
**Fail:** AUC ≤ 0.55

### H5 — Unimodal degradation
Metric: fraction of problems where unimodal family wins BIC.  
**Pass:** fraction ≥ 0.05  
**Fail:** fraction = 0.00

### H6 — Bootstrap stability
Metric: fraction of problems where 95% CI overlaps point estimate ± 0.05.  
**Pass:** fraction ≥ 0.70  
**Fail:** fraction ≤ 0.50

---

## 8. Stop Conditions

- **Cost ≥ $20.00** (80% of $25 hard cap): halt sampling, report status.
- **Extraction rate < 95%** on passes 1–4: iterate prompt template (Phase 3 only), re-tag.
- **Gate -1 fails** (embedder): swap to `jinaai/jina-embeddings-v3`, re-run. If still fails: kill H3, annotate here, re-tag as `pre-pilot-v6.0.2-H3-killed`.
- **Day 0 recon accuracy** < 0.15 or > 0.55: pivot benchmark/model per PRD §5.

---

## 9. Pre-flight Checklist

- [x] Together AI account — key in `.env`
- [x] DeepInfra account — key in `.env`
- [x] HuggingFace account — `HF_TOKEN` in `.env`
- [x] GitHub repo created public, Apache 2.0
- [x] `preregistration.md` written and committed
- [x] 50 stratified problem IDs locked (JSON block above)
- [x] Subject distribution logged
- [x] Prompt hash + scorer hash logged
- [x] All Phase 1 unit tests passing (`pytest tests/ -v` clean)
- [x] `ruff check .` clean
- [x] `black --check .` clean
- [x] Anti-patterns in `CLAUDE.md` re-read

---

## 10. Deviation Log

*(Empty at lock time — any entry here triggers re-tagging)*
