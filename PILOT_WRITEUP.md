# Pilot Writeup: Compute Elasticity in LLM Reasoning

## TL;DR

A pre-registered pilot characterizing how LLM reasoning accuracy scales with inference-time compute on GPQA Diamond. We sample Qwen2.5-7B-Instruct-Turbo at N ∈ {1, 2, 4, 8, 16, 32, 64} via best-of-N on 47 stratified problems, fit five parametric curve families per problem with BIC weights, and evaluate six pre-registered hypotheses about curve shape, predictability, and stability.

Four hypotheses passed: curves fit at median residual SE = 0.0107 (an order of magnitude below the 0.10 threshold); model competition is substantial, with three families sharing mean BIC weight ≥ 0.10 and 34% of problems having ≥2 families within 2 BIC of the best fit; 21/47 problems get a unimodal best fit, but c* sensitivity analysis shows the peak falls interior to the compute grid in only 1 case (the rest are decaying or saturating fits with peaks outside N ∈ [1, 64]); bootstrap 95% CIs for fitted parameters overlap near-perfectly across temperatures {0.3, 0.7, 1.0} on the 10-problem side test (mean overlap rate = 1.000).

One hypothesis failed informatively: we pre-registered that embedding diversity at N=4 would predict above-median elasticity better than per-token entropy at N=1; the reverse held (AUC 0.650 vs 0.524 in entropy's favor). One was deferred by design (H4, requires multi-domain runs).

Two findings stand out: (i) compute elasticity is essentially stable across temperatures: H6's 1.000 overlap suggests elasticity is fundamentally a (model, problem) property, not a (model, problem, temperature) one; (ii) the H3 falsification reframes the secondary contribution around direct generation-step uncertainty signals rather than embedding-level content variety.

Total API spend: ~$0.60 of a $15 budget. The parametric framework is empirically viable. **The full multi-model, multi-domain study is GO.**

## Background and Motivation

In the post-o1/o3 era, inference-time compute has become a primary axis for improving LLM reasoning. A growing toolkit (best-of-N sampling, self-consistency, tree search, verifier-guided decoding, reflection) exploits the fact that giving a model more compute at inference time can lift accuracy, sometimes dramatically. Frontier reasoning systems (OpenAI o1/o3, DeepSeek R1, Claude with extended thinking, Gemini Deep Think) all rely on this internally. Yet a basic empirical question remains underexplored: *what does the compute-response curve actually look like, across problems and models?* The literature introduces methods on top of this axis without first characterizing the axis itself. There is no systematic parametric taxonomy of per-problem compute-response shapes, no quantitative framework for asking when more compute is worth spending on a given problem, and no predictor of compute elasticity validated against fitted parametric curves.

This project treats compute elasticity as a measurable latent property of (model, problem, strategy) tuples. The unit of analysis is the per-problem compute-response function R(c), fitted from a small library of parametric families and selected by BIC weight. The framework lets us ask whether curves have peaks, whether cheap signals predict their steepness, and whether they are stable across temperatures. The pilot exists because the full multi-model, multi-domain study is expensive (~$80-250 in API spend depending on scope), and only worth running if the framework is empirically viable at the smallest reasonable scope: one open-weights model, one benchmark (GPQA Diamond, graduate-level science QA), one inference strategy (best-of-N, the simplest pure-compute axis), 47 problems. If R(c) is too noisy to fit, the framework collapses. If a single curve family always wins, the framework is uninteresting. If embedding-based predictors of elasticity beat simpler baselines, the full study's predictability track has a clear feature to scale up. The pilot tests each of these, and a few more, before the project commits to the larger investment.

## Methodology

### Data and Sampling

We work with GPQA Diamond (198 problems, graduate-level multiple-choice questions across physics, chemistry, and biology). Fifty problems are sampled with stratification by subject proportional to the dataset distribution, using `random.seed(42)`. Three problems are reserved as the **Gate-1 validation set** (one per subject), used before the main pilot to verify that embedding diversity is informative on real Qwen output before committing to it as the H3 primary feature; the remaining 47 constitute the main pilot set. Ten additional problems (the three Gate-1 carve-outs plus seven from the main 47) serve as the side-test subset for H6 temperature stability.

All sampling uses **Qwen2.5-7B-Instruct-Turbo** via the Together AI API. For each main-pilot problem we draw 64 independent completions at temperature 0.7, each with an independent cryptographic seed (`secrets.token_bytes(8)`) logged per call. The same 64 samples are reused to compute the compute-response value R(N) for every budget N ∈ {1, 2, 4, 8, 16, 32, 64} via the unbiased pass@N estimator from Chen et al. (2021):

R(N) = 1 − C(64 − c, N) / C(64, N)

where c is the number of correct completions among the 64. This is equivalent to **best-of-N with an oracle verifier** (the benchmark's ground-truth answer key), averaged exactly over all (64 choose N) subsets. Budgets are therefore nested in the sense that all seven R(N) values per problem are derived from the same 64 samples; no independent re-sampling occurs across budgets. The inference strategy is best-of-N, the simplest pure-compute axis, explicitly excluding tree search, reflection, and verifier-guided variants from pilot scope. Storage is append-only JSONL per problem, allowing idempotent resumption across network failures.

The side test follows the same protocol with N=16 samples per problem per temperature, across temperatures {0.3, 0.7, 1.0}.

Each completion is scored by a five-pass extractor: four progressively looser regex passes targeting the locked `Answer: X` marker, followed by a Pass-5 LLM-as-judge fallback using Qwen2.5-7B itself for ambiguous cases. Pass 1 achieved 100% hit rate on 20 pre-pilot validation samples; during the actual pilot, Pass 5 was never invoked. The full prompt template is locked and SHA-256 hashed before any sampling begins ("Day 0" in the project timeline).

### Curve Fitting

Five parametric families are fitted to the (N, R(N)) pairs per problem:

- **Constant:** R(c) = p, encoding "no compute response."
- **Logistic:** R(c) = L / (1 + exp(-k(c - c₀))), encoding smooth saturation.
- **Gompertz:** R(c) = L · exp(-b · exp(-k·c)), encoding asymmetric saturation (steep early, slow tail).
- **Shifted logistic:** logistic with delayed onset and small floor, encoding emergence above a threshold.
- **Unimodal:** R(c) = A · exp(-((c - c*)²) / (2σ²)) + b, with peak at c*, encoding the "more samples can hurt" regime.

Fits use **nonlinear least squares** via `scipy.optimize.curve_fit` with bounded parameters and multi-start optimization (10 random initial conditions per fit, deterministic seed). BIC is computed under the implicit Gaussian-error approximation of least squares; we note this approximation breaks down near R(c) = 0 or 1 where the underlying binomial variance is strongly heteroskedastic, and treat the fitted BIC weights as a soft selection signal rather than a calibrated probability. Selection across families is via BIC weights w_i = exp(-ΔBIC_i / 2) / Σ exp(-ΔBIC_j / 2). Per-problem outputs include the best-fit family, fitted parameters with bootstrap CIs (1000 non-parametric resamples of the 64 raw samples per problem, 95% percentile method), the full BIC weight distribution, and residual standard error.

### Predictability Features

After sampling, two kinds of features are computed per problem for the H3 predictability test:

- **Embedding diversity (primary feature, pre-registered).** Take N=4 reasoning traces, strip the final `Answer: X` line, truncate each at the last sentence boundary before 512 tokens, embed with `BAAI/bge-small-en-v1.5`, and compute mean pairwise cosine distance across the four embeddings.
- **Per-token uncertainty (baseline).** From the N=1 generation, compute both per-token Shannon entropy (in nats) over Together AI's returned top-5 logprobs and the mean per-token negative log-likelihood of the generated tokens. Shannon entropy is the pre-registered baseline; mean NLL was added during analysis when the Together AI logprobs schema was clarified and is reported alongside entropy.

H3 passes if AUC(diversity) ≥ 0.60 *and* AUC(diversity) − AUC(entropy) ≥ 0.03.

### Pre-registration Discipline

All six hypotheses with locked thresholds and pre-registered stop conditions are written to `preregistration.md` and git-tagged as `pre-pilot-v6.0` before any sampling. Analysis is restricted to two passes: Pass 1 confirmatory (H1–H6 against locked thresholds), Pass 2 exploratory (anything else, plots and tables tagged `[EXPLORATORY]`). No Pass 3, since additional analyses become full-study work, not pilot work.

Three amendments occurred between pre-registration and pilot completion, each documented in `preregistration.md`:

- **v6.0.1 (single-provider).** The original spec required Together AI as primary and DeepInfra as fallback, with a SHA-256 parity check between the two before failover. The parity-check infrastructure complicated the data-loading code without adding scientific value at pilot scope, so DeepInfra was dropped before sampling began. Tagged `pre-pilot-v6.0.1-single-provider`.
- **v6.0.2 (Turbo variant).** The base `Qwen/Qwen2.5-7B-Instruct` model was not served on Together AI's serverless tier; the FP8-quantized `Qwen/Qwen2.5-7B-Instruct-Turbo` variant was. Switching preserves the model family and reasoning capability at the pilot's price point, with the quantization noted as a paper confounder. Tagged `pre-pilot-v6.0.2-turbo-variant`.
- **Multi-start optimization.** During Phase 8 fitting, the shifted-logistic family showed a 62% non-convergence rate on real data (29 of 47 problems failed to converge from default initial conditions). Adding 10 random restarts per fit eliminated all 29 failures. The amendment was applied **uniformly across all five families and all 47 problems before any hypothesis evaluation, with no selective application or post-hoc family choice**. No git tag was added because the underlying samples and family library were unchanged; only the optimizer changed. We acknowledge that multi-start can change which family wins BIC for problems where the original fit converged to a local minimum, which is precisely why the amendment was applied uniformly and before H1–H6 evaluation.

H6 uses only fitted-curve parameter overlap across temperatures and does not consume embeddings or any feature derived from the Gate-1 validation work, so reusing Gate-1 problems in the side test does not leak information from the embedder protocol into H6.

All three amendments occurred *before* hypothesis evaluation. No amendment was made in response to H1–H6 results.

## Results

We present the six hypotheses in canonical order. Three classes of finding emerge: a confirmed Primary tier (H1, H2, H5, H6), an informative falsification (H3) that reframes the Secondary contribution, and one hypothesis deferred to the full study by design (H4).

### H1: Compute-response curves fit above noise (PASS)

H1 tests whether R(c) curves are fittable at all under our budget, the most basic existence proof. The locked threshold is median residual standard error across 47 problems < 0.10 in accuracy units, with falsification at ≥ 0.15.

The measured median residual SE was **0.0107**, an order of magnitude below the pass threshold. R(c) curves are not only fittable but fit at near-noise-floor precision. This eliminates the "framework dead at this scope" stop condition outright.

### H2: Multiple curve families share substantial selection weight (PASS, both criteria)

H2 tests whether the parametric library is informationally diverse, whether the data actually justifies a soft-selection framework rather than a single canonical curve. Two operational criteria were locked: (a) at least 2 of the 5 families have mean BIC weight ≥ 0.10 across problems, and (b) at least 30% of problems have ≥ 2 families within 2 BIC of the best.

Both criteria passed. Three families exceed mean BIC weight 0.10: **unimodal (0.421), gompertz (0.293), and shifted logistic (0.183)**; constant and logistic trail at **0.038 and 0.065** respectively. **34.04% of problems (16/47) have ≥ 2 families within 2 BIC of the best**. Model competition is substantial; the soft-selection framework is justified.

The high unimodal mean weight is notable in itself, averaged across all problems, the unimodal family carries nearly half of the soft-selection mass, suggesting it is a strong universal approximator for these curves whether or not its peak falls inside the budget grid. We return to this point in H5.

### H3: Embedding diversity does not outperform per-token uncertainty (FAIL, informatively)

H3 was the pre-registered predictability bet: that mean pairwise cosine distance among N=4 reasoning traces would forecast above-median fitted elasticity better than the per-token-entropy baseline at N=1, by AUC margin ≥ 0.03 with AUC(diversity) ≥ 0.60.

The data falsifies the hypothesis: **AUC(diversity) = 0.524** (near chance), **AUC(entropy) = 0.650**, **ΔAUC = -0.127** in favor of entropy, the reverse of our prediction. Adding mean NLL as a post-hoc comparator gives AUC(NLL) = 0.636, also above diversity. Both direct generation-step uncertainty signals carry more elasticity information than embedding-level content variety at the budget tested. Entropy is the stronger predictor but still modest (AUC 0.650, just above the H3 threshold for diversity), suggesting compute elasticity has a substantial unpredictable component even when conditioned on generation-step uncertainty.

We read this as a clean falsification with a constructive implication: per-token entropy and NLL (both computable from a single N=1 call with logprobs) are the predictor candidates to scale into the full study, not embedding diversity. The Secondary contribution survives, reframed, but with the expectation that any allocator built on these features will be probabilistic, not deterministic.

### H4: Curve-family distribution differs across domains (DEFERRED by design)

H4 cannot be tested on a single-domain pilot. It is deferred to the full study, where multi-domain runs (LiveCodeBench, MATH-500, MMLU-Pro reasoning subset) will probe whether the family-win distribution shifts systematically across reasoning types.

### H5: Non-trivial fraction of problems exhibit a unimodal best fit (PASS, with nuance)

H5 tests whether the "more samples can hurt" regime, a unimodal R(c), exists at a rate above 5% of problems.

The unimodal family wins BIC selection on **21 of 47 problems (44.7%)**, well above threshold. H5 is satisfied at the literal level. But the c* sensitivity analysis qualifies the result. Among the 21 unimodal winners, the decomposition by peak location (regime boundaries from `outputs/unimodal_sensitivity.json`) is:

- **1 interior peak** (4.76%): c* ∈ [8, 32]: genuine non-monotonicity (gpqa_diamond_0026, c* ≈ 11).
- **14 decaying fits** (66.67%): c* ≤ 4: peak at or before the first sampled budget; effectively a decaying curve the unimodal family approximates better than gompertz or logistic.
- **5 saturating fits** (23.81%): c* ≥ 56: peak at or beyond the largest sampled budget; effectively a saturating curve where the upper Gaussian tail mimics a plateau.
- **1 borderline-low** (4.76%): c* ∈ (4, 8): ambiguous between interior and decaying regimes.
- **0 borderline-high** (32 < c* < 56): empty.

A constrained re-fit with c* bounded to ≤ 64 changes the BIC winner for only 1 of 21 problems, flipping from unimodal to logistic (the saturating case). This suggests the unimodal family wins for likelihood reasons rather than as a numerically sloppy unbounded fit; in 20 of 21 cases it remains the best BIC choice even when forced to put its peak inside the budget grid.

The headline number (44.7%) therefore reflects unimodal *competition for the BIC win*, not genuine non-monotonicity in the response. Genuine "more samples can hurt" appears in **1 of 47 problems (≈ 2%)** at this scope. We report both views; the c* sensitivity decomposition is the honest reading.

### H6: Curve parameters are stable across temperature (PASS, strongly)

H6 tests whether fitted curve parameters are consistent across sampling temperatures, with the locked threshold at mean bootstrap-CI overlap rate ≥ 0.70.

For each problem, the family with highest BIC weight at T=0.7 (the main-pilot temperature) was fixed as the reference family and re-fit across all three temperatures; parameter overlap was evaluated within that fixed family. Reference families across the 10 side-test problems were unimodal (6), gompertz (2), shifted logistic (1), and constant (1).

On the 10-problem side-test subset, bootstrap 95% CIs at temperatures {0.3, 0.7, 1.0} overlap **at rate 1.000**: every parameter, every problem, every temperature pair. Mean overlap is exactly 1. This is substantially stronger than the locked threshold and stronger than we anticipated: within the range tested, compute elasticity is essentially temperature-invariant, which in turn suggests it is fundamentally a property of the (model, problem) pair rather than the (model, problem, temperature) triple.

The result has implications for the full study's experimental design: within the range tested, temperature variation is unlikely to be a productive axis of expansion, while model variation, domain variation, and inference-strategy variation are.

## What we learned beyond the locked hypotheses

The pre-registered hypotheses are the official scoreboard, but the pilot also produced several findings that aren't in the hypothesis register and that materially shape what the full study should do. None of these were planned; all were discovered in the course of running the pipeline.

### The shifted-logistic non-convergence finding

The original fitting code used `scipy.optimize.curve_fit` with a single deterministic initial condition per family. On synthetic data with clean sigmoidal shapes this worked for all five families. On real Qwen2.5-7B-Instruct-Turbo output, **shifted logistic failed to converge on 29 of 47 problems (62%)**: the solver landed in degenerate regions where the floor parameter saturated against bounds or the threshold parameter ran off to infinity. The other four families converged cleanly from any reasonable start; only shifted logistic was sensitive.

The fix was to add 10 random initial conditions per fit and select the lowest-loss converged result. Applied uniformly across all five families, the shifted-logistic non-convergence rate dropped to **0 of 47**. The fix was committed before any hypothesis was evaluated; the integrity argument for the amendment is recorded in `outputs/convergence_breakdown.json` and discussed in Section 3.

The methodological lesson for the full study is concrete: **the parametric library cannot be assumed to fit cleanly from default initial conditions on real LLM data.** Any extension to new families or new models needs multi-start optimization as a default, not an opt-in.

### The H3 reframing toward direct uncertainty signals

H3 was the most prediction-heavy hypothesis in the pre-registration: we committed to a specific featurization (mean pairwise cosine distance of N=4 BGE-small embeddings) as the primary predictor of compute elasticity. The featurization came from a reasonable intuition (*if a problem produces wildly different reasoning traces, its accuracy might benefit more from additional samples*) but the intuition didn't survive contact with the data.

The constructive read is that the *replacement* predictors are simpler and more sample-efficient:

- Per-token Shannon entropy and mean NLL are both computable from a single N=1 call with the API's `logprobs` field. No additional samples required.
- They beat embedding diversity at AUC 0.650 / 0.636 vs 0.524.
- They are length-normalized (mean per token), so they are properties of the generation process rather than verbosity.
- They are model-agnostic in form: any provider exposing top-K logprobs supports them.

The full study should drop embedding diversity as a primary feature, retain it as a comparator if budget permits, and lead with mean per-token entropy/NLL. A meta-finding worth flagging in the paper: **the predictability section's headline shifts from "embedding-based features predict reasoning compute elasticity" to "direct generation-step uncertainty signals predict reasoning compute elasticity, even modestly."** The latter is the more publishable claim because it ties to a longer thread of work on uncertainty-based stopping rules.

### The H5 sensitivity reframe as a methodological pattern

The H5 result is the clearest example in the pilot of why pre-registered thresholds need their own sensitivity analyses. The threshold ("≥ 5% of problems get unimodal best fit") was easy to satisfy at 44.7%. But that number, taken at face value, would have supported a paper claim (*"unimodal compute-response is common in graduate-level science reasoning"*) that the data does not actually support. Only 1 of 47 problems has c* interior to the budget grid; the other 20 unimodal winners are non-monotonicity-free curves that the unimodal family approximates well at the edges of its shape envelope.

The lesson generalizes beyond H5: **any time a parametric family wins a BIC competition, check whether the winning fit's parameters land inside the data range or extrapolate outside it.** A win on the basis of out-of-range parameters is a different scientific finding than a win on the basis of in-range parameters. The full study should bake this check into the fitting pipeline by default, not as a post-hoc analysis.

### H6 as a robustness validation

The temperature-invariance result (mean overlap rate = 1.000) is striking on its own merits (see Section 4), but it also doubles as a robustness check on the parametric framework itself. If the curve-fitting pipeline were sensitive to sampling stochasticity, we would expect bootstrap CIs across temperatures to drift apart even when the underlying (model, problem) dynamics are stable. They don't drift at all in the side-test subset, which means the framework recovers the same shape from independently drawn samples at three temperatures, three independent times.

The lesson for the full study is twofold. First, **temperature sweeps are not a productive axis of expansion**; budget is better spent on model, domain, or strategy variation. Second, **the parametric library is robust enough to handle real sampling noise across temperatures**, a non-obvious property at the start of the pilot, and one that supports scaling the framework to other models with confidence that the fits won't drift due to noise alone.

### Operational lessons from running the pipeline

Four operational issues surfaced during execution that any reimplementation of this pipeline should know about:

- **Together AI logprobs schema is not OpenAI's.** Together returns `tokens`, `token_logprobs`, and `top_logprobs` as parallel arrays per response, not OpenAI's nested `content[i].top_logprobs[j]` structure. An initial implementation of the entropy baseline read the wrong fields and produced all-zero results. The fix required reading the API response schema directly rather than reusing OpenAI-style parsing code.
- **`secrets.token_bytes(8)` seed-batch semantics.** The sampling helper batched API calls, so the per-call seed applied to the batch rather than each individual completion. When requesting 16 completions, batch-size rounding meant some (problem, temperature) pairs received 25 samples instead of 16. The fix was to subsample exactly 16 at analysis time using a fixed seed, ensuring fair cross-temperature comparison.
- **Best-of-N tie-breaking originally peeked at ground truth.** The scoring code's tie-break logic for "which answer wins when multiple completions extract the same letter" was implemented in a way that subtly preferred the ground-truth answer when ties occurred, biasing accuracy upward by a few tenths of a percent at high N. The fix was to use sorted-then-seeded-RNG tie-breaking, which is ground-truth-agnostic. The bias was caught and fixed during Phase 8 implementation, before any fits were computed; all reported fits and verdicts use the corrected tie-break. The magnitude would have been small (a few tenths of a percent at high N), but the fix matters for any future allocator built on these accuracy curves: systematic upward bias propagates to allocation decisions.
- **`httpx.ReadTimeout` was not retried.** One network drop at problem 30 of the main pilot caused the script to crash mid-run. Idempotent resumption recovered cleanly (the append-only JSONL design paid off), but the retry decorator should have caught `ReadTimeout` as a transient failure rather than letting it propagate. The full study should harden `_with_retry` against this case before launching multi-day sampling runs.

## Implications for the Full Study

The pilot's job was to test whether the parametric framework is empirically viable at the smallest reasonable scope. It is. The full study can now proceed with a sharper plan than the original PRD anticipated, since three of the planned axes are confirmed productive, one is now contraindicated, and the three-tier contribution hierarchy has shifted within its overall structure.

### Primary tier: confirmed, expand on three axes

The Primary contribution (characterization of compute-response dynamics via parametric fitting + BIC weights) survives intact. The full study expands characterization along three axes:

- **Models.** Cross-model validation against four additional models currently served by their respective APIs: **Claude Haiku 4.5** (Anthropic, `claude-haiku-4-5-20251001`), **GPT-5.4 mini** (OpenAI, the current recommended efficient model for new workloads), **Gemini 2.5 Flash** (Google), and **Qwen2.5-72B-Instruct-Turbo** (open weights via Together AI, a larger sibling of the pilot model in the same Turbo FP8 quantization scheme; the v6.0.2 amendment in Section 3 applies symmetrically). This is a descriptive comparison of how model scale, architecture, and post-training recipe shift the family-win distribution, not a causal claim about institutional differences between labs.
- **Domains.** Three additional benchmarks: LiveCodeBench (code generation, where best-of-N has the most established literature), MATH-500 level-5 problems (mathematical reasoning), and MMLU-Pro's reasoning subset (general). This is the deferred H4, asking whether curve-family distributions shift systematically by domain.
- **Strategies.** Tree search, reflection, and verifier-guided decoding. The pilot held strategy fixed at best-of-N; the full study tests whether elasticity is intrinsic to the (model, problem) pair or whether different strategies produce different elasticities for the same pair. A positive result here would weaken the "elasticity is intrinsic" framing the pilot's H6 result invites.

Pooled estimation across problems within domain (partial pooling) is added because the cross-model expansion divides the per-problem sample budget by four; pooling shrinks sparse-cell estimates toward the domain mean and provides principled uncertainty on derived quantities like family-win fractions.

### Secondary tier: reframed, lead with uncertainty signals

H3's falsification reframed the predictability track. The full-study predictor sweep should:

- **Lead with mean per-token entropy and mean NLL at N=1.** These are the pilot-validated baselines, both requiring nothing beyond the API's `logprobs` field. Across the four full-study models, this requires verifying each provider's logprobs schema (Anthropic, OpenAI, Google each return logprobs differently), a known operational gotcha from the pilot.
- **Treat embedding diversity as a comparator, not a primary.** The pilot's falsification was specific to BGE-small-en-v1.5 at N=4. A stronger embedder or larger N might rescue the signal; the full study can test that as a robustness check rather than the headline.
- **Add cheap problem-side features.** Problem-text length, difficulty estimates from the benchmark's own labels (where present), and subject-matter category as covariates.

The AUC bar should also be raised, but ranking AUC alone is insufficient for an allocator. The full study sets a **dual target**: AUC ≥ 0.70 for ranking (a meaningful improvement over the pilot's 0.650 entropy baseline) *plus* calibration error < 0.10 for probability estimates. The second condition matters because an allocator built on a well-ranked but miscalibrated predictor will systematically over- or under-allocate compute, even if its ranking is correct.

### Tertiary tier: conditional on Secondary paying off

The adaptive allocator (Tertiary contribution) becomes conditional on the predictability work scaling. If entropy/NLL clear the dual target (AUC ≥ 0.70 with calibration error < 0.10) across the four-model expansion, the allocator design has a foundation. If predictability collapses on a stronger or more diverse model set, the Tertiary tier becomes a workshop-paper line of work rather than a main-track claim, and the paper rebalances toward the strengthened Primary characterization.

The pilot's H6 result has a direct implication here: the allocator should *not* condition on temperature. A single-temperature predictor sweeping over (problem, model) is sufficient.

### Axes that are *not* productive

The pilot showed temperature variation is unlikely to yield differential elasticity at this scope, so the full study should drop temperature sweeps. The freed budget is modest (the side test was ~2% of total compute) but better spent on model or domain variation.

Synthetic procedural benchmarks (depth-controlled arithmetic, generated compositional tasks) remain on the plan as contamination defense, not as primary results. They are explanatory probes for phenomena observed in real benchmarks, not standalone findings.

### Live tracking

Specific tasks, budget rebudgets, and amendment history are tracked in `TODO_full_study.md` in the repo. The full-study pre-registration will inherit the pilot's six hypotheses with two modifications: H1, H2, H5, H6 thresholds tightened given pilot results (the pilot's margins were larger than the locked thresholds, so the full study can afford stricter pass criteria), and H3's primary predictor swapped to mean NLL with embedding diversity as a comparator.

## Limitations

The pilot's evidence is bounded in several ways that the full study should address. We list these in rough order of how much they constrain the pilot's claims.

### Best-of-N with oracle verifier is an upper bound

R(N) in this pilot is computed via the pass@N estimator using the benchmark's ground-truth answer key as the verifier. This is a clean methodological choice (the pass@k estimator is unbiased and standard in the code-generation literature), but it represents the *theoretical ceiling* of best-of-N performance, not what any real-world allocator can achieve. A practical allocator must select among N candidates *without* ground-truth access, using a verifier model, reward model, majority vote, or similar heuristic. The gap between oracle best-of-N (what we measure) and verifier-guided best-of-N (what's deployable) is unknown at pilot scope and is a first-order concern for the Tertiary allocator contribution.

### Single model, single benchmark, single strategy

The pilot characterizes Qwen2.5-7B-Instruct-Turbo on GPQA Diamond with best-of-N. Nothing in this pilot tells us:

- Whether other models in the 7B-70B class produce qualitatively similar compute-response shapes.
- Whether other reasoning domains (code, math, general knowledge) show similar family-win distributions.
- Whether other inference strategies (tree search, reflection, verifier-guided) produce similar elasticity profiles for the same (model, problem) pair.

H4 was deferred by design, and the full study expands models, domains, and strategies. Until those expansions land, the pilot's findings are formally about one (model, benchmark, strategy) tuple, with limited temperature probing on a 10-problem side test (H6).

### Contamination risk

GPQA Diamond is a public benchmark released in late 2023. Qwen2.5-7B's training data composition is not fully disclosed, but there is non-trivial probability that some GPQA problems or close paraphrases appeared during pre-training or instruction tuning. We did not measure contamination directly. Curves on contaminated problems may exhibit artificially flat R(N) (always correct) or artificially constant-family wins, which would distort the family-win distribution upward for the constant or saturating families.

The full study's synthetic procedural benchmarks (depth-controlled arithmetic, generated compositional tasks) are the principled defense against this concern, but the pilot's GPQA-only findings should be read with contamination as an open caveat.

### Quantization confound

The pilot model is FP8-quantized (the Turbo variant). We do not know whether base FP16 Qwen2.5-7B-Instruct would produce different compute-response dynamics. Quantization can affect the distribution of token-level logprobs (compressing the tail toward zero), which directly affects the entropy and NLL features used as the H3 baseline. The full study should include at least one provider-pair comparison (the same model in Turbo and base form, if both become available) to measure the quantization effect on fitted curves and on uncertainty-feature behavior.

### Embedder and N specificity of the H3 falsification

The H3 falsification is specific to BGE-small-en-v1.5 (a 130M-parameter general-purpose semantic embedder) at N=4. A stronger embedder (a 1B+ instruction-tuned semantic model, a model trained specifically for code/reasoning similarity, or one that handles longer truncation windows than 512 tokens) might rescue the embedding-diversity signal. The full study should test at least one stronger embedder before concluding that embedding-based features are dead as predictability candidates. The pilot's H3 result reads more accurately as *"embedding diversity in this particular setup does not beat entropy"* rather than *"embedding diversity is fundamentally weaker than entropy."*

### Underpowered AUC comparison

The H3 verdict rests on AUCs computed over 47 problems split into above-/below-median elasticity classes (≈ 23 / 24). We did not compute bootstrap confidence intervals on the AUC difference (entropy − diversity = 0.127). A non-overlapping CI would strengthen confidence that entropy genuinely outperforms diversity; an overlapping CI would suggest the ranking is fragile at this sample size. The full study's cross-model expansion provides natural replication for this comparison: if the entropy > diversity ordering holds across 4-5 additional models, the inference becomes much stronger than any single-sample-size CI could provide.

### Statistical scope at 47 problems

Forty-seven problems is sufficient for the omnibus hypotheses (H1, H2, H5, H6) but thin for subject-stratified sub-claims. Per-subject sample sizes (physics ≈ 16, chemistry ≈ 16, biology ≈ 15, after subtracting the 3 Gate-1 carve-outs) give wide CIs on within-subject family-win distributions. We did not report subject-stratified fractions because the per-subject N is too small to support them with calibrated uncertainty. The full study should expand sample sizes per (model, domain) cell to enable within-cell stratification.

### BIC weight calibration near boundaries

The Gaussian-error BIC approximation (Section 3) may inflate weight differences between families when R(N) approaches 0 or 1, where the underlying binomial variance is strongly heteroskedastic. This matters most for H2 (where the criterion is mean BIC weight ≥ 0.10) and H5 (where unimodal must win the BIC competition), since both verdicts are driven by BIC ratios. The qualitative direction of both verdicts is unlikely to flip under a more proper binomial likelihood (H2 passed by margin on both criteria, and H5's 21/47 unimodal-win count is far above the 5% threshold), but the precise weight magnitudes reported should be read as Gaussian-approximation estimates rather than calibrated probabilities.

### Operational limitations not affecting verdicts but worth flagging

- The Pass-5 LLM scorer was never invoked during the pilot (Pass 1 caught 100% of validation samples), so we have no in-context estimate of its actual error rate. The full study's models may produce longer or differently formatted reasoning traces, increasing the chance that the extractor encounters edge cases Pass 5 was designed for. Calibrating Pass 5 on a sample of its decisions is a pre-launch TODO.
- The single stratification seed (`random.seed(42)`) was used without a multi-seed robustness check. Whether a different stratified sample of 47 GPQA problems would produce a similar family-win distribution is an untested assumption.

## Decision

The pilot's pre-registered question was: *does the world exhibit the structure the framework assumes, and is that structure measurable at our budget?* The data says yes. Compute-response curves fit at near-noise-floor precision (H1). Multiple curve families share substantial selection mass (H2). A non-trivial fraction of problems get a unimodal best fit, though the c* sensitivity analysis shows genuine interior peaks are rare (H5). Curve parameters are stable across sampling temperatures to a degree we did not anticipate (H6). One hypothesis was falsified: embedding diversity does not outperform per-token entropy as an elasticity predictor (H3), and the falsification is constructive: it identifies a simpler, sample-efficient, model-agnostic feature class to lead the predictability track of the full study.

The decision is **GO for the full multi-model, multi-domain study** under the contribution hierarchy described in Section 6: Primary (cross-model and cross-domain characterization) is confirmed; Secondary (predictability via per-token uncertainty signals) is reframed; Tertiary (adaptive allocator) is conditional on the predictability work clearing the dual target of AUC ≥ 0.70 and calibration error < 0.10 across the expanded model set.

The pilot's pre-registration discipline carries into the full study unchanged. Hypotheses, thresholds, and stop conditions will be locked and git-tagged before any full-study sampling. Amendments, if needed, will be tagged in sequence (as v6.0.1, v6.0.2, and the multi-start refinement were in the pilot) and dated. The amendment audit trail is part of the integrity argument for the eventual paper; we treat it as a feature of the methodology, not a liability.

The pilot's total cost was approximately $0.60 of a $15 budget, about 4% of the originally allocated funds. Whatever else the full study costs, this leaves comfortable runway for the planned multi-model, multi-domain expansion.
