# When Self-Consistency Backfires: Confidence Does Not Track Correctness on Hard Reasoning

*Draft for a workshop submission (double-blind: anonymize author block before submitting).*

---

## Abstract

Self-consistency (majority voting over sampled chains of thought) is widely treated as a low-risk way to spend inference compute for higher accuracy. We show that on graduate-level science multiple-choice questions (GPQA Diamond), majority voting net-harms per-problem accuracy on a large fraction of problems: 47% of problems for Qwen2.5-7B-Instruct and 66% for Llama-3-8B-Instruct, two models from different families. The harm is structural: voting concentrates on a wrong plurality whenever a model's single-sample accuracy on a problem is below one half. An oracle that routes each problem between one sample and full voting could recover 7 to 11 accuracy points, so the headroom to avoid the harm exists. That headroom is not cheaply accessible, however: a deploy-time gate based on agreement among a few probe samples captures essentially none of it (about 0% to 3%), because high agreement does not imply correctness. In the highest-agreement bin, the plurality answer is still wrong 44% (Qwen) and 50% (Llama) of the time. We conclude that on hard reasoning problems self-consistency can hurt, the harm is real and sizeable, and cheap confidence signals do not tell you when. This is a small, single-benchmark study, and we frame the result accordingly. Code and data are released.

---

## 1. Introduction

Inference-time compute has become a primary lever for improving LLM reasoning, and self-consistency (sampling several chains of thought and returning the majority answer) is among the simplest and most widely used techniques in this family. It is commonly assumed to be a near-free accuracy boost: spend more samples, get a more reliable answer.

This paper documents a failure mode of that assumption and shows that it is hard to detect cheaply. On a graduate-level science benchmark, majority voting reduces per-problem accuracy on a large fraction of problems, because for any problem where the model is correct on fewer than half of its samples, voting consolidates a wrong answer. We then ask the practical follow-up question an efficiency-minded practitioner would ask: can a cheap signal computed from a handful of samples tell you which problems to vote on and which to skip? We find the answer is largely no, and we identify why.

Our contributions are:

1. **A quantified, replicated backfire effect.** Majority voting net-harms per-problem accuracy on 47% of GPQA Diamond problems for Qwen2.5-7B-Instruct and 66% for Llama-3-8B-Instruct. The effect replicates across two model families, and the per-model 95% confidence intervals do not overlap.

2. **Evidence that the headroom is real but not cheaply reachable.** An oracle that perfectly routes each problem between a single sample and full voting recovers 7.4 to 11.4 accuracy points, but a deploy-time gate based on probe-sample agreement captures essentially none of this (about 0% to 3%), and on the weaker model the gate can drop below the fixed-budget baseline.

3. **A direct mechanism: confidence does not track correctness.** Binning problems by how concentrated the model's answers are, we find that even in the highest-agreement bin the plurality answer is wrong 44% (Qwen) and 50% (Llama) of the time. High self-consistency is indistinguishable from confidently-wrong consensus, which is precisely why an agreement-based gate fails.

We are explicit about scope: this is two small non-reasoning models on one benchmark of 47 problems, and several of our estimates are noisy at that size. We report confidence intervals throughout and discuss the limitations in Section 6.

---

## 2. Related Work

**Self-consistency and chain-of-thought.** Chain-of-thought prompting (Wei et al., 2022) and self-consistency decoding (Wang et al., 2023), which marginalizes over sampled reasoning paths by majority vote, are standard tools for improving reasoning accuracy. We study exactly this majority-vote procedure and the conditions under which it helps or hurts.

**Inference-time scaling.** A growing literature characterizes how accuracy scales with sampled compute, including repeated-sampling analyses (Brown et al., 2024) and compute-optimal test-time strategies (Snell et al., 2024). The unbiased pass@k estimator (Chen et al., 2021) measures the probability that at least one of k samples is correct, i.e. best-of-N under an oracle verifier; we deliberately do not use it as our primary metric, because it measures attainable accuracy under perfect selection rather than the realizable accuracy of majority voting, which is what a verifier-free deployment actually gets.

**Verification and selection.** Outcome and process verifiers (Cobbe et al., 2021; Lightman et al., 2023) can in principle select correct answers from a sample set, which is the deployable analogue of the oracle routing we use as an upper bound. Our negative result concerns the verifier-free case, where the only signal is agreement among the model's own samples.

**Calibration and confidence.** Modern networks are often miscalibrated and overconfident (Guo et al., 2017), and language models have a limited but real ability to report their own uncertainty (Kadavath et al., 2022). Our mechanism finding is a self-consistency-specific instance of this: agreement among samples behaves like a confidence signal that is poorly calibrated to correctness on hard problems.

**Adaptive sampling.** Adaptive-consistency methods stop sampling early when the running majority is stable (Aggarwal et al., 2023), saving compute at little accuracy cost. Our agreement gate is in this family. Our contribution is not a new stopping rule but the finding that, on hard problems, such agreement-based rules cannot recover the accuracy headroom that voting leaves on the table, because the agreement they rely on does not indicate correctness.

**Benchmark.** We use GPQA Diamond (Rein et al., 2023), a set of graduate-level, Google-proof science multiple-choice questions chosen to be genuinely hard for current models.

*(Bibliography to be completed and verified before submission; the above are canonical references cited by author and year.)*

---

## 3. Setup

**Data.** We use the GPQA Diamond subset, stratified by subject. After reserving three problems for embedder validation in an earlier phase, we evaluate on the remaining 47 problems. All quantitative claims below are over these 47 problems.

**Models.** Two instruction-tuned models from different families, both served on Together AI: `Qwen2.5-7B-Instruct-Turbo` and `Meta-Llama-3-8B-Instruct-Lite`. Both are small (7 to 8B), non-reasoning models.

**Sampling.** For each problem we draw independent completions at temperature 0.7 under a single fixed prompt template (its SHA-256 hash is logged and unchanged across the study). Qwen has 65 to 72 samples per problem (accumulated across earlier phases); Llama has 64 per problem from a single run. Each completion is scored by a five-pass answer extractor; the unparseable rate is 1.9% for Qwen and 1.9% for Llama, and unparseable completions are counted as incorrect.

**Metrics.** For a budget of N samples, let MV_acc(N) be the expected accuracy of the majority-vote (plurality) answer over N samples, with ties broken uniformly at random independent of the ground truth. We estimate MV_acc(N) by Monte Carlo over sampled subsets (2000 draws, fixed seed; exact enumeration where the number of subsets is small), reusing all available samples per problem. We define the **self-consistency gain** of a problem as

> mv_gain = MV_acc(64) − MV_acc(1),

the change in accuracy from a single sample to full voting, and we say voting **backfires** on a problem when mv_gain < 0.

**Routing and gating.** As an upper bound on what any per-problem routing could achieve, we define an **oracle gate** that uses ground truth to choose, per problem, the better of one sample and full voting. As a deployable counterpart, we define an **agreement gate**: draw k probe samples, and if the plurality fraction among them is at least a threshold tau, stop and return that plurality (cost k); otherwise continue to full voting (cost 64). The agreement gate uses no ground truth in its decision; ground truth is used only to score the result. We sweep tau to trace an accuracy-versus-compute curve.

**Uncertainty.** We report 95% confidence intervals by problem-level bootstrap (resampling the 47 problems with replacement, 1000 iterations, fixed seed).

---

## 4. Results

### 4.1 Self-consistency backfires on a large fraction of problems

Majority voting reduces per-problem accuracy on a substantial fraction of problems for both models (Table 1, Figure 1). The backfire rate is 46.8% (22 of 47) for Qwen and 66.0% (31 of 47) for Llama; the 95% confidence intervals, [31.9%, 61.7%] and [51.1%, 80.9%], do not overlap, so the effect is not a single-model artifact and is in fact stronger on Llama.

The mechanism is structural. Voting returns the plurality answer, which is correct only when the model is correct on a majority of its samples. For any problem where single-sample accuracy is below one half, voting concentrates on a wrong answer and does worse than a single sample in expectation. The worst single case in our data loses 41 accuracy points.

The aggregate picture and the per-problem picture diverge in an instructive way. For Qwen, voting helps on average (aggregate accuracy rises from 0.408 at N=1 to 0.506 at N=64) yet still harms 47% of individual problems. For Llama, voting is nearly useless on aggregate (0.333 to 0.340) and harms two-thirds of problems individually, the gains and losses very nearly cancelling. An average improvement can therefore hide widespread per-problem harm.

**Table 1. Per-model summary (47 GPQA Diamond problems).**

| | Qwen2.5-7B-Instruct | Llama-3-8B-Instruct |
|---|---|---|
| Single-sample accuracy (N=1) | 0.408 | 0.333 |
| Self-consistency accuracy (N=64) | 0.506 | 0.340 |
| Backfire rate (mv_gain < 0) | 46.8% (22/47) | 66.0% (31/47) |
| Backfire rate, 95% CI | [31.9%, 61.7%] | [51.1%, 80.9%] |
| Oracle-routing accuracy (upper bound) | 0.580 | 0.455 |
| Oracle ceiling gain over N=64 | +7.4 pp | +11.4 pp |
| Oracle mean compute (samples/problem) | 31.8 | 22.5 |
| Agreement gate accuracy (k=8, tau=0.75) | 0.503 | 0.340 |
| Oracle ceiling captured by agreement gate | ~0% | ~3% |

*(Figure 1: `backfire_both.png` — distribution of mv_gain for both models, with the backfire region marked.)*

### 4.2 The headroom is real

The harm is not inevitable in principle. An oracle that routes each problem to the better of one sample or full voting reaches 0.580 accuracy for Qwen and 0.455 for Llama, gains of 7.4 and 11.4 points over always voting, and at lower mean compute (31.8 and 22.5 samples per problem rather than 64). So there is real accuracy to be recovered by deciding per problem whether to vote, if one could make that decision well.

### 4.3 But cheap gating cannot capture it

The deployable agreement gate does not recover this headroom (Figure 2). At k=8, tau=0.75 the gate reaches 0.503 accuracy for Qwen at 41.2 samples per problem, roughly a 35% compute saving over always voting at a cost of 0.3 accuracy points. Measured against the oracle ceiling, however, it captures about 0% of the available gain for Qwen and about 3% for Llama, and the bootstrap intervals on this quantity are wide and span zero. On Llama the gate is worse than neutral: at intermediate thresholds its accuracy drops below the matched fixed-budget baseline, because the high-agreement probes it trusts are reliably wrong.

In short, the agreement gate buys modest compute savings at a small accuracy cost, which is the known benefit of early stopping, but it does not and cannot capture the accuracy headroom that the oracle shows is present.

*(Figure 2: `pareto_both.png` — accuracy versus mean compute for fixed-budget voting, the oracle gate, and the agreement gate, both models.)*

### 4.4 Why: confidence does not track correctness

The reason the gate fails is direct (Table 2, Figure 3). We bin problems by a confidence proxy, the fraction of all samples that agree on the plurality answer, and report how often the plurality is correct in each bin. If agreement indicated correctness, the highest-agreement bin would be near-perfectly correct. It is not. In the highest-confidence bin (plurality fraction at least 0.75), the plurality answer is wrong 44% of the time for Qwen and 50% of the time for Llama. For Llama the relationship is essentially flat to inverted: mid-confidence problems are no more accurate than low-confidence ones.

High self-consistency is therefore indistinguishable from confidently-wrong consensus on these problems. Any gate that decides whether to trust a vote using agreement among the model's own samples is reading a signal that does not carry the information it needs.

**Table 2. Confidence does not track correctness.** Confidence is the fraction of samples agreeing on the plurality answer; the value reported is the fraction of problems in each bin whose plurality answer is correct. Bin counts are small, so these fractions are approximate.

| Confidence bin | Qwen n | Qwen % plurality correct | Llama n | Llama % plurality correct |
|---|---|---|---|---|
| [0.25, 0.50) | 18 | 38.9% | 18 | 27.8% |
| [0.50, 0.75) | 13 | 61.5% | 15 | 26.7% |
| [0.75, 1.00] | 16 | 56.3% | 14 | 50.0% |

*(Figure 3: `calibration_both.png` — confidence bin versus fraction of plurality answers correct, both models, with a diagonal reference line.)*

---

## 5. Discussion

The practical takeaway is narrow and, we think, useful. Self-consistency is not a free accuracy boost on hard problems: it harmed roughly half of the problems for one model and two-thirds for another, and an average gain can conceal that. Whether voting helps a given problem is governed by whether the model's single-sample accuracy on that problem exceeds one half, which is exactly the quantity a deployment cannot observe without ground truth. The natural cheap substitute, agreement among a few samples, does not work, because on hard problems the model is frequently confident and wrong, and confidence and correctness come apart.

This suggests that recovering the routing headroom requires a signal external to the model's own agreement, for example a trained verifier or process reward model, rather than a confidence statistic derived from repeated sampling. We do not test such signals here; we only show that the cheapest verifier-free option is insufficient.

---

## 6. Limitations

We state these plainly, as they bound every claim above.

- **One benchmark.** All results are on GPQA Diamond, graduate-level science multiple choice. We do not test other domains (math, code, general knowledge) or easier difficulty regimes, where the backfire rate and the calibration picture could differ.
- **Small sample.** With 47 problems, several estimates are noisy. The backfire rates are well separated across models, but the oracle-ceiling-capture estimates have wide intervals that span zero, and the calibration bins contain only 13 to 18 problems each, so their fractions should be read as approximate, not precise.
- **Two small, non-reasoning models.** Both models are 7 to 8B instruction-tuned models. Whether backfire shrinks for larger or reasoning-tuned models, which may be better calibrated, is an open and important question we did not test.
- **One selection rule.** We study majority voting. Other verifier-free aggregations (for example weighted or confidence-weighted voting) and verifier-guided selection may behave differently; the oracle gate is an upper bound, not a deployable method.
- **Mechanism is a known phenomenon in a new setting.** That confidence is imperfectly calibrated to correctness is established; our contribution is the quantified, replicated, self-consistency-specific demonstration and its direct consequence for agreement-based gating, not the bare existence of miscalibration.

---

## 7. Conclusion

On hard reasoning problems, self-consistency can hurt rather than help, and it does so on a large and replicable fraction of problems across two model families. The accuracy this leaves on the table is real, but it is not recoverable with a cheap, verifier-free agreement signal, because high agreement does not imply correctness. Practitioners should not assume self-consistency is safe on hard inputs, and should not expect agreement among samples to tell them when it is. Whether a stronger external verifier, or larger and better-calibrated models, can close the gap is the natural next question.

---

*Artifacts: code, raw samples, fitted curves, and analysis outputs are released (repository link to be added after de-anonymization). All figures and tables are reproducible from the released outputs with a fixed seed.*
