# Self-Consistency Backfire: Majority Vote Hurts on Nearly Half of Expert-Level Problems

**Abstract.** Self-consistency (SC) via majority vote is the dominant approach for scaling LLM inference-time compute: run N forward passes, return the plurality answer. We show that this strategy _backfires_ on a substantial fraction of expert-level multiple-choice questions: scaling from N=1 to N=64 decreases accuracy on 47% of problems for Qwen2.5-7B and on 66% for Llama-3-8B, across 47 GPQA Diamond problems. A perfect oracle gate over {N=1, N=64} would lift accuracy by 7.4 percentage points (pp) for Qwen and 11.4 pp for Llama, but a deploy-time agreement gate using k probe samples captures approximately 0% (Qwen) and 2.7% (Llama) of that ceiling. Calibration analysis confirms the mechanism: even in the highest-confidence bin, plurality is wrong 44% (Qwen) and 50% (Llama) of the time. Confident and wrong are indistinguishable.

---

## 1. Introduction

Scaling inference compute by running more forward passes and taking a majority vote is a simple, model-agnostic technique that reliably improves aggregate accuracy on benchmarks [@wei2022chain; @wang2023self]. The implicit assumption is that the correct answer is a plurality attractors: enough passes, and the right answer wins the vote.

That assumption breaks on hard expert-level problems. For a problem where the model assigns the highest probability to an _incorrect_ answer across many independent samples, adding more samples only entrenches the wrong vote. We call this _backfire_: mv_gain = MV\_acc(64) - MV\_acc(1) < 0. The correct answer was a minority view, and majority vote amplified the majority error.

Backfire is not a new observation in isolation, but its _prevalence_ and _mechanism_ have not been characterized quantitatively across two independently-family models at the same problem set. We address that gap here.

**Contributions:**

1. We characterize the backfire rate on GPQA Diamond at N=64 for two models from different families (Qwen2.5-7B and Llama-3-8B), finding rates of 47% and 66% respectively.
2. We measure the oracle gate upper bound and show that a realistic deploy-time agreement gate captures essentially none of it (0% and 2.7%).
3. We confirm the mechanism via calibration analysis: agreement fraction (confidence) is a poor proxy for correctness even at high values.

---

## 2. Setup

**Dataset.** We use 47 problems from the GPQA Diamond benchmark [@rein2023gpqa], the main-pilot split used in our pre-registered study (the remaining 3 are held out for gate calibration). GPQA Diamond contains graduate-level multiple-choice questions across biology, chemistry, and physics, designed to be above the ability of non-experts with internet access.

**Models.** Two 7-8B parameter instruction-tuned models from different families:
- Qwen2.5-7B-Instruct-Turbo (Qwen family)
- Meta-Llama-3-8B-Instruct-Lite (Llama-3 family)

Both use temperature T=0.7, N=64 independent samples per problem, and the same locked prompt template (SHA-256 verified). Parse rate: 98.0% (Qwen), 98.1% (Llama).

**Majority vote.** For each N in {1, 2, 4, 8, 16, 32, 64}, mean MV accuracy is computed by: for each problem, enumerate all C(n\_total, N) subsets of size N (exact) or draw 2000 Monte Carlo subsets (if C > 5000), take the plurality answer, compare to ground truth, average over problems.

**Gate simulation.** The _agreement gate_ uses k probe samples: compute probe\_agreement = count(plurality answer) / k. If probe\_agreement >= tau, stop and return the probe plurality; else run full N=64 MV. Mean compute = k * stop\_rate + 64 * (1 - stop\_rate). We sweep k in {4, 8} and tau in [0.50, 1.00] with 2000 Monte Carlo draws per problem (seed 42 for k=4, seed 43 for k=8).

---

## 3. Results

### 3.1 Backfire is Prevalent and Replicates Across Model Families

![EXPLORATORY - Backfire distributions for Qwen2.5-7B (left) and Llama-3-8B (right). Dark bars: backfire problems (mv\_gain < 0). Dashed vertical line at zero.](../outputs/gate_model2/backfire_both.png)

**Figure 1** [EXPLORATORY]. Histogram of mv\_gain = MV\_acc(64) - MV\_acc(1) for each model. Dark gray bars show problems where scaling hurts (mv\_gain < 0). For Qwen, backfire affects 22/47 = 47% of problems (95% bootstrap CI: 32-62%). For Llama, it affects 31/47 = 66% of problems (95% CI: 51-81%). The distributions are distinctly bimodal: either scaling helps substantially (gains up to +63pp for Qwen, +63pp for Llama) or it hurts (losses as deep as -41pp for Qwen, -45pp for Llama).

### 3.2 Oracle Ceiling Is Large, Realistic Gate Captures Almost None

![EXPLORATORY - Accuracy vs mean compute per problem for both models. Star: oracle gate upper bound. Solid: fixed-budget SC. Dashed/dotted: agreement gate sweeps for k=4 and k=8.](../outputs/gate_model2/pareto_both.png)

**Figure 2** [EXPLORATORY]. Accuracy vs mean compute (samples/problem). For Qwen (left), fixed-budget SC improves from 40.8% at N=1 to 50.6% at N=64. The oracle gate (selecting the better of N=1 vs N=64 per problem with ground truth) reaches 58.0% at mean compute 31.8 -- a 7.4 pp gain over fixed N=64. Agreement gate curves (k=4 dashed, k=8 dotted) stay close to or below the fixed-budget curve at matched compute: they Pareto-dominate fixed-budget at some operating points, but by small margins. For Llama (right), fixed-budget SC yields only marginal net gain (33.3% at N=1 to 34.0% at N=64, +0.7 pp) with a non-monotone curve that dips below the N=2 peak at intermediate N, consistent with a 66% backfire rate. The oracle gate reaches 45.5% at mean compute 22.5 -- an 11.4 pp gain. Agreement gate curves fail to Pareto-dominate the fixed-budget curve.

### 3.3 High Confidence Does Not Imply Correctness

![EXPLORATORY - Calibration: fraction of problems where the plurality answer is correct, by confidence bin. Reference diagonal (dotted) shows perfect calibration. Horizontal dashed line at 0.5.](../outputs/gate_model2/calibration_both.png)

**Figure 3** [EXPLORATORY]. Calibration of confidence (probe\_agreement = max vote fraction over full N samples) against accuracy. For Qwen (circles, solid line), the highest-confidence bin [0.75, 1.00] contains n=16 problems, of which 56.3% have the correct plurality. For Llama (squares, dashed line), the highest-confidence bin [0.75, 1.00] contains n=14 problems, of which exactly 50.0% are correct -- equivalent to a coin flip. Both models are far below the perfect-calibration diagonal. The agreement gate uses confidence as its stopping criterion; the calibration data explain why it fails.

---

## 4. Cross-Model Summary Table

**Table 1.** Cross-model comparison. All accuracy values are mean MV accuracy over 47 problems. Backfire CI is 95% bootstrap CI (problem-level resample, 1000 iterations). Oracle ceiling captured = fraction of the (oracle - fixed\_64) gap recovered by the realistic gate.

| Metric | Qwen2.5-7B | Llama-3-8B |
|---|---|---|
| N=1 accuracy | 40.8% | 33.3% |
| N=64 MV accuracy | 50.6% | 34.0% |
| Backfire rate | 47% [32%, 62%] | 66% [51%, 81%] |
| Oracle gate accuracy | 58.0% | 45.5% |
| Oracle mean compute | 31.8 samples | 22.5 samples |
| Oracle gain over N=64 | +7.4 pp | +11.4 pp |
| Agreement gate (k=8, tau=0.75): accuracy | 50.3% | 34.0% |
| Agreement gate (k=8, tau=0.75): compute | 41.2 samples | 44.0 samples |
| Oracle ceiling captured | ~0% | 2.7% |

**Table 2.** Calibration detail by confidence bin. Confidence = max vote fraction across all N=64 samples for each problem.

| Confidence bin | Qwen n | Qwen frac correct | Llama n | Llama frac correct |
|---|---|---|---|---|
| [0.25, 0.50) | 18 | 38.9% | 18 | 27.8% |
| [0.50, 0.75) | 13 | 61.5% | 15 | 26.7% |
| [0.75, 1.00] | 16 | 56.3% | 14 | 50.0% |

---

## 5. Discussion

**Why does backfire occur?** On hard problems, the model's sampling distribution places most probability mass on an incorrect answer. At N=1, the question is whether the single sample is correct. At N=64, if the incorrect answer holds plurality in most subsets, MV locks in the error. Backfire is not a sampling artifact; it reflects a model's per-problem answer distribution being concentrated on a wrong answer. Expert-level problems are designed to have plausible distractors [@rein2023gpqa], which likely amplify this effect.

**Why does the agreement gate fail?** The gate asks: "did k probe samples agree?" and interprets high agreement as a signal to stop and trust the probe plurality. But agreement measures concentration of the sampling distribution, not alignment with the correct answer. When the model is consistently wrong -- the backfire regime -- it will also agree consistently. The calibration data (Table 2, Figure 3) make this concrete: even in the highest-confidence bin, Llama's plurality answer is wrong half the time.

**Comparison to prior work.** [@wang2023self] demonstrated SC gains on GSM8K, MATH, and commonsense benchmarks -- datasets where backfire is presumably rare because models have higher baseline accuracy. [@snell2024scaling] characterize compute-optimal test-time scaling strategies but focus on problems where accuracy is monotone in compute. [@brown2024large] analyze best-of-N sampling curves; their pass@N oracle metric is analytically tractable but does not decompose the majority vote vs oracle gap as we do here. The backfire phenomenon is closely related to calibration failures studied in [@guo2017calibration] and the "model knows what it knows" literature [@kadavath2022language], but prior work does not specifically quantify its effect on the majority-vote accuracy curve.

**Limitations.** Results are on 47 problems from one benchmark with two models; broader generalization requires more problems and more models. GPQA Diamond is unusually hard relative to typical deployed tasks. Bootstrap CIs on backfire rates are wide (20-30 pp), reflecting the small problem count. The calibration bins contain 13-18 problems each, too few for precise estimates.

**Implications.** For practitioners deploying majority-vote SC on expert-level tasks: (1) assume the backfire rate is non-trivial; (2) do not use agreement fraction as a proxy for correctness; (3) the oracle ceiling is large but unreachable with agreement-based gating -- more expressive routing signals are needed. For researchers: developing a deploy-time signal that distinguishes "confidently correct" from "confidently wrong" would unlock the oracle ceiling and is the key open problem.

---

## 6. Related Work

Chain-of-thought prompting [@wei2022chain] and self-consistency [@wang2023self] establish the foundation for reasoning via sampling. Process reward models [@lightman2023verify] provide per-step supervision but require additional training. Outcome reward model training [@cobbe2021training] similarly requires labeled solutions. Adaptive consistency [@aggarwal2023adaptive] proposes early stopping based on answer stability, which is equivalent to the agreement gate we evaluate here; our results provide evidence that agreement stability is insufficient on hard problems. Best-of-N sampling [@chen2021codex; @brown2024large] optimizes for pass@N (oracle) rather than majority vote; the two metrics diverge substantially in the backfire regime as we show.

---

## References

[@aggarwal2023adaptive]: Aggarwal et al. 2023. Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning with LLMs.

[@brown2024large]: Brown et al. 2024. Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.

[@chen2021codex]: Chen et al. 2021. Evaluating Large Language Models Trained on Code.

[@cobbe2021training]: Cobbe et al. 2021. Training Verifiers to Solve Math Word Problems.

[@guo2017calibration]: Guo et al. 2017. On Calibration of Modern Neural Networks.

[@kadavath2022language]: Kadavath et al. 2022. Language Models (Mostly) Know What They Know.

[@lightman2023verify]: Lightman et al. 2023. Let's Verify Step by Step.

[@rein2023gpqa]: Rein et al. 2023. GPQA: A Graduate-Level Google-Proof Q&A Benchmark.

[@snell2024scaling]: Snell et al. 2024. Scaling LLM Test-Time Compute Optimally.

[@wang2023self]: Wang et al. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models.

[@wei2022chain]: Wei et al. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
