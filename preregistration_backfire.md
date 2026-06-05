# Pre-Registration: Self-Consistency Backfire (Confirmatory)

**Status:** Locked before confirmatory analysis. Git-tag this file before computing
any result on the confirmatory set.

## Design

The original 47 GPQA Diamond problems are **exploratory**: the hypotheses below were
generated from them. The 151 remaining GPQA Diamond problems are the **confirmatory**
test set. Every result is reported on exploratory (47), confirmatory (151), and pooled
(198), but PASS/FAIL is decided on the confirmatory set only.

**Models:** Qwen2.5-7B-Instruct-Turbo, Meta-Llama-3-8B-Instruct-Lite.
**Sampling:** N=64 per problem, T=0.7, single locked prompt template (SHA-256 unchanged).
**Majority vote:** plurality answer; ties broken uniformly at random, independent of ground truth.

## Definitions

- mv_gain = MV_acc(64) - MV_acc(1). **Backfire** = mv_gain < 0.
- **Oracle gate:** ground-truth-optimal per-problem routing between N=1 and N=64 (upper bound only, not deployable).
- **Agreement gate:** stop on probe plurality fraction >= tau (verifier-free).
- **Entropy gate:** route by a mean per-token entropy threshold (verifier-free).
- **Ceiling captured** = (gate_acc - fixed_64_acc) / (oracle_acc - fixed_64_acc).

## Confirmatory hypotheses

Thresholds are set below the exploratory point estimates so each is a genuine prediction.

- **PH1 (backfire prevalence).** Majority vote backfires on >= 33% of confirmatory
  problems for BOTH models. Exploratory: Qwen 46.8%, Llama 66.0%. Falsified for a
  model if its confirmatory backfire rate < 33%.

- **PH2 (agreement gate fails).** The agreement gate (k=8, tau=0.75) captures <= 10%
  of the oracle ceiling for BOTH models. Exploratory: Qwen ~0%, Llama 2.7%. Falsified
  for a model if capture > 10%.

- **PH3 (confidence does not track correctness).** In the highest-confidence bin
  (plurality fraction >= 0.75), the plurality answer is correct <= 70% of the time for
  BOTH models. Exploratory: Qwen 56.3%, Llama 50.0%. Falsified for a model if > 70%.

- **PH4 (uncertainty signal also fails; contingent on logprobs).** An entropy gate
  captures <= 10% of the oracle ceiling for BOTH models. Exploratory: not tested.
  Falsified for a model if capture > 10%. If logprobs are unavailable for a model,
  PH4 is not evaluated for that model, and this is reported.

## Reporting and discipline

Report each hypothesis PASS/FAIL on the confirmatory set with 95% bootstrap CIs
(problem-level, 1000 iterations, seed 42), and report exploratory and pooled values
alongside for transparency. Confirmatory analysis is limited to PH1-PH4; anything
else is labeled exploratory.
