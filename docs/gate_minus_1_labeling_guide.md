# Gate -1 Labeling Guide

Manual labeling guide for Phase 4 (Gate -1): embedder validation.

## Task

For each pair of reasoning traces from `outputs/gate_minus_1/`, assign one of three labels:

- **same-strategy**: Both traces follow the same high-level reasoning approach (e.g., both eliminate answers by testing against known facts, both compute numerically)
- **different-strategy**: Traces use meaningfully different reasoning approaches (e.g., one reasons from first principles, the other pattern-matches to known examples)
- **ambiguous**: Cannot clearly distinguish strategy or responses are too short to judge

## Label definitions

### same-strategy
- Both traces use the same type of reasoning (deductive, elimination, calculation, analogy)
- The structure of the argument is similar, even if surface wording differs
- One is not clearly a "different path" to the answer

### different-strategy
- The traces start from different premises or use different inferential machinery
- One might use domain knowledge recall, the other systematic elimination
- A domain expert would recognize them as genuinely different approaches

### ambiguous
- Traces are too short or generic to classify (e.g., both just say "the answer is X")
- One or both traces are mostly incoherent
- The distinction is unclear even after careful reading

## Procedure

1. Read both traces fully before labeling
2. Focus on the reasoning process, not the final answer
3. If the same answer is reached by different paths → different-strategy
4. Record labels in `outputs/gate_minus_1_labels.json`

## Expected output format

```json
[
  {
    "problem_id": "gpqa_diamond_001",
    "trace_a_idx": 0,
    "trace_b_idx": 1,
    "label": "different-strategy",
    "notes": "Trace A uses elimination; Trace B calculates directly"
  }
]
```
