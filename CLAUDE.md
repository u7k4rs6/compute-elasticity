# CLAUDE.md — Repository conventions for Claude Code

This file tells Claude Code how to work inside this repository. The canonical project spec is `PRD.md`; this file is the operational layer that sits on top.

**If anything in this file conflicts with `PRD.md`, the PRD wins. Surface the conflict and ask.**

---

## What this project is

A pilot research study characterizing how LLM reasoning accuracy responds to inference-time compute, using GPQA Diamond + Qwen2.5-7B-Instruct + best-of-N sampling. The pilot tests 6 pre-registered hypotheses (H1–H6) and produces a go/no-go decision for a larger multi-model study.

This is research code, not production code, but it is **pre-registered research code**. That means:
- Methodology is locked before data collection.
- Locked things stay locked.
- Departures require explicit re-tagging of `preregistration.md`.

---

## Key commands

```bash
# Install
pip install -r requirements.txt

# Pre-commit checks (run before any commit)
ruff check .
black --check .
pytest tests/ -v --tb=short

# Phase-specific entry points
python scripts/smoke_test.py                # Phase 3
python scripts/extraction_rate_check.py     # Phase 3
python scripts/run_gate_minus_1.py          # Phase 4
python scripts/run_recon.py                 # Phase 5
python scripts/run_main_pilot.py            # Phase 6
python scripts/run_side_test.py             # Phase 7
python scripts/run_fitting.py               # Phase 8
python scripts/run_diversity.py             # Phase 8
python scripts/run_analysis.py              # Phase 9
python scripts/validate_data.py outputs/    # any time

# Falsification suite (Phase 9 main test)
pytest tests/falsification.py -v
```

---

## Code conventions

- **Python:** 3.10+. Type hints on all public functions. `from __future__ import annotations` at top of every module.
- **Formatting:** `black` with default 88-char line length.
- **Linting:** `ruff` with default config.
- **Docstrings:** Google style. Every public function needs one.
- **Imports:** standard library, then third-party, then local. `isort`-compatible.
- **Magic numbers:** none in module bodies. All constants live in `pilot/config.py`.
- **Dataclasses:** use `@dataclass(frozen=True, slots=True)` for any record type.
- **Errors:** raise typed exceptions, not bare `Exception`. Inherit from a `PilotError` base when feature-specific.
- **Logging:** `logging` stdlib, not `print`. Module-level `logger = logging.getLogger(__name__)`. Scripts configure the root logger; modules never do.

---

## Testing discipline

- Every module in `pilot/` has a matching `tests/test_<module>.py`.
- All curve families in `pilot/fitting.py` MUST pass synthetic recovery tests. This is the single most important pre-pilot test — if it fails, nothing downstream is trustworthy.
- Mock the API layer in unit tests. Never make live calls from `tests/`.
- Live calls happen only via `scripts/`, never from inside library modules running under pytest.
- `tests/falsification.py` is special: it reads `outputs/` and runs the locked H1–H6 checks. It is a pytest file but represents the experiment's verdict, not a unit test of code correctness.

---

## Git workflow

- **Branches:** `main` is the only long-lived branch. Work in `feature/phase-N-<short-name>` or `fix/<issue>` branches.
- **Commits:** imperative-mood, prefix with task ID from PRD where applicable. Examples:
  - `T1.3: implement Gompertz family + synthetic recovery test`
  - `phase-2: lock preregistration, tag pre-pilot-v6.0`
  - `fix(sampling): retry on 429 with jitter`
- **Tags:** every phase completion gets a tag:
  - `phase-0-complete`, `phase-1-complete`, …, `phase-10-complete`
  - Plus the special `pre-pilot-v6.0` tag at end of Phase 2 (this is the pre-registration timestamp proof — it must be pushed to GitHub immediately and never deleted).
- **Never force-push to `main`.** Force-pushing branches before PR merge is fine.
- **Never rebase past the `pre-pilot-v6.0` tag.** The tag's commit must remain in `main` history forever.

### What stays out of git

`.gitignore` must exclude:
- `.env` (real API keys)
- `outputs/` (all generated data; results go to releases or HF dataset, not main branch)
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`
- `data/` cache (HuggingFace datasets cache reproducible from code)
- IDE files: `.vscode/`, `.idea/`

### What goes IN git

- All source code under `pilot/`, `tests/`, `scripts/`
- All `.md` documentation
- `requirements.txt`, `pyproject.toml`, `.env.example`
- `data/problem_ids.json` (the 50 locked problem IDs — small file, reproducibility-critical)
- `LICENSE`

---

## API key handling

- All keys read via `os.getenv("TOGETHER_API_KEY")` etc., never hardcoded.
- Missing key → raise `ConfigError` with explicit message naming the missing env var.
- Never log a key or any prefix of a key. If you write a logger call near key handling, manually verify nothing leaks.
- `.env.example` lists required vars with placeholder values. `.env` itself is gitignored.

---

## What NOT to do (anti-patterns)

These are the failure modes the pilot is explicitly designed to avoid. Refuse to do any of them, even if it seems like a small change:

### 1. Do not modify locked things after their lock point
- Prompt template (locked at Phase 2 / Phase 3 if extraction rate forces iteration)
- Hypothesis thresholds in `HYPOTHESIS_THRESHOLDS`
- The 5 curve families (don't add a 6th)
- Sample budget per problem (N values are fixed: {1, 2, 4, 8, 16, 32, 64})
- The 47 main pilot problems (don't swap, don't add)

If a lock needs to break, surface it to the user, document in `preregistration.md`, and re-tag with an explicit changelog. Never silently change.

### 2. Do not expand scope
Out-of-scope items (full list in `PRD.md` §2):
- Don't add a second model "just to compare"
- Don't add tree search "since it's easy"
- Don't add a fancier predictor feature
- Don't run a 3rd analysis pass
- Don't add new hypotheses beyond H1–H6

Every temptation goes to `TODO_full_study.md`. That file is the pressure-release valve.

### 3. Do not change analysis after seeing results
- Pass 1 (confirmatory) uses pre-locked thresholds. The thresholds were chosen before data; they cannot move after.
- If a result is *almost* significant, that's a fail. Reframe in the writeup, don't push the threshold.

### 4. Do not skip unit tests on curve fitting
The 5-family synthetic recovery test is the most important code-correctness check in the project. Never disable it. Never lower the recovery tolerance to make it pass. If it fails, fix the fit, not the test.

### 5. Do not silently swallow API errors
Retry with backoff (max 5 attempts) → on persistent failure, log loudly, switch providers, or halt the run. Never write a "fake" sample on failure. Better to have a missing sample than a fabricated one.

### 6. Do not commit `outputs/` or `.env`
The first is reproducibility-hostile (huge, regenerable). The second is a security catastrophe.

### 7. Do not run live API calls from inside `tests/`
Tests are mocked. Anything that touches money lives in `scripts/`.

### 8. Do not modify `preregistration.md` without re-tagging
That file is the timestamp proof. Edits without a new tag invalidate the pre-registration claim.

---

## When you're uncertain

Order of operations:
1. **Check `PRD.md`** — it's the canonical spec.
2. **Check this file** — it's the operational layer.
3. **Ask the user** — surface the question, propose 2-3 options with tradeoffs, wait. Don't guess on methodology decisions.

Specifically, ask before:
- Skipping or modifying any locked element
- Adding any dependency not in `requirements.txt`
- Switching API providers mid-run
- Touching `preregistration.md` after Phase 2
- Anything that costs more than $5 of API time without prior approval

---

## Style notes for documentation

- Keep `PILOT_WRITEUP.md` honest. H1–H6 results go first, exactly as observed. If H3 fails, say so plainly. The paper's credibility is built by reporting failures as cleanly as successes.
- Plots: always include error bars or CIs. Single-color minimalist; no chartjunk.
- Captions: include `[CONFIRMATORY]` or `[EXPLORATORY]` prefix.

---

## Definition of done (per phase)

A phase is done when:
1. Its acceptance test in `PRD.md` passes.
2. All new code passes `ruff` and `black`.
3. New code has matching tests passing.
4. Phase tag pushed to GitHub.

Only then start the next phase.

---

## Quick sanity check before each session

When resuming work after a break, run:

```bash
git status                         # clean?
git log --oneline -5               # where am I?
git tag | tail -5                  # last tagged phase
pytest tests/ -v --tb=short        # tests still pass?
```

Then re-read the relevant phase in `PRD.md` §5 before writing any code.
