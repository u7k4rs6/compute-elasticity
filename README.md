# Compute Elasticity in LLM Reasoning

A pilot study measuring how **Qwen2.5-7B-Instruct** accuracy on **GPQA Diamond** responds to inference-time compute via best-of-N sampling at N ∈ {1, 2, 4, 8, 16, 32, 64}.

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
| 1 | Pure-Python modules + unit tests | pending |
| 2 | Pre-registration lock-in | pending |
| 3 | API smoke tests | pending |
| 4 | Gate -1: Embedder validation | pending |
| 5 | Day 0 reconnaissance | pending |
| 6 | Main pilot sampling | pending |
| 7 | Temperature side test | pending |
| 8 | Fitting | pending |
| 9 | Pass 1: confirmatory analysis | pending |
| 10 | Pass 2: exploratory + go/no-go | pending |

## License

Apache 2.0 — see `LICENSE`.
