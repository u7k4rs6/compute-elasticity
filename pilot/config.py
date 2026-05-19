"""Constants for the compute elasticity pilot.

All locked values live here. Never scatter magic numbers in module bodies.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model & providers
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen2.5-7B-Instruct"

TOGETHER_API_BASE = "https://api.together.xyz/v1"

# ---------------------------------------------------------------------------
# Compute grid
# ---------------------------------------------------------------------------
N_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
TEMPERATURE_MAIN: float = 0.7
TEMPERATURES_SIDE: tuple[float, ...] = (0.3, 0.7, 1.0)
N_SIDE_TEST: int = 16  # samples per problem in temperature side test

# ---------------------------------------------------------------------------
# Pilot scope
# ---------------------------------------------------------------------------
N_MAIN_PROBLEMS: int = 47
N_GATE_PROBLEMS: int = 3  # Gate-1 problems (part of the 47)
N_RECON_PROBLEMS: int = 10  # Day-0 recon subset (Gate-1 + 7 additional)
STRATIFIED_SEED: int = 42
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_N: int = 4  # N value at which diversity is computed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
SAMPLES_DIR: Path = OUTPUTS_DIR / "samples"
FITS_DIR: Path = OUTPUTS_DIR / "fits"
DIVERSITY_DIR: Path = OUTPUTS_DIR / "diversity"
FLAGS_DIR: Path = OUTPUTS_DIR / "flags"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
DATA_DIR: Path = REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# Cost guardrails
# ---------------------------------------------------------------------------
COST_HARD_CAP: float = 12.0  # USD; halt if cumulative cost exceeds CAP_WARN
COST_WARN_FRACTION: float = 0.80  # warn / halt at 80% of hard cap

# Together AI pricing (USD / token, 2026-05 snapshot)
TOGETHER_INPUT_PRICE_PER_TOKEN: float = 0.18 / 1_000_000
TOGETHER_OUTPUT_PRICE_PER_TOKEN: float = 0.18 / 1_000_000

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
MAX_CONCURRENT_CALLS: int = 32
MAX_RETRIES: int = 5
CHECKPOINT_INTERVAL: int = 5  # git checkpoint every N completed problems

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
N_FIT_POINTS: int = len(N_VALUES)  # = 7; n in the BIC formula
CURVE_FAMILIES: tuple[str, ...] = (
    "constant",
    "logistic",
    "gompertz",
    "shifted_logistic",
    "unimodal",
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_VERSION: str = "v6.0-pilot"

# ---------------------------------------------------------------------------
# Hypothesis thresholds — LOCKED after Phase 2.  Never modify mid-pilot.
# ---------------------------------------------------------------------------
HYPOTHESIS_THRESHOLDS: dict[str, object] = {
    "H1": {
        "metric": "median_residual_se",
        "pass_threshold": 0.10,
        "fail_threshold": 0.15,
    },
    "H2a": {
        "metric": "n_families_mean_bic_weight",
        "min_families": 2,
        "min_weight": 0.10,
    },
    "H2b": {
        "metric": "fraction_competing_families",
        "min_fraction": 0.30,
        "bic_margin": 2.0,
    },
    "H3": {
        "metric": "auc_embedding_vs_entropy",
        "pass_auc": 0.60,
        "pass_delta_auc": 0.03,
        "fail_auc": 0.55,
        "elasticity_c_range": (8, 64),
    },
    "H5": {
        "metric": "fraction_unimodal_wins",
        "pass_threshold": 0.05,
        "fail_threshold": 0.00,
    },
    "H6": {
        "metric": "bootstrap_ci_overlap_rate",
        "pass_threshold": 0.70,
        "fail_threshold": 0.50,
    },
}
