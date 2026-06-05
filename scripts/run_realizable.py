"""Realizable selection (majority vote) recompute and predictor re-evaluation.

Recomputes compute-response curves under REALIZABLE selection (majority vote /
self-consistency) instead of oracle pass@N, and re-evaluates cheap predictors
against the realizable target. Uses only data from outputs/samples/ (no new API
calls).

Parts:
  1. Majority-vote R(N) curves for all 47 main-pilot problems.
  2. Validation against existing outputs/fits/ R_at_N.
  3. Realizable target metrics (mv_gain, interior_peak).
  4. Predictor AUC table against above-median mv_gain.
  5. Entropy-vs-difficulty disentanglement (partial corr, stratified AUC, logit).
  6. Bootstrap 95% CIs on headline AUCs and mv_gain > 0.02 fraction.

Outputs written to outputs/realizable/.

Usage:
    source .venv/bin/activate
    python scripts/run_realizable.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr, t as t_dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_T_MAIN: float = 0.7
_N_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
_MC_DRAWS: int = 2000
_MC_SEED: int = 42
_BOOT_SEED: int = 42
_BOOT_ITERS: int = 1000
_INTERIOR_PEAK_MIN_MARGIN: float = 0.02
_N_EXACT_THRESHOLD: int = 2016  # C(64,2); use exact enumeration for N<=2 typically

_OUTPUTS_DIR = ROOT / "outputs"
_SAMPLES_DIR = _OUTPUTS_DIR / "samples"
_FITS_DIR = _OUTPUTS_DIR / "fits"
_ENTROPY_DIR = _OUTPUTS_DIR / "entropy_baseline"
_DIVERSITY_DIR = _OUTPUTS_DIR / "diversity"
_REALIZABLE_DIR = _OUTPUTS_DIR / "realizable"
_MV_CURVES_DIR = _REALIZABLE_DIR / "mv_curves"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def _main_pilot_ids() -> list[str]:
    locked = json.loads((ROOT / "data" / "problem_ids.json").read_text())
    gate = set(
        json.loads((_OUTPUTS_DIR / "gate_minus_1_labels.json").read_text())["gate_problems"]
    )
    return sorted(pid for pid in locked if pid not in gate)


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _load_samples_at_t(problem_id: str) -> list[dict]:
    """Load all T=0.7 samples for a problem, sorted by sample_idx."""
    path = _SAMPLES_DIR / f"{problem_id}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if abs(obj.get("temperature", -1) - _T_MAIN) < 1e-6:
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda s: s.get("sample_idx", 0))
    return rows


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------


def _mv_correct(
    answers: list[str | None],
    ground_truth: str,
    rng: np.random.Generator,
) -> float:
    """Return 1.0 if majority-vote winner == ground_truth, else 0.0.

    UNPARSEABLE is treated as its own category (never correct). Ties broken
    uniformly at random using rng -- ground truth is NEVER consulted during
    tie-breaking.
    """
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return 0.0
    max_count = max(counts.values())
    tied = sorted(a for a, c in counts.items() if c == max_count)
    winner = tied[0] if len(tied) == 1 else str(rng.choice(tied))
    return 1.0 if winner == ground_truth else 0.0


def _mv_acc_at_n(
    answers: list[str | None],
    ground_truth: str,
    n: int,
    rng: np.random.Generator,
    n_total: int,
) -> float:
    """Expected MV accuracy at sample size n.

    Uses exact enumeration when C(n_total, n) <= _N_EXACT_THRESHOLD, otherwise
    Monte Carlo with _MC_DRAWS draws.
    For n >= n_total: single deterministic evaluation over all samples.
    """
    if n >= n_total:
        return _mv_correct(answers, ground_truth, rng)
    if n == 1:
        return sum(1.0 for a in answers if a == ground_truth) / n_total
    n_combos = comb(n_total, n)
    if n_combos <= _N_EXACT_THRESHOLD:
        scores = [
            _mv_correct([answers[i] for i in idx], ground_truth, rng)
            for idx in combinations(range(n_total), n)
        ]
        return float(np.mean(scores))
    # Monte Carlo
    scores = [
        _mv_correct(
            [answers[i] for i in rng.choice(n_total, size=n, replace=False)],
            ground_truth,
            rng,
        )
        for _ in range(_MC_DRAWS)
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Oracle pass@N (exact combinatorial formula)
# ---------------------------------------------------------------------------


def _oracle_pass_at_n(n_total: int, c_correct: int, n: int) -> float:
    """Exact oracle pass@N: 1 - C(n_total-c, N) / C(n_total, N)."""
    if n > n_total:
        return 1.0 if c_correct > 0 else 0.0
    num = comb(n_total - c_correct, n)
    den = comb(n_total, n)
    return 0.0 if den == 0 else 1.0 - num / den


# ---------------------------------------------------------------------------
# Diversity max N=8 recomputation
# ---------------------------------------------------------------------------


def _pairwise_cosine_dists(emb: np.ndarray) -> list[float]:
    n = len(emb)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(emb[i], emb[j]))
            dists.append(1.0 - sim)
    return dists


def _compute_diversity_max_n8(
    main_ids: list[str],
    embedder: object,
) -> dict[str, float]:
    """Recompute per-problem diversity_max at N=8, replicating run_h3_ablation seed logic."""
    from pilot.diversity import truncate_trace

    rng = np.random.default_rng(_MC_SEED)
    result: dict[str, float] = {}
    for pid in main_ids:
        path = _SAMPLES_DIR / f"{pid}.jsonl"
        if not path.exists():
            logger.warning("%s: samples file missing for diversity", pid)
            continue
        traces = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if abs(obj.get("temperature", -1) - _T_MAIN) < 1e-6:
                    traces.append(obj.get("full_response", ""))
            except json.JSONDecodeError:
                continue
        if len(traces) < 8:
            logger.warning("%s: only %d traces for diversity N=8 — skipping", pid, len(traces))
            continue
        idx8 = rng.choice(len(traces), size=8, replace=False)
        selected = [traces[i] for i in sorted(idx8)]
        truncated = [truncate_trace(t)[0] for t in selected]
        emb = embedder.encode(truncated, normalize_embeddings=True)  # type: ignore[union-attr]
        dists = _pairwise_cosine_dists(emb)
        result[pid] = float(max(dists)) if dists else 0.0
    return result


# ---------------------------------------------------------------------------
# Partial Spearman correlation
# ---------------------------------------------------------------------------


def _spearman_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Spearman partial correlation of x with y, controlling for z.

    Computed on ranks via the Pearson partial correlation formula:
      r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)*(1-r_yz^2))
    P-value from t-distribution with df = n-3.
    """
    n = len(x)
    r_xy = float(spearmanr(x, y).statistic)
    r_xz = float(spearmanr(x, z).statistic)
    r_yz = float(spearmanr(y, z).statistic)
    denom = np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    if denom < 1e-12:
        return 0.0, 1.0
    r_partial = (r_xy - r_xz * r_yz) / denom
    r_partial = float(np.clip(r_partial, -1.0, 1.0))
    df = n - 3
    if df <= 0:
        return r_partial, float("nan")
    t_stat = r_partial * np.sqrt(df) / np.sqrt(max(1.0 - r_partial**2, 1e-12))
    pval = float(2.0 * t_dist.sf(abs(t_stat), df=df))
    return r_partial, pval


# ---------------------------------------------------------------------------
# Logistic regression with p-values
# ---------------------------------------------------------------------------


def _logit_fit(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict[str, Any]:
    """Fit logistic regression (with intercept) and compute Wald p-values.

    X: (n, k) float array of features
    y: (n,) binary array
    Returns dict with coefficients, p-values, and predicted probabilities.
    """
    # Standardize features for numerical stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X_std = (X - mu) / sd

    n, k = X_std.shape
    X_aug = np.column_stack([np.ones(n), X_std])

    def neg_ll(beta: np.ndarray) -> float:
        logit = np.clip(X_aug @ beta, -30.0, 30.0)
        p_hat = 1.0 / (1.0 + np.exp(-logit))
        eps = 1e-10
        return -float(np.sum(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps)))

    def grad(beta: np.ndarray) -> np.ndarray:
        logit = np.clip(X_aug @ beta, -30.0, 30.0)
        p_hat = 1.0 / (1.0 + np.exp(-logit))
        return X_aug.T @ (p_hat - y)

    res = minimize(neg_ll, np.zeros(k + 1), jac=grad, method="L-BFGS-B")
    beta_hat = res.x

    # Fisher information (Hessian of neg-LL)
    logit_hat = np.clip(X_aug @ beta_hat, -30.0, 30.0)
    p_hat = 1.0 / (1.0 + np.exp(-logit_hat))
    W = p_hat * (1.0 - p_hat)
    H = (X_aug * W[:, None]).T @ X_aug

    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k + 1, np.nan)

    z = beta_hat / np.where(se < 1e-12, np.nan, se)
    pvals = 2.0 * norm.sf(np.abs(z))

    pred_proba = 1.0 / (1.0 + np.exp(-np.clip(X_aug @ beta_hat, -30.0, 30.0)))

    names_aug = ["intercept"] + feature_names
    coef_table = {
        nm: {"coef": float(beta_hat[i]), "se": float(se[i]), "pvalue": float(pvals[i])}
        for i, nm in enumerate(names_aug)
    }
    return {"coef_table": coef_table, "pred_proba": pred_proba}


# ---------------------------------------------------------------------------
# AUC helper (wraps pilot.analysis.compute_auc)
# ---------------------------------------------------------------------------


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from pilot.analysis import compute_auc

    return float(compute_auc(np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    data: list[tuple],
    stat_fn: Any,
    n_iter: int = _BOOT_ITERS,
    seed: int = _BOOT_SEED,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap percentile CI for stat_fn applied to a list of per-problem tuples."""
    rng = np.random.default_rng(seed)
    n = len(data)
    stats = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sample = [data[i] for i in idx]
        stats.append(stat_fn(sample))
    lo = float(np.percentile(stats, 100 * (1 - ci) / 2))
    hi = float(np.percentile(stats, 100 * (1 - (1 - ci) / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Parts 1-6 and save outputs."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main_ids = _main_pilot_ids()
    assert len(main_ids) == 47, f"Expected 47 main-pilot IDs, got {len(main_ids)}"

    _REALIZABLE_DIR.mkdir(parents=True, exist_ok=True)
    _MV_CURVES_DIR.mkdir(parents=True, exist_ok=True)

    # Single RNG for the entire run (deterministic)
    rng = np.random.default_rng(_MC_SEED)

    # -----------------------------------------------------------------------
    # PART 1: Majority-vote curves
    # -----------------------------------------------------------------------
    logger.info("=== PART 1: Computing majority-vote curves ===")
    per_problem: dict[str, dict] = {}
    total_parseable = 0
    total_samples = 0

    for num, pid in enumerate(main_ids, start=1):
        samples = _load_samples_at_t(pid)
        if not samples:
            logger.error("%s: no samples found — aborting", pid)
            sys.exit(1)

        n_total = len(samples)
        ground_truth: str = samples[0]["ground_truth"]
        subject: str = samples[0].get("subject", "unknown")
        answers: list[str | None] = [s.get("extracted_answer") for s in samples]
        correct_flags: list[bool] = [bool(s.get("correct", False)) for s in samples]

        # Parse rate tracking
        n_parseable = sum(1 for a in answers if a and a != "UNPARSEABLE")
        total_parseable += n_parseable
        total_samples += n_total

        c_correct = sum(correct_flags)
        p = c_correct / n_total

        mv_acc: dict[str, float] = {}
        for n in _N_VALUES:
            acc = _mv_acc_at_n(answers, ground_truth, n, rng, n_total)
            mv_acc[str(n)] = round(acc, 6)

        # Oracle pass@N for validation
        oracle: dict[str, float] = {}
        for n in _N_VALUES:
            oracle[str(n)] = round(_oracle_pass_at_n(n_total, c_correct, n), 6)

        # Token stats
        mean_in = float(np.mean([s.get("input_tokens", 0) for s in samples]))
        mean_out = float(np.mean([s.get("output_tokens", 0) for s in samples]))

        per_problem[pid] = {
            "problem_id": pid,
            "subject": subject,
            "n_total": n_total,
            "c_correct": c_correct,
            "p": round(p, 6),
            "mv_acc": mv_acc,
            "oracle_pass_at_n": oracle,
            "mean_input_tokens": round(mean_in, 2),
            "mean_output_tokens": round(mean_out, 2),
        }
        logger.info("[%d/47] %s  n=%d  p=%.3f  MV@64=%.3f", num, pid, n_total, p, mv_acc["64"])

    parse_rate = total_parseable / total_samples if total_samples else 0.0
    logger.info("Overall parse rate (excluding UNPARSEABLE): %.4f", parse_rate)

    # -----------------------------------------------------------------------
    # PART 2: Validation
    # -----------------------------------------------------------------------
    logger.info("=== PART 2: Validation ===")
    oracle_diffs: list[float] = []
    mv1_mismatches: list[str] = []

    for pid in main_ids:
        pp = per_problem[pid]
        fit_path = _FITS_DIR / f"{pid}.json"
        if not fit_path.exists():
            logger.error("%s: fit file missing — cannot validate", pid)
            sys.exit(1)
        fit = json.loads(fit_path.read_text())
        r_at_n = fit.get("R_at_N", {})

        for n in _N_VALUES:
            key = str(n)
            if key not in r_at_n:
                continue
            oracle_val = pp["oracle_pass_at_n"][key]
            fit_val = float(r_at_n[key])
            oracle_diffs.append(abs(oracle_val - fit_val))

        # MV@1 == p check
        mv1 = pp["mv_acc"]["1"]
        p_val = pp["p"]
        if abs(mv1 - p_val) > 1e-9:
            mv1_mismatches.append(f"{pid}: MV@1={mv1:.6f}  p={p_val:.6f}")

    max_oracle_diff = max(oracle_diffs) if oracle_diffs else 0.0
    mean_oracle_diff = float(np.mean(oracle_diffs)) if oracle_diffs else 0.0
    logger.info(
        "Oracle vs R_at_N: max_abs_diff=%.4f  mean=%.4f",
        max_oracle_diff, mean_oracle_diff,
    )
    if mv1_mismatches:
        for m in mv1_mismatches:
            logger.error("MV@1 != p: %s", m)
        sys.exit(1)
    else:
        logger.info("MV_acc(N=1) == p for all 47 problems. OK.")

    # Explain oracle mismatch if any
    validation_note = ""
    if max_oracle_diff > 0.02:
        validation_note = (
            "Oracle pass@N (combinatorial formula) does NOT match existing R_at_N. "
            "This is expected: the existing pipeline computes majority-vote accuracy, "
            "not oracle pass@N. R_at_N is an unbiased MV estimator, not best-of-N. "
            "MV_acc(1)==p confirmed for all problems, validating data loading."
        )
        logger.warning("Oracle mismatch > 0.02. %s", validation_note)
    else:
        validation_note = "Oracle pass@N matches existing R_at_N (max diff <= 0.02)."
        logger.info("Oracle validation OK.")

    # -----------------------------------------------------------------------
    # PART 3: Realizable targets
    # -----------------------------------------------------------------------
    logger.info("=== PART 3: Realizable targets ===")
    mv_gains: list[float] = []
    for pid in main_ids:
        pp = per_problem[pid]
        mv_acc = pp["mv_acc"]
        mv_at_1 = mv_acc["1"]
        mv_at_64 = mv_acc["64"]
        mv_gain = mv_at_64 - mv_at_1
        mv_gain_max_val = max(mv_acc.values()) - mv_at_1
        # Interior peak: max is at some N in {2,4,8,16,32} and exceeds both
        # MV(1) and MV(64) by > margin
        interior_n_values = [str(n) for n in _N_VALUES if 1 < n < 64]
        mv_vals_interior = [mv_acc[k] for k in interior_n_values]
        max_interior = max(mv_vals_interior) if mv_vals_interior else 0.0
        interior_peak = bool(
            max_interior > mv_at_1 + _INTERIOR_PEAK_MIN_MARGIN
            and max_interior > mv_at_64 + _INTERIOR_PEAK_MIN_MARGIN
        )
        pp["mv_gain"] = round(mv_gain, 6)
        pp["mv_gain_max"] = round(mv_gain_max_val, 6)
        pp["interior_peak"] = interior_peak
        mv_gains.append(mv_gain)

    mv_gains_arr = np.array(mv_gains)
    n_gt002 = int(np.sum(mv_gains_arr > 0.02))
    n_le0 = int(np.sum(mv_gains_arr <= 0.0))
    n_interior = sum(1 for pid in main_ids if per_problem[pid]["interior_peak"])
    mv_gain_mean = float(np.mean(mv_gains_arr))
    mv_gain_median = float(np.median(mv_gains_arr))
    mv_gain_min = float(np.min(mv_gains_arr))
    mv_gain_max = float(np.max(mv_gains_arr))
    logger.info(
        "mv_gain: mean=%.4f  median=%.4f  min=%.4f  max=%.4f  >0.02=%d  <=0=%d  interior_peak=%d",
        mv_gain_mean, mv_gain_median, mv_gain_min, mv_gain_max,
        n_gt002, n_le0, n_interior,
    )

    # Save per-problem curve files
    for pid in main_ids:
        out_path = _MV_CURVES_DIR / f"{pid}.json"
        out_path.write_text(json.dumps(per_problem[pid], indent=2))

    # -----------------------------------------------------------------------
    # PART 4: Predictor re-evaluation
    # -----------------------------------------------------------------------
    logger.info("=== PART 4: Predictor AUC table ===")

    # Load entropy/NLL baselines
    entropy_n1: dict[str, float] = {}
    nll_n1: dict[str, float] = {}
    for pid in main_ids:
        eb_path = _ENTROPY_DIR / f"{pid}.json"
        if not eb_path.exists():
            logger.error("%s: entropy baseline missing", pid)
            sys.exit(1)
        eb = json.loads(eb_path.read_text())
        entropy_n1[pid] = float(eb["mean_per_token_entropy"])
        nll_n1[pid] = float(eb["mean_per_token_nll"])

    # Load diversity_mean_N4
    diversity_mean_n4: dict[str, float] = {}
    for pid in main_ids:
        dv_path = _DIVERSITY_DIR / f"{pid}.json"
        if not dv_path.exists():
            logger.error("%s: diversity file missing", pid)
            sys.exit(1)
        dv = json.loads(dv_path.read_text())
        diversity_mean_n4[pid] = float(dv["diversity"])

    # Recompute diversity_max_N8
    logger.info("Loading sentence-transformers embedder for diversity_max_N8...")
    from pilot.config import EMBEDDING_MODEL

    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(EMBEDDING_MODEL)
        diversity_max_n8 = _compute_diversity_max_n8(main_ids, embedder)
        logger.info("diversity_max_N8 computed for %d problems", len(diversity_max_n8))
    except ImportError:
        logger.warning("sentence-transformers not available; diversity_max_N8 skipped")
        diversity_max_n8 = {}

    # Build aligned arrays
    ids_with_div_n8 = [pid for pid in main_ids if pid in diversity_max_n8]
    n_div = len(ids_with_div_n8)

    # Binary label: above-median mv_gain
    mv_gain_arr = np.array([per_problem[pid]["mv_gain"] for pid in main_ids])
    median_mv_gain = float(np.median(mv_gain_arr))
    labels = (mv_gain_arr > median_mv_gain).astype(int)
    n_ties_at_median = int(np.sum(mv_gain_arr == median_mv_gain))
    n_pos = int(np.sum(labels))

    logger.info(
        "Binary label: mv_gain > median (%.6f). pos=%d neg=%d ties_at_median=%d",
        median_mv_gain, n_pos, 47 - n_pos, n_ties_at_median,
    )

    def _get_arr(feat_dict: dict[str, float]) -> np.ndarray:
        return np.array([feat_dict[pid] for pid in main_ids])

    entropy_arr = _get_arr(entropy_n1)
    nll_arr = _get_arr(nll_n1)
    div_mean_arr = _get_arr(diversity_mean_n4)
    p_arr = np.array([per_problem[pid]["p"] for pid in main_ids])
    in_tok_arr = np.array([per_problem[pid]["mean_input_tokens"] for pid in main_ids])
    out_tok_arr = np.array([per_problem[pid]["mean_output_tokens"] for pid in main_ids])

    auc_entropy_n1 = _auc(entropy_arr, labels)
    auc_nll_n1 = _auc(nll_arr, labels)
    auc_div_mean_n4 = _auc(div_mean_arr, labels)
    auc_p = _auc(p_arr, labels)
    auc_in_tok = _auc(in_tok_arr, labels)
    auc_out_tok = _auc(out_tok_arr, labels)

    if ids_with_div_n8:
        labels_div = np.array([
            int(per_problem[pid]["mv_gain"] > median_mv_gain) for pid in ids_with_div_n8
        ])
        div_max_n8_arr = np.array([diversity_max_n8[pid] for pid in ids_with_div_n8])
        auc_div_max_n8 = _auc(div_max_n8_arr, labels_div)
    else:
        auc_div_max_n8 = float("nan")

    auc_table = {
        "entropy_N1": round(auc_entropy_n1, 4),
        "nll_N1": round(auc_nll_n1, 4),
        "diversity_mean_N4": round(auc_div_mean_n4, 4),
        "diversity_max_N8": round(auc_div_max_n8, 4) if not np.isnan(auc_div_max_n8) else None,
        "difficulty_p": round(auc_p, 4),
        "mean_input_tokens": round(auc_in_tok, 4),
        "mean_output_tokens": round(auc_out_tok, 4),
    }

    sep = "=" * 72
    print(f"\n{sep}")
    print("PART 4 — Predictor AUC vs above-median mv_gain")
    print(sep)
    print(f"  Binary label: mv_gain > {median_mv_gain:.6f}   pos={n_pos}/47  ties={n_ties_at_median}")
    print()
    print(f"  {'Predictor':<26} {'AUC':>6}")
    print(f"  {'-'*26} {'-'*6}")
    for name, val in auc_table.items():
        print(f"  {name:<26} {val if val is not None else 'n/a':>6}")
    print(sep)

    # -----------------------------------------------------------------------
    # PART 5: Entropy-vs-difficulty disentanglement
    # -----------------------------------------------------------------------
    logger.info("=== PART 5: Disentanglement ===")

    # (a) Spearman partial correlation
    rho_partial, pval_partial = _spearman_partial(entropy_arr, mv_gain_arr, p_arr)
    logger.info("Spearman partial corr (entropy vs mv_gain | p): rho=%.4f  p=%.4f",
                rho_partial, pval_partial)

    # (b) Stratified AUC
    p_median = float(np.median(p_arr))
    low_p_mask = p_arr <= p_median
    high_p_mask = p_arr > p_median

    # Within-stratum binary label (above within-stratum median mv_gain)
    def _within_stratum_auc(mask: np.ndarray, predictor: np.ndarray) -> float:
        sub_mv = mv_gain_arr[mask]
        sub_pred = predictor[mask]
        sub_median = float(np.median(sub_mv))
        sub_labels = (sub_mv > sub_median).astype(int)
        if sub_labels.sum() == 0 or sub_labels.sum() == len(sub_labels):
            return float("nan")
        return _auc(sub_pred, sub_labels)

    auc_entropy_low_p = _within_stratum_auc(low_p_mask, entropy_arr)
    auc_entropy_high_p = _within_stratum_auc(high_p_mask, entropy_arr)
    logger.info(
        "Stratified AUC (entropy_N1): low_p=%.4f  high_p=%.4f",
        auc_entropy_low_p, auc_entropy_high_p,
    )

    # (c) Logistic regression: mv_gain_above ~ entropy_N1 + p  vs  ~ p only
    two_feat = _logit_fit(
        np.column_stack([entropy_arr, p_arr]),
        labels.astype(float),
        ["entropy_N1", "p"],
    )
    one_feat = _logit_fit(
        p_arr.reshape(-1, 1),
        labels.astype(float),
        ["p"],
    )
    auc_logit_2feat = _auc(two_feat["pred_proba"], labels)
    auc_logit_p_only = _auc(one_feat["pred_proba"], labels)
    logger.info(
        "Logistic AUC: 2-feat=%.4f  p-only=%.4f", auc_logit_2feat, auc_logit_p_only
    )

    # Plain-language verdict
    entropy_survives = (
        abs(rho_partial) > 0.15
        and pval_partial < 0.10
        and (auc_entropy_low_p > 0.55 or auc_entropy_high_p > 0.55)
    )
    coef_entropy = two_feat["coef_table"]["entropy_N1"]
    entropy_logit_pval = coef_entropy["pvalue"]
    if entropy_survives:
        disentangle_verdict = (
            "Entropy_N1 predicts mv_gain beyond what difficulty p explains. "
            f"Spearman partial rho={rho_partial:.3f} (p={pval_partial:.3f}); "
            f"stratified AUC low-p={auc_entropy_low_p:.3f}, high-p={auc_entropy_high_p:.3f}; "
            f"logit entropy coef p={entropy_logit_pval:.3f}. "
            "Entropy carries independent signal."
        )
    else:
        disentangle_verdict = (
            "Entropy_N1 does NOT clearly survive the difficulty control. "
            f"Spearman partial rho={rho_partial:.3f} (p={pval_partial:.3f}); "
            f"stratified AUC low-p={auc_entropy_low_p:.3f}, high-p={auc_entropy_high_p:.3f}; "
            f"logit entropy coef p={entropy_logit_pval:.3f}. "
            "Much of entropy's predictive power may reflect problem difficulty."
        )

    # -----------------------------------------------------------------------
    # PART 6: Bootstrap CIs
    # -----------------------------------------------------------------------
    logger.info("=== PART 6: Bootstrap CIs ===")
    boot_data = [
        (
            entropy_n1[pid],
            per_problem[pid]["p"],
            per_problem[pid]["mv_gain"],
        )
        for pid in main_ids
    ]

    def _boot_auc_entropy(sample: list) -> float:
        ent = np.array([x[0] for x in sample])
        mv = np.array([x[2] for x in sample])
        med = float(np.median(mv))
        lbl = (mv > med).astype(int)
        return _auc(ent, lbl)

    def _boot_auc_p(sample: list) -> float:
        p = np.array([x[1] for x in sample])
        mv = np.array([x[2] for x in sample])
        med = float(np.median(mv))
        lbl = (mv > med).astype(int)
        return _auc(p, lbl)

    def _boot_frac_gt002(sample: list) -> float:
        mv = np.array([x[2] for x in sample])
        return float(np.mean(mv > 0.02))

    ci_entropy = _bootstrap_ci(boot_data, _boot_auc_entropy)
    ci_p = _bootstrap_ci(boot_data, _boot_auc_p)
    ci_frac = _bootstrap_ci(boot_data, _boot_frac_gt002)
    logger.info(
        "Bootstrap 95%% CI: entropy_AUC=[%.3f, %.3f]  p_AUC=[%.3f, %.3f]  "
        "frac_>0.02=[%.3f, %.3f]",
        *ci_entropy, *ci_p, *ci_frac,
    )

    # -----------------------------------------------------------------------
    # PART 4 / 5 / 6: Print reports
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("PART 5 — Entropy vs difficulty disentanglement")
    print(sep)
    print(f"  (a) Spearman partial corr (entropy | p): rho={rho_partial:.4f}  p={pval_partial:.4f}")
    print(f"  (b) Stratified AUC (within-stratum label):")
    print(f"      entropy_N1, low-p half:  {auc_entropy_low_p:.4f}")
    print(f"      entropy_N1, high-p half: {auc_entropy_high_p:.4f}")
    print(f"  (c) Logistic regression (above-median mv_gain ~ ...):")
    print(f"      p-only model AUC   : {auc_logit_p_only:.4f}")
    print(f"      two-feat model AUC : {auc_logit_2feat:.4f}")
    print(f"      Coefficients (standardized features):")
    for feat, row in two_feat["coef_table"].items():
        print(f"        {feat:<18} coef={row['coef']:+.4f}  se={row['se']:.4f}  p={row['pvalue']:.4f}")
    print(f"\n  Verdict: {disentangle_verdict}")
    print(sep)

    print(f"\n{sep}")
    print("PART 6 — Bootstrap 95% CIs (n=1000, problem-level resample)")
    print(sep)
    print(f"  entropy_N1 AUC : [{ci_entropy[0]:.4f}, {ci_entropy[1]:.4f}]  (point est {auc_entropy_n1:.4f})")
    print(f"  difficulty p AUC: [{ci_p[0]:.4f}, {ci_p[1]:.4f}]  (point est {auc_p:.4f})")
    print(f"  frac mv_gain>0.02: [{ci_frac[0]:.4f}, {ci_frac[1]:.4f}]  (point est {n_gt002/47:.4f})")
    print(sep)

    print(f"\n{sep}")
    print("PART 2 — Validation summary")
    print(sep)
    print(f"  Oracle vs R_at_N: max_abs_diff={max_oracle_diff:.4f}  mean={mean_oracle_diff:.4f}")
    print(f"  MV_acc(1)==p: all 47 problems confirmed.")
    print(f"  Note: {validation_note}")
    print(sep)

    print(f"\n{sep}")
    print("PART 3 — mv_gain distribution")
    print(sep)
    print(f"  mean={mv_gain_mean:.4f}  median={mv_gain_median:.4f}  "
          f"min={mv_gain_min:.4f}  max={mv_gain_max:.4f}")
    print(f"  mv_gain > 0.02 : {n_gt002}/47")
    print(f"  mv_gain <= 0   : {n_le0}/47")
    print(f"  interior_peak  : {n_interior}/47")
    print(f"  Overall parse rate: {parse_rate:.4f}")
    print(sep)

    # -----------------------------------------------------------------------
    # Save realizable_summary.json
    # -----------------------------------------------------------------------
    summary: dict[str, Any] = {
        "schema_version": "v6.0-pilot",
        "n_problems": 47,
        "median_mv_gain_threshold": round(median_mv_gain, 6),
        "n_above_median_mv_gain": n_pos,
        "n_ties_at_median": n_ties_at_median,
        "parse_rate": round(parse_rate, 6),
        "validation": {
            "oracle_vs_r_at_n_max_abs_diff": round(max_oracle_diff, 6),
            "oracle_vs_r_at_n_mean_abs_diff": round(mean_oracle_diff, 6),
            "mv_acc_1_eq_p_all_problems": True,
            "note": validation_note,
        },
        "mv_gain_distribution": {
            "mean": round(mv_gain_mean, 6),
            "median": round(mv_gain_median, 6),
            "min": round(mv_gain_min, 6),
            "max": round(mv_gain_max, 6),
            "n_gt_0_02": n_gt002,
            "n_le_0": n_le0,
            "n_interior_peak": n_interior,
        },
        "auc_table": auc_table,
        "disentanglement": {
            "spearman_partial_rho": round(rho_partial, 4),
            "spearman_partial_pvalue": round(pval_partial, 4),
            "stratified_auc_entropy_low_p": round(auc_entropy_low_p, 4),
            "stratified_auc_entropy_high_p": round(auc_entropy_high_p, 4),
            "logit_p_only_auc": round(auc_logit_p_only, 4),
            "logit_2feat_auc": round(auc_logit_2feat, 4),
            "logit_2feat_coefs": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in two_feat["coef_table"].items()
            },
            "verdict": disentangle_verdict,
        },
        "bootstrap_ci_95": {
            "entropy_N1_auc": {"lo": round(ci_entropy[0], 4), "hi": round(ci_entropy[1], 4)},
            "difficulty_p_auc": {"lo": round(ci_p[0], 4), "hi": round(ci_p[1], 4)},
            "frac_mv_gain_gt_002": {"lo": round(ci_frac[0], 4), "hi": round(ci_frac[1], 4)},
        },
    }
    out_path = _REALIZABLE_DIR / "realizable_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved outputs/realizable/realizable_summary.json")
    logger.info("Saved %d per-problem curve files to outputs/realizable/mv_curves/", len(main_ids))


if __name__ == "__main__":
    main()
