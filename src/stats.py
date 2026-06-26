"""Uncertainty + tests (paper/main.tex §IV).

- bootstrap_ci      : percentile CI over queries (epistemic over the test set)
- posterior_resample: draw p ~ Beta posterior (sampling uncertainty in p_hat)
- nested_ci         : combine both — bootstrap queries AND resample p posteriors
- mcnemar_test      : paired binary (rare-correct recall before/after correction)
"""
from __future__ import annotations
import math
import numpy as np


def bootstrap_ci(values: np.ndarray, B: int = 1000, alpha: float = 0.05,
                 seed: int = 0, statfn=np.mean) -> tuple[float, float, float]:
    """Percentile bootstrap CI of statfn over the 1-D sample `values`."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    n = len(v)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        stats[b] = statfn(v[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(statfn(v)), float(lo), float(hi)


def posterior_resample(b_counts: np.ndarray, k: int, prior=(1.0, 1.0),
                       seed: int = 0) -> np.ndarray:
    """Draw one posterior sample of p (N, M) from Beta(a0+correct, b0+k-correct)."""
    rng = np.random.default_rng(seed)
    a0, b0 = prior
    a = a0 + np.asarray(b_counts, dtype=float)
    bb = b0 + (k - np.asarray(b_counts, dtype=float))
    return rng.beta(a, bb)


def nested_ci(b_tensor: np.ndarray, statfn_from_phat, B: int = 1000,
              alpha: float = 0.05, prior=(1.0, 1.0), seed: int = 0):
    """Two-layer CI: for each bootstrap of queries, also draw a Beta posterior of
    p, so the interval reflects both test-set and per-cell sampling uncertainty.

    b_tensor : (N, M, k) 0/1.  statfn_from_phat: callable(p_hat (N,M)) -> scalar.
    """
    rng = np.random.default_rng(seed)
    b = np.asarray(b_tensor, dtype=float)
    N, M, k = b.shape
    counts = b.sum(axis=2)
    stats = np.empty(B)
    for t in range(B):
        idx = rng.integers(0, N, N)
        p_draw = posterior_resample(counts[idx], k, prior, seed=int(rng.integers(1 << 31)))
        stats[t] = statfn_from_phat(p_draw)
    point = statfn_from_phat(counts / k)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def wcvar_radius(k: int, M: int, delta: float = 0.05) -> float:
    """Winner's-curse / order-statistic radius R_k = sqrt( ln(max(M,2)) / (2k) ).

    Used ONE-SIDED and conservatively (paper §V): ADD R_k to O^repro_upper (since
    max_m p_hat is already UPWARD-biased for O^repro) and SUBTRACT R'_k from
    O^exp_lower. `delta` is the per-side error after a Bonferroni split; it can scale
    the radius if a tail-exact form is desired (default keeps the Hoeffding form)."""
    return math.sqrt(math.log(max(M, 2)) / (2.0 * max(k, 1)))


def gate_known_p(b: np.ndarray, seed: int = 0, max_tv: float = 0.06) -> dict:
    """GATE A (paper §V): known-p simulation. Compare the observed per-draw
    #-correct-models distribution against an INDEPENDENT Bernoulli(p_hat) simulation
    (the Poisson-binomial null), using total-variation distance (sample-size robust,
    unlike a raw chi-square at N*k observations). Cross-model coupling (e.g. shared
    easy/hard structure) shifts the observed distribution away from the independent
    null. Pass if TV <= max_tv. A FAILED gate means the magnitude study reports nothing.

    We compare PER-DRAW counts (#models correct on one draw), not '>=1 correct in k':
    the latter conditions on p_hat>0 and is biased even for clean i.i.d. data."""
    b = np.asarray(b, dtype=float)
    N, M, k = b.shape
    phat = b.sum(axis=2) / k
    rng = np.random.default_rng(seed)
    # simulate 5x the draws to keep the null histogram low-variance, then normalize
    b_sim = (rng.random((N, M, 5 * k)) < phat[:, :, None]).astype(float)
    bins = np.arange(0, M + 2)
    o, _ = np.histogram(b.sum(axis=1).reshape(-1).astype(int), bins=bins)
    s, _ = np.histogram(b_sim.sum(axis=1).reshape(-1).astype(int), bins=bins)
    o = o / max(o.sum(), 1); s = s / max(s.sum(), 1)
    tv = 0.5 * float(np.sum(np.abs(o - s)))
    return {"pass": tv <= max_tv, "tv": tv, "max_tv": max_tv}


def _runs_z(seq: np.ndarray) -> float:
    """Wald-Wolfowitz runs-test z statistic for a 0/1 sequence (0 if degenerate)."""
    seq = np.asarray(seq).astype(int)
    n = len(seq); n1 = int(seq.sum()); n0 = n - n1
    if n1 == 0 or n0 == 0:
        return 0.0
    runs = 1 + int(np.sum(seq[1:] != seq[:-1]))
    mu = 1.0 + 2.0 * n1 * n0 / n
    var = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n)) / (n * n * (n - 1)) if n > 1 else 0.0
    return (runs - mu) / math.sqrt(var) if var > 0 else 0.0


def gate_independence(b: np.ndarray, max_overdisp: float = 1.25,
                      min_in_band: float = 0.90) -> dict:
    """GATE B, tests A1 (paper §V): per-cell over-dispersion of the draw sequence vs
    the Bernoulli variance p(1-p), plus a Wald-Wolfowitz runs test, to catch provider
    caching / non-i.i.d. draws. Pass if mean over-dispersion <= max_overdisp AND the
    fraction of cells with |runs z| < 1.96 is >= min_in_band (≈0.95 under i.i.d.).
    Returns {'pass','overdisp','frac_in_band'}. Only cells with 0<p_hat<1 are used."""
    b = np.asarray(b, dtype=float)
    N, M, k = b.shape
    phat = b.sum(axis=2) / k
    var_obs = b.var(axis=2)                       # (N,M), ddof=0
    var_ber = phat * (1.0 - phat)
    mask = (phat > 0) & (phat < 1)
    if not mask.any():
        return {"pass": True, "overdisp": float("nan"), "frac_in_band": float("nan"),
                "note": "no informative cells"}
    overdisp = float(np.mean(var_obs[mask] / np.clip(var_ber[mask], 1e-9, None)))
    idx = np.argwhere(mask)
    zs = np.array([_runs_z(b[i, m]) for i, m in idx])
    frac_in_band = float(np.mean(np.abs(zs) < 1.96))
    return {"pass": (overdisp <= max_overdisp) and (frac_in_band >= min_in_band),
            "overdisp": overdisp, "frac_in_band": frac_in_band, "n_cells": int(mask.sum())}


def mcnemar_test(before: np.ndarray, after: np.ndarray):
    """Paired binary McNemar test (e.g., router hit on rare-correct queries,
    single-draw vs reproducible-oracle labeling). Returns (statistic, p_value)."""
    before = np.asarray(before).astype(bool)
    after = np.asarray(after).astype(bool)
    b01 = int(np.sum(~before & after))
    b10 = int(np.sum(before & ~after))
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[0, b01], [b10, 0]], exact=(b01 + b10) < 25)
        return float(res.statistic), float(res.pvalue)
    except Exception:  # fallback: continuity-corrected chi-square
        if b01 + b10 == 0:
            return 0.0, 1.0
        stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        from math import erfc, sqrt
        return float(stat), float(erfc(sqrt(stat / 2)))
