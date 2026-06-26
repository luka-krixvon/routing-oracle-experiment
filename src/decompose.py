"""Gap decomposition (paper/main.tex §III.B, the main contribution).

Honest reproducible ceiling = O_repro[i] = max_m p_hat[i,m]  (committable by a
per-query router).  Identity (always holds, noise term >= 0 since O_exp>=O_repro):

  G_single = mean_i (O_exp[i]  - q_router[i])
           = mean_i (O_repro[i] - q_router[i])   # (a) recoverable specialist advantage
           + mean_i (O_exp[i]   - O_repro[i])    # (b) single-draw label noise (>=0)
"""
from __future__ import annotations
import numpy as np
from . import oracles


def decompose_gap(p_hat: np.ndarray, q_router: np.ndarray,
                  O_exp: np.ndarray | None = None) -> dict:
    """Decompose the router-to-(expected single-draw)-oracle gap into recoverable
    specialist advantage vs single-draw label noise. tau-free (MAIN result).

    Pass the seed-aligned O^exp (oracles.oracle_expected_seed_aligned(b)) as `O_exp`
    for the dependence-aware estimand; if omitted, falls back to the independent-
    coupling upper envelope O^{exp,perp}=1-prod(1-p_hat) (valid only as a ceiling)."""
    p_hat = np.asarray(p_hat, dtype=float)
    q = np.asarray(q_router, dtype=float)
    if O_exp is None:
        O_exp = oracles.oracle_expected_perp_envelope(p_hat)   # envelope fallback
    else:
        O_exp = np.asarray(O_exp, dtype=float)
    O_repro = oracles.oracle_reproducible(p_hat)

    g_single = float(np.mean(O_exp - q))
    recoverable = float(np.mean(O_repro - q))      # (a)
    noise = float(np.mean(O_exp - O_repro))        # (b) >= 0
    noise_share = noise / g_single if g_single != 0 else float("nan")
    return {
        "O_exp_mean": float(O_exp.mean()),
        "O_repro_mean": float(O_repro.mean()),
        "router_mean": float(q.mean()),
        "G_single": g_single,
        "recoverable": recoverable,
        "noise": noise,
        "noise_share": noise_share,                # (b)/G_single  <-- headline
        "_per_query": {"O_exp": O_exp, "O_repro": O_repro, "q": q},
    }


def decompose_gap_conservative(b: np.ndarray, q_router: np.ndarray,
                               delta: float = 0.05, n_strata: int = 1) -> dict:
    """One-sided, winner's-curse-corrected conservative lower bound on the single-draw
    noise term (paper §V "Estimators & tests"). Uses the raw-frequency p_hat and the
    seed-aligned O^exp from the draw tensor b.

    Direction (the conservative, thesis-protecting choice):
      O_repro_upper = max_m p_hat + R_k     (ADD: max_m p_hat is UPWARD-biased for O^repro)
      O_exp_lower   = O^exp(seed-aligned) - R_k   (SUBTRACT)
      Delta_lower   = (O_exp_lower - O_repro_upper)_+   (SAME one-sided form every stratum
                       so the aggregate lower bound is valid termwise)
    Subtracting the radius from O_repro instead would be the anti-conservative bug."""
    from . import stats
    b = np.asarray(b, dtype=float)
    assert b.ndim == 3, "expected (N, M, k)"
    k, M = b.shape[2], b.shape[1]
    q = np.asarray(q_router, dtype=float)
    phat = oracles.estimate_p_hat_raw(b)
    Rk = stats.wcvar_radius(k, M, delta / (2.0 * max(n_strata, 1)))   # Bonferroni 2-delta split
    O_exp = oracles.oracle_expected_seed_aligned(b)
    O_repro = oracles.oracle_reproducible(phat)
    O_repro_upper = O_repro + Rk          # ADD
    O_exp_lower = O_exp - Rk              # SUBTRACT
    delta_lower = np.clip(O_exp_lower - O_repro_upper, 0.0, None)
    return {
        "R_k": float(Rk),
        "O_exp_mean": float(O_exp.mean()),
        "O_repro_mean": float(O_repro.mean()),
        "O_repro_upper_mean": float(O_repro_upper.mean()),
        "O_exp_lower_mean": float(O_exp_lower.mean()),
        "noise_point": float((O_exp - O_repro).mean()),   # uncorrected point estimate
        "Delta_lower_mean": float(delta_lower.mean()),    # conservative noise lower bound (>=0)
        "noise_recoverable_share_lower": float(
            delta_lower.mean() / max((O_exp - q).mean(), 1e-12)),
        "_per_query": {"O_exp": O_exp, "O_repro": O_repro, "Delta_lower": delta_lower},
    }


def threshold_sensitivity(p_hat: np.ndarray, tau: float) -> dict:
    """Auxiliary view: fraction of queries with a 'reliably correct' model
    (max_m p>=tau). Report across tau as a robustness check on 'how much of the
    pool is dependably useful', NOT as the main ceiling."""
    O_thr = oracles.oracle_threshold(p_hat, tau)
    return {"tau": tau, "reliable_frac": float(O_thr.mean())}


def recall_on_rare_correct(p_hat: np.ndarray, q_router: np.ndarray,
                           rare_max: int, tau: float) -> dict:
    """Re-evaluate 'model-recall failure' on the rare-correct stratum using a
    reproducible (tau) definition instead of single-draw counts."""
    p_hat = np.asarray(p_hat, dtype=float)
    nc_single = oracles.num_correct_models(p_hat >= 0.5, tau=None)  # crude single proxy
    nc_repro = oracles.num_correct_models(p_hat, tau=tau)
    rare_single = nc_single <= rare_max
    rare_repro = nc_repro <= rare_max
    q = np.asarray(q_router, dtype=float)
    return {
        "tau": tau,
        "rare_frac_single": float(rare_single.mean()),
        "rare_frac_repro": float(rare_repro.mean()),
        "router_recall_rare_single": float(q[rare_single].mean()) if rare_single.any() else float("nan"),
        "router_recall_rare_repro": float(q[rare_repro].mean()) if rare_repro.any() else float("nan"),
    }
