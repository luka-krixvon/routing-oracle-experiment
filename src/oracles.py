"""Per-instance oracle estimators under stochastic decoding.

Notation (see paper/main.tex §II):
  b[i, m, j] in {0,1} : correctness of model m on query i, sample j (T>0)
  p[i, m]            : reproducible correctness probability (latent), estimated p_hat
  Oracles per query i:
    single   : max_m b_single[i, m]                      (one realized draw)
    expected : 1 - prod_m (1 - p[i, m])  =  E[single]    (=> upward-biased ceiling)
    threshold: 1{ max_m p[i, m] >= tau } (reproducible)  (the honest ceiling)
"""
from __future__ import annotations
import numpy as np


def estimate_p_hat(b: np.ndarray, prior: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Beta-Bernoulli posterior mean p_hat from a (N, M, k) correctness tensor.

    Returns p_hat of shape (N, M). p_hat[i,m] = (a0 + sum_j b) / (a0 + b0 + k).
    """
    b = np.asarray(b, dtype=float)
    assert b.ndim == 3, "expected (N_queries, M_models, k_samples)"
    a0, b0 = prior
    k = b.shape[2]
    correct = b.sum(axis=2)
    return (a0 + correct) / (a0 + b0 + k)


def oracle_single(b_single: np.ndarray) -> np.ndarray:
    """Realized single-draw oracle per query: max over models of one 0/1 draw.
    b_single: (N, M) in {0,1}. Returns (N,) in {0,1}."""
    return (np.asarray(b_single) > 0).any(axis=1).astype(float)


def estimate_p_hat_raw(b: np.ndarray) -> np.ndarray:
    """Raw-frequency point estimate p_hat[i,m] = (1/k) sum_j b[i,m,j]. (N,M,k)->(N,M).

    This is the point estimate the theory uses (matches Thm. finitek and Algorithm 1);
    the Beta-Bernoulli posterior (estimate_p_hat / src.stats) is for CONFIDENCE
    INTERVALS ONLY, never the point estimate."""
    b = np.asarray(b, dtype=float)
    assert b.ndim == 3, "expected (N_queries, M_models, k_samples)"
    return b.sum(axis=2) / b.shape[2]


def oracle_expected_seed_aligned(b: np.ndarray) -> np.ndarray:
    """Dependence-aware O^exp_i = (1/k) sum_j max_m b[i,m,j]: the empirical max over
    the SAME draw index j across models, averaged over j (paper §V).

    Each summand 1{max_m b[i,m,j]=1} is i.i.d. Bernoulli(O^exp_i), so this estimator
    is EXACTLY unbiased -- no A2, no O(1/k) winner's-curse bias -- with a 2 exp(-2k t^2)
    tail. Requires seed-aligned K-tuples per query (A8). b: (N,M,k) -> (N,).
    This is the headline O^exp; the product form below is only an upper envelope."""
    b = np.asarray(b, dtype=float)
    assert b.ndim == 3, "expected (N_queries, M_models, k_samples)"
    return b.max(axis=1).mean(axis=1)


def oracle_expected_perp_envelope(p_hat: np.ndarray) -> np.ndarray:
    """Independent-coupling oracle O^{exp,perp}_i = 1 - prod_m (1 - p_hat[i,m]).

    This equals O^exp ONLY under cross-model independence (A2); in general it is the
    maximal-inflation / FKG UPPER ENVELOPE (Prop. dep), valid as a ceiling only when
    a per-stratum Cov >= 0 check passes. For the actual dependence-preserving O^exp use
    oracle_expected_seed_aligned(b). (Estimating this from an independent per-model
    resample estimates O^{exp,perp}, NOT O^exp.)"""
    p = np.clip(np.asarray(p_hat, dtype=float), 0.0, 1.0)
    return 1.0 - np.prod(1.0 - p, axis=1)


def oracle_expected(p_hat: np.ndarray) -> np.ndarray:
    """DEPRECATED alias of oracle_expected_perp_envelope (the independent-coupling
    upper envelope). Kept for backward compatibility; new code should pass the
    seed-aligned O^exp from oracle_expected_seed_aligned(b)."""
    return oracle_expected_perp_envelope(p_hat)


def oracle_reproducible(p_hat: np.ndarray) -> np.ndarray:
    """Reproducible (committable) oracle per query: max_m p_hat[i,m].
    This is the honest ceiling a per-query router can reproduce by committing to
    one model. Key invariant: O_exp >= O_repro >= p_hat[:, any model] (always),
    so the inflation O_exp - O_repro >= 0. This is the MAIN decomposition axis."""
    return np.asarray(p_hat, dtype=float).max(axis=1)


def oracle_threshold(p_hat: np.ndarray, tau: float) -> np.ndarray:
    """Auxiliary strict-reliability oracle: 1{ max_m p_hat[i,m] >= tau }.
    A binary {0,1} view used only for tau-sensitivity (NOT the main ceiling,
    since it is not comparable in magnitude to the continuous O_exp)."""
    return (np.asarray(p_hat, dtype=float).max(axis=1) >= tau).astype(float)


def best_single_correctness(p_hat: np.ndarray) -> np.ndarray:
    """Per-query correctness probability of the in-hindsight best single model
    (the model with the highest mean p_hat across queries). Returns (N,)."""
    p = np.asarray(p_hat, dtype=float)
    best = int(np.argmax(p.mean(axis=0)))
    return p[:, best]


def oracle_exp_frechet_bracket(p_hat: np.ndarray):
    """Assumption-free Frechet bracket for O^exp_i when A8 (draw alignment) is absent
    -- e.g. the typical released matrix with only per-cell marginals. Returns
    (lower, upper) = (max_m p_hat[i,m], min(sum_m p_hat[i,m], 1)). The product form
    O^{exp,perp} sits inside as the FKG upper envelope under positive association."""
    p = np.clip(np.asarray(p_hat, dtype=float), 0.0, 1.0)
    lower = p.max(axis=1)
    upper = np.minimum(p.sum(axis=1), 1.0)
    return lower, upper


def oracle_agg_from_labels(Y: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """One verifier-free aggregation oracle (Lemma aggfloor): the plurality /
    self-consistency vote over all produced draw LABELS, scored against gold.

    Y: (N, M, k) answer-label ids; gold: (N,) gold label id. Returns (N,) in {0,1}.

    NOTE 1 -- O^agg is defined as the SUP over draw-grounded verifier-free
    aggregators (each must OUTPUT one of the produced labels), so this single
    plurality aggregator is a LOWER BOUND on O^agg, not the sup; per query it can
    fall below O^repro (only the sup is guaranteed >= O^repro).
    NOTE 2 -- O^agg is genuinely NOT computable from the correctness tensor b alone;
    it needs the answer labels Y. Scoring (03_score) must therefore persist labels,
    not only correctness counts."""
    from collections import Counter
    Y = np.asarray(Y); gold = np.asarray(gold)
    assert Y.ndim == 3, "expected (N, M, k) label ids"
    N = Y.shape[0]
    out = np.empty(N, dtype=float)
    for i in range(N):
        # Drop unparseable draws (extract_answer -> None): a verifier-free aggregator
        # must OUTPUT a produced label, and None is "no answer". Counter avoids
        # np.unique's internal sort, which raises TypeError on mixed None/str object
        # arrays ("'<' not supported between instances of 'NoneType' and 'str'").
        labels = [v for v in Y[i].reshape(-1).tolist() if v is not None]
        if not labels:
            out[i] = 0.0           # no parseable label anywhere -> vote produces nothing -> wrong
            continue
        winner = Counter(labels).most_common(1)[0][0]
        out[i] = float(winner == gold[i])
    return out


def num_correct_models(b_or_p: np.ndarray, tau: float | None = None) -> np.ndarray:
    """#models 'correct' per query. If tau is None, treat input as (N,M) 0/1 draws;
    else treat as p_hat and count p>=tau. Used to define the rare-correct stratum."""
    x = np.asarray(b_or_p, dtype=float)
    if tau is None:
        return (x > 0).sum(axis=1).astype(int)
    return (x >= tau).sum(axis=1).astype(int)


if __name__ == "__main__":  # quick sanity demo
    rng = np.random.default_rng(0)
    N, M, k = 2000, 12, 10
    p_true = rng.beta(0.3, 2.0, size=(N, M))        # mostly-hard pool
    b = (rng.random((N, M, k)) < p_true[:, :, None]).astype(int)
    b_single = (rng.random((N, M)) < p_true).astype(int)
    p_hat = estimate_p_hat(b)
    print("mean single  :", oracle_single(b_single).mean())
    print("mean expected:", oracle_expected(p_hat).mean())
    print("mean thr@0.5 :", oracle_threshold(p_hat, 0.5).mean())
    print("mean thr@0.9 :", oracle_threshold(p_hat, 0.9).mean())
    # expected should exceed threshold -> the noise inflation
