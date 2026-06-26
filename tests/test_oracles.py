"""Sanity tests for the pure-math core (run: python -m pytest -q, or python tests/test_oracles.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import oracles, decompose, stats


def _toy(N=1000, M=8, k=10, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.4, 2.0, size=(N, M))
    b = (rng.random((N, M, k)) < p[:, :, None]).astype(int)
    return p, b


def test_expected_ge_reproducible():
    # KEY invariant: 1 - prod(1-p) >= max_m p, elementwise.
    p, b = _toy(); p_hat = oracles.estimate_p_hat(b)
    assert np.all(oracles.oracle_expected(p_hat) + 1e-9 >= oracles.oracle_reproducible(p_hat))


def test_reproducible_ge_any_single_model():
    p, b = _toy(); p_hat = oracles.estimate_p_hat(b)
    O_repro = oracles.oracle_reproducible(p_hat)
    assert np.all(O_repro + 1e-9 >= oracles.best_single_correctness(p_hat))


def test_expected_in_unit_interval():
    p, b = _toy(); O = oracles.oracle_expected(oracles.estimate_p_hat(b))
    assert O.min() >= 0 and O.max() <= 1


def test_decomposition_sums_and_noise_nonneg():
    p, b = _toy(); p_hat = oracles.estimate_p_hat(b)
    q = p_hat.max(axis=1) * 0.5
    d = decompose.decompose_gap(p_hat, q)
    assert abs(d["recoverable"] + d["noise"] - d["G_single"]) < 1e-9
    assert d["noise"] >= -1e-9        # noise term is non-negative by construction


# ---------------------------------------------------------------------------
# Corrected-estimator tests (seed-aligned O^exp, raw p_hat, one-sided radius, gates)
# ---------------------------------------------------------------------------

def test_raw_phat_is_counts_over_k():
    p, b = _toy()
    assert np.allclose(oracles.estimate_p_hat_raw(b), b.sum(axis=2) / b.shape[2])
    # raw frequency differs from the Beta(1,1)-shrunk posterior mean (which is CI-only)
    assert not np.allclose(oracles.estimate_p_hat_raw(b), oracles.estimate_p_hat(b))


def test_seed_aligned_ordering():
    # Empirical, assumption-free: max_m mean_j b  <=  mean_j max_m b.
    p, b = _toy()
    O_repro = oracles.oracle_reproducible(oracles.estimate_p_hat_raw(b))
    O_exp = oracles.oracle_expected_seed_aligned(b)
    assert np.all(O_exp + 1e-12 >= O_repro)
    assert O_exp.min() >= 0 and O_exp.max() <= 1


def test_seed_aligned_matches_perp_under_independence():
    # Draws are independent across models in _toy, so the dependence-aware seed-aligned
    # O^exp and the independent-coupling envelope agree in aggregate (up to MC noise).
    p, b = _toy(N=4000, M=8, k=20)
    sa = oracles.oracle_expected_seed_aligned(b).mean()
    perp = oracles.oracle_expected_perp_envelope(oracles.estimate_p_hat_raw(b)).mean()
    assert abs(sa - perp) < 0.02, (sa, perp)


def test_frechet_bracket_contains_envelope():
    p, b = _toy(); phat = oracles.estimate_p_hat_raw(b)
    lo, hi = oracles.oracle_exp_frechet_bracket(phat)
    perp = oracles.oracle_expected_perp_envelope(phat)
    assert np.all(lo - 1e-12 <= perp) and np.all(perp <= hi + 1e-12)
    assert np.allclose(lo, phat.max(axis=1))
    assert np.all(hi <= 1.0 + 1e-12)


def test_conservative_is_one_sided_correct_direction():
    p, b = _toy()
    q = oracles.estimate_p_hat_raw(b).max(axis=1) * 0.5
    d = decompose.decompose_gap_conservative(b, q)
    # radius is ADDED to O_repro (upper), so the corrected repro exceeds the point repro
    assert d["O_repro_upper_mean"] > d["O_repro_mean"]
    assert d["O_exp_lower_mean"] < d["O_exp_mean"]
    # conservative lower bound never exceeds the uncorrected point estimate, and stays >=0
    assert d["Delta_lower_mean"] <= d["noise_point"] + 1e-12
    assert d["Delta_lower_mean"] >= -1e-12


def test_decompose_accepts_seed_aligned_O_exp():
    p, b = _toy(); phat = oracles.estimate_p_hat_raw(b)
    q = phat.max(axis=1) * 0.5
    O_exp = oracles.oracle_expected_seed_aligned(b)
    d = decompose.decompose_gap(phat, q, O_exp=O_exp)
    assert abs(d["recoverable"] + d["noise"] - d["G_single"]) < 1e-9
    assert abs(d["O_exp_mean"] - O_exp.mean()) < 1e-12


def test_gates_pass_on_iid_data():
    # Both gates are designed to PASS on clean i.i.d. Bernoulli draws.
    p, b = _toy(N=2000, M=8, k=20)
    gA = stats.gate_known_p(b)
    gB = stats.gate_independence(b)
    assert gA["pass"], gA
    assert gB["pass"], gB


def test_gate_independence_flags_caching():
    # Inject BLOCK caching on half the models: each cell's k draws are 2 repeated
    # blocks (e.g. [0]*10 + [1]*10). Such cells stay interior (phat=0.5) but have
    # far too few runs, so the Wald-Wolfowitz runs test fires -> gate fails.
    rng = np.random.default_rng(1)
    N, M, k = 1500, 8, 20
    p = rng.beta(0.6, 0.6, size=(N, M))                # mass off {0,1} -> cells informative
    b = (rng.random((N, M, k)) < p[:, :, None]).astype(int)   # models 0-3 genuine i.i.d.
    nb, bs = 2, k // 2
    for mdl in range(M // 2, M):                        # models 4-7 cached
        buckets = (rng.random((N, nb)) < p[:, mdl][:, None]).astype(int)  # (N, nb)
        b[:, mdl, :] = np.repeat(buckets, bs, axis=1)  # 2 blocks -> <=2 runs
    gB = stats.gate_independence(b)
    assert not gB["pass"], gB


def test_agg_from_labels_plurality():
    # gold=2; model/draw labels mostly 2 on q0, mostly 9 (wrong) on q1.
    Y = np.array([[[2, 2, 1], [2, 3, 2]], [[9, 9, 9], [9, 1, 9]]])
    gold = np.array([2, 2])
    out = oracles.oracle_agg_from_labels(Y, gold)
    assert out[0] == 1.0 and out[1] == 0.0


def test_score_exact_match():
    """Guard the live scorers (the data-producing layer): GSM8K numeric normalization
    and MMLU final-letter extraction -- the bugs that silently deflate p_hat."""
    from src import score
    assert score.exact_match("#### 72.0", "72", "gsm8k") == 1
    assert score.exact_match("The answer is 18.", "18", "gsm8k") == 1
    assert score.exact_match("#### 1,200", "1200", "gsm8k") == 1
    assert score.exact_match("we checked 2 times, #### 18", "18", "gsm8k") == 1
    assert score.exact_match("#### 19", "18", "gsm8k") == 0
    assert score.exact_match("Option A is wrong, so the answer is C.", "C", "mmlu_pro") == 1
    assert score.exact_match("Choice A... but B", "B", "mmlu_pro") == 1
    assert score.exact_match("answer: D", "C", "mmlu_pro") == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok:", fn.__name__)
    print(f"ok: all {len(fns)} sanity tests passed")
