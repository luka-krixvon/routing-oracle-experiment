"""Synthetic correctness generator for pipeline dry-runs / MVP smoke tests (NO GPU).

Real generation (src/generate.py, vLLM/API) produces text that src/score.py turns
into a (N, M, k) correctness tensor b[i,m,j]. On a box without GPUs we cannot run
the LLMs, but we CAN exercise the entire analysis pipeline (estimators, gates,
decomposition, best-of-K test, CIs, figures) by drawing the correctness tensor
directly from a latent per-cell probability p_true[i,m]. This validates the code
end-to-end and previews the output shape; it does NOT replace the real run.

The 'thin' scenario puts most mass on hard queries (few models can solve them),
which is exactly the regime where the single-draw noise term G_noise is largest.
"""
from __future__ import annotations
import numpy as np


def simulate_correctness(N: int = 200, M: int = 5, k: int = 10,
                         scenario: str = "thin", seed: int = 0) -> dict:
    """Return a dict with a seed-aligned correctness tensor and a realistic router.

    b        : (N, M, k) in {0,1}, draw j of model m on query i (i.i.d. Bernoulli(p_true)).
    b_single : (N, M) the one recorded label per cell (= draw index 0), as released matrices store.
    p_true   : (N, M) latent reproducible success probabilities (unknown in reality).
    q_router : (N,) realized correctness prob of an imperfect feature-router (picks argmax of a
               NOISY score, then is scored on the TRUE prob of whichever model it committed to, A4).
    """
    rng = np.random.default_rng(seed)
    if scenario == "thin":        # mostly-hard pool (many queries solvable by few models)
        p_true = rng.beta(0.30, 2.2, size=(N, M))
    elif scenario == "dense":     # easy pool (most models can solve)
        p_true = rng.beta(1.6, 1.6, size=(N, M))
    else:                          # mixed
        p_true = rng.beta(0.5, 2.0, size=(N, M))
    b = (rng.random((N, M, k)) < p_true[:, :, None]).astype(int)   # A1 (i.i.d.) + A2 (independent)
    b_single = b[:, :, 0].copy()                                   # the single recorded draw
    noisy_score = p_true + rng.normal(0.0, 0.15, size=(N, M))      # router sees a noisy estimate
    chosen = noisy_score.argmax(axis=1)                            # commits to one model (A4)
    q_router = p_true[np.arange(N), chosen]                        # scored on chosen model's true p
    return {"b": b, "b_single": b_single, "p_true": p_true, "q_router": q_router,
            "meta": {"N": N, "M": M, "k": k, "scenario": scenario, "seed": seed,
                     "note": "SIMULATED correctness (no LLM); pipeline dry-run only"}}
