"""Stage 04 — correctness tensor -> corrected oracles -> gap decomposition -> gates
-> best-of-K falsifiable test -> CIs (MAIN result).

Uses the FINAL corrected estimators (src/oracles, src/decompose, src/stats):
  - O^repro from raw-frequency p_hat (Beta only for CIs)
  - O^exp via the seed-aligned estimator over the draw tensor b (not the product form)
  - one-sided, winner's-curse-corrected conservative lower bound on the noise term
  - two pre-gates (known-p simulation; per-draw independence) that ABORT the magnitude study
  - the single falsifiable check: matched-budget best-of-K on the committed best model vs O^{exp,perp}

Run modes:
  --mvp           : SIMULATED end-to-end smoke test (no GPU) — thin-support scenario
  --npz PATH      : real run on a correctness_kxN.npz produced by 03_score.py
Outputs results/decomposition.json (+ results/mvp_decomposition.png for --mvp).
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import oracles, decompose, stats, simulate


def _bootstrap_share_ci(O_exp, O_repro, q, B=2000, alpha=0.05, seed=0):
    """Percentile CI for the noise share = mean(O_exp-O_repro)/mean(O_exp-q) by
    resampling queries (the ratio-of-means is bootstrapped jointly)."""
    rng = np.random.default_rng(seed)
    n = len(O_exp); num = O_exp - O_repro; den = O_exp - q
    pt = float(num.mean() / den.mean()) if den.mean() else float("nan")
    est = np.empty(B)
    for bI in range(B):
        idx = rng.integers(0, n, n)
        d = den[idx].mean()
        est[bI] = num[idx].mean() / d if d else np.nan
    lo, hi = np.nanpercentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return pt, float(lo), float(hi)


def run(b, b_single, q_router, outdir, B=2000, seed=0, n_strata=1):
    b = np.asarray(b, float); q = np.asarray(q_router, float)
    N, M, k = b.shape

    # --- pre-gates: must pass before reporting any magnitude ---
    gA = stats.gate_known_p(b, seed=seed)
    gB = stats.gate_independence(b)
    gates_pass = bool(gA["pass"] and gB["pass"])

    # --- corrected estimators ---
    phat = oracles.estimate_p_hat_raw(b)                      # raw frequency (point estimate)
    O_exp = oracles.oracle_expected_seed_aligned(b)           # seed-aligned, dependence-aware
    O_repro = oracles.oracle_reproducible(phat)
    O_perp = oracles.oracle_expected_perp_envelope(phat)      # independent-coupling upper envelope
    O_single = oracles.oracle_single(b_single)

    main = decompose.decompose_gap(phat, q, O_exp=O_exp); main.pop("_per_query", None)
    cons = decompose.decompose_gap_conservative(b, q, n_strata=n_strata); cons.pop("_per_query", None)
    share_pt, share_lo, share_hi = _bootstrap_share_ci(O_exp, O_repro, q, B=B, seed=seed)

    # --- falsifiable check: matched-budget best-of-K on the PER-QUERY committed best model
    #     m*(i)=argmax_m p_hat[i,m] (thm:recoverability is per-query), vs O^{exp,perp} ---
    mstar_i = phat.argmax(axis=1)                            # per-query committed best model
    bok = b[np.arange(N), mstar_i, :].max(axis=1).mean()     # empirical best-of-k of m*(i)
    perp_mean = float(O_perp.mean())
    slack = 0.01
    bestofK = {"committed_model": "per-query argmax_m p_hat",
               "best_of_k_committed": float(bok),
               "O_exp_perp_mean": perp_mean, "slack": slack,
               "passes_recoverability": bool(bok + slack >= perp_mean)}

    # --- thin-support stratum (#models with a correct single draw <= 3) ---
    n_correct = (b_single > 0).sum(axis=1)
    thin = n_correct <= 3
    strat = {}
    for name, mask in [("thin(<=3)", thin), ("dense(>3)", ~thin)]:
        if mask.any():
            ds = float((O_exp[mask] - O_repro[mask]).mean())
            de = float((O_exp[mask] - q[mask]).mean())
            strat[name] = {"frac": float(mask.mean()), "noise": ds,
                           "noise_share": ds / de if de else float("nan")}

    out = {
        "mode": "simulated" if outdir.endswith("mvp") else "data",
        "gates": {"known_p": gA, "independence": gB, "all_pass": gates_pass},
        "oracles_mean": {"single": float(O_single.mean()), "exp_seed_aligned": float(O_exp.mean()),
                          "exp_perp_envelope": perp_mean, "reproducible": float(O_repro.mean()),
                          "router": float(q.mean())},
        "decomposition_point": main,
        "noise_share_ci": {"point": share_pt, "lo": share_lo, "hi": share_hi,
                            "lower_bound_above_0": bool(share_lo > 0)},
        "conservative_one_sided": cons,          # Delta_lower_mean is the winner's-curse-corrected floor
        "falsifiable_best_of_K": bestofK,
        "stratified": strat,
    }
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "decomposition.json")
    json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)
    return out, path


def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    INK, GRAY, ACC = "#1A1A1A", "#8A8F94", "#3B6EA5"
    om = out["oracles_mean"]; q = om["router"]; rep = om["reproducible"]; exp = om["exp_seed_aligned"]
    cons_lo = out["conservative_one_sided"]["Delta_lower_mean"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(0, q, color=GRAY, label="router q (achieved)")
    ax.bar(0, rep - q, bottom=q, color="#C9CDD2", label="$G_{rec}$ recoverable (selection)")
    ax.bar(0, exp - rep, bottom=rep, color=ACC, label="$G_{noise}$ single-draw noise")
    ax.plot([-0.4, 0.4], [rep, rep], color=INK, lw=1, ls="--")
    ax.text(0.45, rep, "$O^{repro}$", va="center", fontsize=8)
    ax.text(0.45, exp, "$O^{exp}$", va="center", fontsize=8, color=ACC)
    ax.text(0.45, q, "router", va="center", fontsize=8, color=GRAY)
    sh = out["noise_share_ci"]
    ax.set_title(f"Simulated MVP — noise share {sh['point']:.0%} "
                 f"(95% CI [{sh['lo']:.0%}, {sh['hi']:.0%}])", fontsize=9, color=INK)
    ax.set_xlim(-0.7, 1.4); ax.set_ylim(0, max(exp * 1.15, 0.05)); ax.set_xticks([])
    ax.set_ylabel("per-query mean"); ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); figp = os.path.join(os.path.dirname(path), "mvp_decomposition.png")
    fig.savefig(figp, dpi=200, bbox_inches="tight"); plt.close(fig)
    return figp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvp", action="store_true", help="simulated end-to-end smoke test (no GPU)")
    ap.add_argument("--npz", help="real correctness_kxN.npz from 03_score.py")
    ap.add_argument("--N", type=int, default=200); ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", default="thin"); ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(__file__))
    if a.mvp:
        sim = simulate.simulate_correctness(a.N, a.M, a.k, a.scenario, a.seed)
        outdir = a.outdir or os.path.join(root, "results", "mvp")
        out, path = run(sim["b"], sim["b_single"], sim["q_router"], outdir, seed=a.seed)
        figp = make_figure(out, path)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        print("\nwrote", path, "\nwrote", figp)
    elif a.npz:
        d = np.load(a.npz)
        outdir = a.outdir or os.path.join(root, "results", "data")
        out, path = run(d["b"], d["b_single"], d["q_router"], outdir, seed=a.seed)
        print(json.dumps(out, indent=2, ensure_ascii=False)); print("wrote", path)
    else:
        print("use --mvp (simulated smoke test) or --npz PATH (real data from 03_score.py)")
