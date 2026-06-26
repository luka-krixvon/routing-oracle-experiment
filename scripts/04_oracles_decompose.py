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


# ---- pool-composition robustness: lineage map, sub-pool decomposition, correlation ----
# Group by PRETRAINING lineage, not vendor brand: a Qwen-distilled model counts as Qwen.
LINEAGE_RULES = [
    ("qwen", "Qwen"),               # incl. DeepSeek-R1-Distill-Qwen (arch=qwen2, Qwen2.5 base)
    ("ministral", "Mistral"), ("mixtral", "Mistral"), ("mistral", "Mistral"),
    ("phi", "Phi"), ("gemma", "Gemma"), ("llama", "Llama"), ("olmo", "OLMo"),
    ("yi-", "Yi"), ("/yi", "Yi"), ("granite", "Granite"), ("internlm", "InternLM"),
    ("glm", "GLM"), ("falcon", "Falcon"), ("exaone", "EXAONE"),
    ("aya", "Cohere"), ("command", "Cohere"), ("cohere", "Cohere"),
]


def classify_lineage(name):
    """Map an HF repo id to its pretraining lineage; fall back to the org prefix."""
    s = str(name).lower()
    for sub, lin in LINEAGE_RULES:
        if sub in s:
            return lin
    return s.split("/")[0] if "/" in s else s


def _pool_summary(b_sub, B=2000, seed=0, n_strata=1):
    """Decomposition for a sub-pool, router baseline = best single model WITHIN the
    sub-pool (apples-to-apples across pool definitions)."""
    N, M, k = b_sub.shape
    phat = oracles.estimate_p_hat_raw(b_sub)
    O_exp = oracles.oracle_expected_seed_aligned(b_sub)
    O_repro = oracles.oracle_reproducible(phat)
    q = phat[:, int(phat.mean(axis=0).argmax())]                 # best-single within sub-pool
    pt, lo, hi = _bootstrap_share_ci(O_exp, O_repro, q, B=B, seed=seed)
    cons = decompose.decompose_gap_conservative(b_sub, q, n_strata=n_strata); cons.pop("_per_query", None)
    return {"M": int(M), "router_basis": "best-single-within-pool",
            "oracles_mean": {"exp_seed_aligned": float(O_exp.mean()),
                             "reproducible": float(O_repro.mean()), "router": float(q.mean())},
            "noise_mean": float((O_exp - O_repro).mean()), "gap_mean": float((O_exp - q).mean()),
            "noise_share": {"point": pt, "lo": lo, "hi": hi, "lower_bound_above_0": bool(lo > 0)},
            "conservative_floor_mean": cons.get("Delta_lower_mean")}


def pool_definitions(b, models, B=2000, seed=0, n_strata=1):
    """noise_share under THREE pool definitions so the headline is not a composition
    artifact: FULL, ONE-PER-LINEAGE (strongest member per pretraining lineage), and the
    QWEN size-sweep (same-lineage-only control). Bias direction: intra-lineage redundancy
    depresses the recoverable term and inflates noise_share, so one-per-lineage is the
    CONSERVATIVE headline and full is an upper bound."""
    import re
    names = [str(m) for m in models]
    lin = [classify_lineage(n) for n in names]
    phat_mean = oracles.estimate_p_hat_raw(b).mean(axis=0)       # per-model mean accuracy
    rep = {}
    for j, L in enumerate(lin):                                  # strongest member per lineage
        if L not in rep or phat_mean[j] > phat_mean[rep[L]]:
            rep[L] = j

    def _size(n):
        m = re.search(r"(\d+(?:\.\d+)?)\s*b", n.lower())
        return float(m.group(1)) if m else 0.0
    qwen_idx = sorted([j for j, n in enumerate(names)
                       if "qwen2.5" in n.lower() and "distill" not in n.lower()],
                      key=lambda j: _size(names[j]))
    defs = {"full": list(range(len(names))),
            "one_per_lineage": sorted(rep.values()),
            "qwen_size_sweep": qwen_idx}
    out = {}
    for key, idx in defs.items():
        entry = {"models": [names[j] for j in idx], "lineages": sorted(set(lin[j] for j in idx))}
        if len(idx) >= 2:
            entry.update(_pool_summary(b[:, idx, :], B=B, seed=seed, n_strata=n_strata))
        else:
            entry["note"] = "skipped: <2 models available in this definition"
        out[key] = entry
    out["_note"] = ("router baseline = best-single-within-pool for comparability; lineage grouped "
                    "by pretraining (a Qwen-distilled model counts as Qwen); one_per_lineage keeps "
                    "the strongest member of each lineage and is the conservative headline.")
    return out


def family_correlation(b, models, outdir):
    """Lineage-clustered pairwise error-correlation diagnostic on the correctness tensor +
    an effective-pool-size (participation ratio of the correlation eigenvalues). Writes
    family_correlation.csv. Confirms whether same-lineage models are more correlated than
    cross-lineage ones (which would inflate noise_share)."""
    names = [str(m) for m in models]
    lin = [classify_lineage(n) for n in names]
    order = sorted(range(len(names)), key=lambda j: (lin[j], names[j]))   # cluster by lineage
    nm, lo = [names[j] for j in order], [lin[j] for j in order]
    phat = oracles.estimate_p_hat_raw(b)[:, order]                        # N x M (clustered)
    C = np.nan_to_num(np.corrcoef(phat.T), nan=0.0)
    if C.ndim == 0:                                                       # M==1 guard
        C = np.array([[1.0]])
    w = np.clip(np.linalg.eigvalsh(C), 0, None)
    eff = float((w.sum() ** 2) / (w ** 2).sum()) if (w ** 2).sum() else float("nan")
    M = len(nm); win, cro = [], []
    for i in range(M):
        for j in range(i + 1, M):
            (win if lo[i] == lo[j] else cro).append(C[i, j])
    os.makedirs(outdir, exist_ok=True)
    csvp = os.path.join(outdir, "family_correlation.csv")
    with open(csvp, "w") as f:
        f.write("model,lineage," + ",".join(nm) + "\n")
        for i in range(M):
            f.write(f"{nm[i]},{lo[i]}," + ",".join(f"{C[i, j]:.3f}" for j in range(M)) + "\n")
    return {"order": nm, "lineage": lo,
            "matrix": [[round(float(C[i, j]), 3) for j in range(M)] for i in range(M)],
            "within_lineage_mean": float(np.mean(win)) if win else None,
            "cross_lineage_mean": float(np.mean(cro)) if cro else None,
            "effective_pool_size": eff, "n_models": M,
            "note": "Pearson corr of per-query p_hat across queries, clustered by lineage; "
                    "effective_pool_size = participation ratio (1=identical .. M=independent). "
                    "CSV: family_correlation.csv"}


def run(b, b_single, q_router, outdir, B=2000, seed=0, n_strata=1, models=None, Y=None, gold=None):
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

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "decomposition.json")
    # PROTOCOL: a failed pre-gate suppresses the magnitude study (report nothing but the gates).
    if not gates_pass:
        out = {"mode": "simulated" if outdir.endswith("mvp") else "data",
               "magnitude_suppressed": True,
               "gates": {"known_p": gA, "independence": gB, "all_pass": False},
               "oracles_mean": {"single": float(O_single.mean()), "exp_seed_aligned": float(O_exp.mean()),
                                "exp_perp_envelope": float(O_perp.mean()),
                                "reproducible": float(O_repro.mean()), "router": float(q.mean())},
               "note": "A pre-gate failed (known-p TV and/or per-draw independence vs A1/provider caching); "
                       "per protocol the magnitude study reports NO decomposition/noise_share/best-of-K."}
        json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)
        return out, path

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
    # verifier-free aggregation split (Lemma aggfloor / Cor. scopefloor): needs answer labels Y + gold
    if Y is not None and gold is not None:
        O_agg = oracles.oracle_agg_from_labels(Y, gold)
        d_know = np.clip(O_agg - O_repro, 0, None)            # recoverable without a verifier (vote up to O^agg)
        d_guess = np.clip(O_exp - O_agg, 0, None)             # union-minus-deliverable; needs a verifier
        out["verifier_free_split"] = {
            "O_agg_mean": float(O_agg.mean()),
            "Delta_know_mean": float(d_know.mean()), "Delta_guess_mean": float(d_guess.mean()),
            "note": "O_agg = single plurality/self-consistency vote (a LOWER bound on the sup O^agg)."}
    # pool-composition robustness (only with real model names; needs >=2 models)
    if models is not None and M >= 2:
        out["pool_definitions"] = pool_definitions(b, models, B=B, seed=seed, n_strata=n_strata)
        out["family_correlation"] = family_correlation(b, models, outdir)
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
        d = np.load(a.npz, allow_pickle=True)
        models = None
        if "meta" in d.files:                       # combine.py stores {"models": [...]} order-matched to b
            try:
                raw = d["meta"]; raw = raw.item() if hasattr(raw, "item") else str(raw)
                models = json.loads(raw).get("models")
            except Exception:
                models = None
        Y = d["Y"] if "Y" in d.files else None
        gold = d["gold"] if "gold" in d.files else None
        outdir = a.outdir or os.path.join(root, "results", "data")
        out, path = run(d["b"], d["b_single"], d["q_router"], outdir, seed=a.seed,
                        models=models, Y=Y, gold=gold)
        print(json.dumps(out, indent=2, ensure_ascii=False)); print("wrote", path)
    else:
        print("use --mvp (simulated smoke test) or --npz PATH (real data from 03_score.py)")
