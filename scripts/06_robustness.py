#!/usr/bin/env python3
"""Robustness suite for the gap decomposition (paper §Results, Table tab:robust).

Reproduces, from the committed re-generation tensor, the four robustness controls
the paper reports and one learned-router baseline:

  (1) k-sweep   : noise share using only the first k' of the k seed-aligned draws
                  (k'=1 is the non-identifiable degenerate case, Thm. finitek(d)).
  (2) K-sweep   : the ORACLE noise share S(K)=Delta_i/O^exp_i (Cor. noiseshare) vs
                  the GAP share G_noise/G, over random model-subsets of size K.
                  S(K) rises monotonically in K (theory); the gap share falls
                  because larger heterogeneous pools raise O^repro/router.
  (3) jackknife : leave-one-pretraining-lineage-out noise share (stability control iii).
  (4) tau       : reliable_frac = P(max_m p_hat >= tau) at tau in {0.5,0.9}.
  (5) learned router : a pure-numpy TF-IDF + cosine-kNN per-query router trained
                  (5-fold out-of-fold) on the k=1 slice + question text; scored on
                  its chosen model's k-draw p_hat. Shows the noise SHARE is
                  baseline-dependent while the FLOOR G_noise is router-invariant.

Inputs (repo-relative, produced by the pipeline 01..04):
  data/processed/correctness_kxN.npz : b (N,M,k), b_single (N,M)
  data/subset.json                   : [{id, prompt, gold, task}, ...]
  data/per_model/*.npz               : per-model 'ids' (query order) + 'model'

Usage:
  python scripts/06_robustness.py --data data/processed/correctness_kxN.npz \
      --subset data/subset.json --out results/data/robustness.json
No GPU, no third-party ML dependency (numpy only).
"""
from __future__ import annotations
import argparse, glob, json, os, re
from collections import Counter
import numpy as np

# pretraining lineage of each pool model (by pretraining base, not vendor brand):
# the Qwen-distilled DeepSeek model counts as Qwen. Keyed by a substring of the HF id.
LINEAGE_RULES = [
    ("Mistral-7B", "Mistral"), ("DeepSeek-R1-Distill-Qwen", "Qwen"),
    ("Qwen2.5", "Qwen"), ("phi-4", "Phi"), ("OLMo-2", "OLMo"),
    ("Yi-1.5", "Yi"), ("granite", "Granite"), ("gemma-2", "Gemma"),
    ("Llama-3.1", "Llama"),
]


def lineage_of(model_id: str) -> str:
    for key, lin in LINEAGE_RULES:
        if key.lower() in model_id.lower():
            return lin
    return "Other"


def _oracles(b: np.ndarray):
    """Return (O_exp seed-aligned, O_repro, p_hat) for a (N,M,k) tensor."""
    k = b.shape[2]
    p_hat = b.sum(2) / k
    O_exp = b.max(1).mean(1)          # (1/k) sum_j max_m b[:,:,j]  (dependence-aware)
    O_repro = p_hat.max(1)            # max_m p_hat
    return O_exp, O_repro, p_hat


def _shares(b: np.ndarray, cols=None, q_router=None):
    bb = b if cols is None else b[:, cols, :]
    O_exp, O_repro, p_hat = _oracles(bb)
    if q_router is None:              # default router = in-hindsight best single model
        q_router = p_hat[:, int(np.argmax(p_hat.mean(0)))]
    G = float((O_exp - q_router).mean())
    noise = float((O_exp - O_repro).mean())
    rec = float((O_repro - q_router).mean())
    Oexp = float(O_exp.mean())
    return {
        "router_acc": float(np.mean(q_router)), "O_exp": Oexp, "O_repro": float(O_repro.mean()),
        "G": G, "G_rec": rec, "G_noise": noise,
        "gap_share": (noise / G if abs(G) > 1e-12 else None),
        "oracle_share_S": (noise / Oexp if Oexp > 1e-12 else None),
    }


def _tfidf(prompts):
    def tok(s):
        return re.findall(r"[a-zA-Z]+|\\[a-zA-Z]+|[0-9]+", s.lower())
    docs = [tok(p) for p in prompts]
    df = Counter()
    for d in docs:
        df.update(set(d))
    N = len(docs)
    vocab = {w: j for j, w in enumerate(w for w, c in df.items() if c >= 2)}
    idf = np.zeros(len(vocab))
    for w, j in vocab.items():
        idf[j] = np.log((1 + N) / (1 + df[w])) + 1
    X = np.zeros((N, len(vocab)))
    for i, d in enumerate(docs):
        for w, cnt in Counter(w for w in d if w in vocab).items():
            X[i, vocab[w]] = cnt
    X = X * idf[None, :]
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def learned_knn_router(b_single, X, knn=25, folds=5, seed=0):
    """5-fold out-of-fold TF-IDF cosine-kNN router: route each query to the model
    with highest mean k=1 correctness among its nearest training neighbours."""
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.array_split(rng.permutation(N), folds)
    routed = np.zeros(N, dtype=int)
    for f in range(folds):
        te = idx[f]
        tr = np.concatenate([idx[g] for g in range(folds) if g != f])
        sim = X[te] @ X[tr].T
        nn = np.argsort(-sim, axis=1)[:, :knn]
        for a, i in enumerate(te):
            routed[i] = int(np.argmax(b_single[tr[nn[a]]].mean(0)))
    return routed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/correctness_kxN.npz")
    ap.add_argument("--subset", default="data/subset.json")
    ap.add_argument("--per_model_glob", default="data/per_model/*.npz")
    ap.add_argument("--out", default="results/data/robustness.json")
    ap.add_argument("--knn", type=int, default=25)
    ap.add_argument("--k_grid", default="1,2,5,10,20,30")
    ap.add_argument("--K_grid", default="2,4,8")
    ap.add_argument("--K_subsets", type=int, default=60)
    ap.add_argument("--tau_grid", default="0.5,0.9")
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    b = np.asarray(z["b"], dtype=float)
    b_single = np.asarray(z["b_single"], dtype=float)
    N, M, k = b.shape
    models = list(json.loads(str(z["meta"]))["models"]) if "meta" in z.files else \
        [f"m{i}" for i in range(M)]
    lineages = [lineage_of(m) for m in models]

    out = {"N": N, "M": M, "k": k, "models": models, "lineages": lineages}
    out["baseline"] = _shares(b)

    # (1) k-sweep
    ks = [int(x) for x in args.k_grid.split(",")]
    out["k_sweep"] = [{"k": kk, **_shares(b[:, :, :kk])} for kk in ks]

    # (2) K-sweep (random subsets) — oracle share S(K) and gap share
    rng = np.random.default_rng(42)
    Ks = [int(x) for x in args.K_grid.split(",")] + [M]
    ksweep = []
    for K in Ks:
        combos = 1 if K >= M else args.K_subsets
        S, GS = [], []
        for _ in range(combos):
            cols = list(range(M)) if K >= M else sorted(rng.choice(M, K, replace=False).tolist())
            s = _shares(b, cols)
            if s["oracle_share_S"] is not None:
                S.append(s["oracle_share_S"])
            if s["gap_share"] is not None:
                GS.append(s["gap_share"])
        ksweep.append({"K": K, "oracle_share_S_mean": float(np.mean(S)),
                       "gap_share_mean": float(np.mean(GS))})
    out["K_sweep"] = ksweep

    # (3) leave-one-lineage-out jackknife
    jk = []
    for L in sorted(set(lineages)):
        cols = [i for i, l in enumerate(lineages) if l != L]
        if len(cols) >= 2:
            jk.append({"dropped": L, "n_left": len(cols), **_shares(b, cols)})
    shares = [d["gap_share"] for d in jk if d["gap_share"] is not None]
    out["jackknife"] = {"folds": jk, "gap_share_min": min(shares), "gap_share_max": max(shares),
                        "full": out["baseline"]["gap_share"]}

    # (4) tau reliability
    _, O_repro, _ = _oracles(b)
    out["tau"] = [{"tau": float(t), "reliable_frac": float((O_repro >= float(t)).mean())}
                  for t in args.tau_grid.split(",")]

    # (5) learned kNN router
    try:
        ids = None
        for p in sorted(glob.glob(args.per_model_glob)):
            zz = np.load(p, allow_pickle=True)
            if "ids" in zz.files:
                ids = [str(x) for x in zz["ids"]]
                break
        sub = {it["id"]: it["prompt"] for it in json.load(open(args.subset))}
        prompts = [sub[i] for i in ids] if ids else [it["prompt"] for it in json.load(open(args.subset))]
        X = _tfidf(prompts)
        routed = learned_knn_router(b_single, X, knn=args.knn)
        _, _, p_hat = _oracles(b)
        q = p_hat[np.arange(N), routed]
        out["learned_router"] = {"type": "tfidf_knn_5fold_oof", "knn": args.knn, **_shares(b, q_router=q)}
    except Exception as e:  # keep the suite running if text/ids are unavailable
        out["learned_router"] = {"error": str(e)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    bs = out["baseline"]
    print(f"  full pool: gap_share={bs['gap_share']:.3f}  O_exp={bs['O_exp']:.3f}  O_repro={bs['O_repro']:.3f}")
    print(f"  k-sweep gap_share: " + ", ".join(f"k{d['k']}={d['gap_share']:.3f}" if d['gap_share'] else f"k{d['k']}=NA" for d in out["k_sweep"]))
    print(f"  K-sweep oracle S(K): " + ", ".join(f"K{d['K']}={d['oracle_share_S_mean']:.3f}" for d in out["K_sweep"]))
    print(f"  jackknife gap_share range: [{out['jackknife']['gap_share_min']:.3f}, {out['jackknife']['gap_share_max']:.3f}]")
    lr = out["learned_router"]
    if "error" not in lr:
        print(f"  learned kNN router: acc={lr['router_acc']:.3f} gap_share={lr['gap_share']:.3f} (floor G_noise={lr['G_noise']:.3f})")


if __name__ == "__main__":
    main()
