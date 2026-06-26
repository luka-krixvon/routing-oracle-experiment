"""Combine per-model correctness columns (data/per_model/m*.npz) into the (N,M,k)
tensor for 04_oracles_decompose.py. Run after all models finish; the per-model npz
are tiny (0/1 arrays), so this needs no model weights and no GPU.

  python scripts/combine.py --subset data/subset.json
"""
import argparse, os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--per_model", default=None)
    ap.add_argument("--routes", default=None, help="optional {id: model_index} for the audited router")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(__file__))
    pmdir = a.per_model or os.path.join(root, "data", "per_model")
    out = a.out or os.path.join(root, "data", "processed", "correctness_kxN.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    files = sorted(glob.glob(os.path.join(pmdir, "m*_*.npz")),
                   key=lambda f: int(os.path.basename(f).split("_")[0][1:]))
    if not files:
        sys.exit(f"no per-model npz in {pmdir}; run run_one_model.py first")
    cols = [np.load(f, allow_pickle=True) for f in files]
    N, k = cols[0]["b_m"].shape; M = len(files)
    b = np.stack([c["b_m"] for c in cols], axis=1).astype(np.int8)        # (N, M, k)
    greedy = np.stack([c["greedy_m"] for c in cols], axis=1).astype(np.int8)
    b_single = b[:, :, 0].copy()
    phat = b.mean(axis=2)
    # answer labels Y (N,M,k) + gold (N,) carried through for the O^agg split (Lemma aggfloor)
    have_Y = all("Y_m" in c.files for c in cols) and "gold" in cols[0].files
    Y = np.stack([c["Y_m"] for c in cols], axis=1) if have_Y else None
    gold = cols[0]["gold"] if have_Y else None
    subset = json.load(open(a.subset)); ids = [q["id"] for q in subset]
    if a.routes:
        route = json.load(open(a.routes)); idx = {q: i for i, q in enumerate(ids)}
        m_bs = int(phat.mean(axis=0).argmax())
        # keep length N and aligned to b: fall back to best-single for any id missing from the route map
        q_router = np.array([phat[idx[q], route[q]] if q in route else phat[idx[q], m_bs]
                             for q in ids], float)
        n_missing = sum(1 for q in ids if q not in route)
        if n_missing:
            print(f"[warn] {n_missing}/{N} ids missing from route map -> filled with best-single")
    else:
        m_bs = int(phat.mean(axis=0).argmax()); q_router = phat[:, m_bs]   # best-single default

    def _scalar(c, key, default=None):
        return c[key].item() if key in c.files else default
    c0 = cols[0]
    save_kw = dict(b=b, b_single=b_single, greedy=greedy, q_router=q_router,
                   meta=json.dumps({"N": int(N), "M": int(M), "k": int(k),
                                    "models": [str(c["model"]) for c in cols],
                                    "router": "best-single" if not a.routes else "audited",
                                    "has_labels": bool(have_Y),
                                    "seed": _scalar(c0, "seed"), "temperature": _scalar(c0, "temperature"),
                                    "top_p": _scalar(c0, "top_p"), "max_tokens": _scalar(c0, "max_tokens")},
                                   ensure_ascii=False))
    if Y is not None:
        save_kw["Y"] = Y; save_kw["gold"] = gold
    np.savez(out, **save_kw)
    print(f"combined {M} models -> {out}  (b shape {b.shape})")
    print("next: python scripts/04_oracles_decompose.py --npz", out)


if __name__ == "__main__":
    main()
