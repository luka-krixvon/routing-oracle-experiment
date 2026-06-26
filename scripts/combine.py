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
    subset = json.load(open(a.subset)); ids = [q["id"] for q in subset]
    if a.routes:
        route = json.load(open(a.routes)); idx = {q: i for i, q in enumerate(ids)}
        q_router = np.array([phat[idx[q], route[q]] for q in ids if q in route], float)
    else:
        m_bs = int(phat.mean(axis=0).argmax()); q_router = phat[:, m_bs]   # best-single default
    np.savez(out, b=b, b_single=b_single, greedy=greedy, q_router=q_router,
             meta=json.dumps({"N": int(N), "M": int(M), "k": int(k),
                              "models": [str(c["model"]) for c in cols],
                              "router": "best-single" if not a.routes else "audited"},
                             ensure_ascii=False))
    print(f"combined {M} models -> {out}  (b shape {b.shape})")
    print("next: python scripts/04_oracles_decompose.py --npz", out)


if __name__ == "__main__":
    main()
