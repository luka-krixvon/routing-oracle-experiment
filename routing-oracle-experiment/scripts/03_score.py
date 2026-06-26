"""Stage 03 — score the raw generations from 02 into the (N,M,k) correctness tensor.

Reads data/raw/gen_m*.jsonl (one file per model, written by 02_generate.py), scores each
of the k seed-aligned samples against the gold answer (exact-match scorers in src.score),
and writes data/processed/correctness_kxN.npz with:
  b        (N,M,k)  0/1 correctness, ALIGNED by draw index j (preserves A8 for the seed-aligned O^exp)
  b_single (N,M)    the one recorded label per cell (draw index 0)
  greedy   (N,M)    greedy (T=0) correctness
  q_router (N,)     router correctness; default = best-single baseline, or per --routes
  meta     json
Exact-match and LLM-judge datasets are scored separately (judge handled elsewhere).
"""
import argparse, os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="subset .json from 01 (id, gold, task)")
    ap.add_argument("--raw", default=None, help="dir with gen_m*.jsonl (default data/raw)")
    ap.add_argument("--task", default=None, help="override task name for all (e.g. mmlu_pro)")
    ap.add_argument("--routes", default=None, help="optional json {id: model_index} for the audited router")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(__file__))
    rawdir = a.raw or os.path.join(root, "data", "raw")
    out = a.out or os.path.join(root, "data", "processed", "correctness_kxN.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    subset = json.load(open(a.subset))
    ids = [q["id"] for q in subset]; idx = {qid: i for i, qid in enumerate(ids)}
    gold = {q["id"]: q.get("gold") for q in subset}
    task = {q["id"]: (a.task or q.get("task", "exact")) for q in subset}
    files = sorted(glob.glob(os.path.join(rawdir, "gen_m*.jsonl")))
    if not files:
        sys.exit(f"no gen_m*.jsonl in {rawdir}; run 02_generate.py first")
    M = len(files); N = len(ids)
    k = len(json.loads(open(files[0]).readline())["samples"])
    b = np.zeros((N, M, k), dtype=int); greedy = np.zeros((N, M), dtype=int)
    for mi, fp in enumerate(files):
        for line in open(fp):
            r = json.loads(line); i = idx.get(r["id"])
            if i is None: continue
            g = gold[r["id"]]; t = task[r["id"]]
            for j, samp in enumerate(r["samples"][:k]):
                b[i, mi, j] = score.exact_match(samp or "", g, t)
            if r.get("greedy") is not None:
                greedy[i, mi] = score.exact_match(r["greedy"], g, t)
    b_single = b[:, :, 0].copy()
    phat = b.mean(axis=2)
    if a.routes:
        route = json.load(open(a.routes))                       # {id: model_index}
        q_router = np.array([phat[idx[qid], route[qid]] for qid in ids if qid in route], float)
    else:
        m_bs = int(phat.mean(axis=0).argmax())                  # default: best-single-model baseline
        q_router = phat[:, m_bs]
    np.savez(out, b=b, b_single=b_single, greedy=greedy, q_router=q_router,
             meta=json.dumps({"N": N, "M": M, "k": k, "models": [os.path.basename(f) for f in files],
                              "router": "best-single" if not a.routes else "audited"}, ensure_ascii=False))
    print(f"scored {N}x{M}x{k} -> {out}\nnext: python scripts/04_oracles_decompose.py --npz {out}")


if __name__ == "__main__":
    main()
