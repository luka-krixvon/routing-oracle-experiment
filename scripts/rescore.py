"""Re-score every per-model column from its saved RAW completions under the CURRENT
src/score.py, so ALL models -- including ones generated under an older scorer -- are
scored by identical code. CPU-only (seconds); needs the m*_raw.npz sidecars. Run once
after generation, before combine, to guarantee scoring consistency without re-running
the GPU. Prints old->new p_hat per model with a CHANGED flag.

  python scripts/rescore.py            # rescore all in data/per_model using data/subset.json
"""
import argparse, os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import score as scorer


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_model", default=None)
    ap.add_argument("--subset", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(__file__))
    pmdir = a.per_model or os.path.join(root, "data", "per_model")
    subset = json.load(open(a.subset or os.path.join(root, "data", "subset.json")))
    by_id = {q["id"]: q for q in subset}
    mains = sorted([f for f in glob.glob(os.path.join(pmdir, "m*_*.npz")) if not f.endswith("_raw.npz")])
    if not mains:
        sys.exit(f"no per-model npz in {pmdir}")
    changed = 0
    for mf in mains:
        raw = mf[:-4] + "_raw.npz"
        if not os.path.exists(raw):
            print(f"[skip] {os.path.basename(mf):52} no _raw sidecar -- cannot re-score without raw text")
            continue
        d = dict(np.load(mf, allow_pickle=True))                 # preserve every field
        r = np.load(raw, allow_pickle=True)
        samples = r["samples"]; ids = [str(x) for x in r["ids"]]
        N, k = samples.shape
        b = np.zeros((N, k), dtype=np.int8); Y = np.empty((N, k), dtype=object)
        gold_arr = np.empty(N, dtype=object)
        for i, qid in enumerate(ids):
            q = by_id.get(qid, {}); gold, task = q.get("gold"), q.get("task", "exact")
            gold_arr[i] = _safe(scorer.extract_answer, str(gold), task)   # canonical gold
            for j in range(k):
                s = samples[i, j] or ""
                b[i, j] = _safe(scorer.exact_match, s, gold, task) or 0
                Y[i, j] = _safe(scorer.extract_answer, s, task)
        old_p = float(np.asarray(d["b_m"], float).mean()); new_p = float(b.mean())
        d["b_m"] = b; d["Y_m"] = Y; d["gold"] = gold_arr
        if "greedy" in r.files:                                   # re-score greedy too if raw kept it
            g = r["greedy"]; gm = np.zeros(N, dtype=np.int8)
            for i, qid in enumerate(ids):
                q = by_id.get(qid, {})
                if g[i] is not None:
                    gm[i] = _safe(scorer.exact_match, g[i], q.get("gold"), q.get("task", "exact")) or 0
            d["greedy_m"] = gm
        np.savez(mf, **d)
        diff = abs(old_p - new_p)
        flag = "" if diff < 1e-9 else f"  <-- CHANGED by {diff:+.4f}"
        changed += diff >= 1e-9
        print(f"[rescore] {os.path.basename(mf):52} p_hat {old_p:.4f} -> {new_p:.4f}{flag}")
    print(f"done. re-scored {len(mains)} model(s) under the current src/score.py; {changed} changed.")


if __name__ == "__main__":
    main()
