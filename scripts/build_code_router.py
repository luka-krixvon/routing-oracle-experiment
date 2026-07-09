#!/usr/bin/env python3
"""Backfill the learned-router baseline (q_router) + rich meta into a code
benchmark's correctness_slim.npz.

The code generators (gen_code_tensor.py / run_sequential_code.sh combine) write
`q_router = zeros` as a placeholder, so the `learned_router` pareto baseline reads
0.0. This script fills it with the SAME router the exact-match benchmarks use:
06_robustness.py's pure-numpy TF-IDF + cosine-kNN (5-fold out-of-fold, knn=25),
routing each query to the model with the highest mean k=1 correctness among its
nearest TF-IDF neighbours, then q_router[i] = b_single[i, routed[i]].

It also enriches `meta` to match the exact-match tensors (temperature/top_p/
max_tokens/seed/router/has_labels) so the four benchmarks are self-documenting
identically.

Prereqs in artifacts/<bench>/:
  * correctness_slim.npz  (b, b_single, greedy, q_router, gold, meta)
  * subset.json           [{id, prompt}, ...]  aligned to the tensor's `gold` ids

Usage:
  python scripts/build_code_router.py --bench humanevalplus
  python scripts/build_code_router.py --bench humanevalplus --max_tokens 2048
"""
import argparse
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load_router_fns():
    """Import _tfidf + learned_knn_router verbatim from 06_robustness.py so the
    code benchmark uses byte-identical router logic to the exact-match ones."""
    spec = importlib.util.spec_from_file_location(
        "_rob", os.path.join(HERE, "06_robustness.py"))
    rob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rob)
    return rob._tfidf, rob.learned_knn_router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="humanevalplus")
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--knn", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--router_seed", type=int, default=0)
    # sampling params for meta enrichment (not saved by the code combine)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--gen_seed", type=int, default=42)
    a = ap.parse_args()

    bench_dir = os.path.join(a.artifacts, a.bench)
    tensor_path = os.path.join(bench_dir, "correctness_slim.npz")
    subset_path = os.path.join(bench_dir, "subset.json")
    for p in (tensor_path, subset_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    tfidf, knn_router = _load_router_fns()

    d = dict(np.load(tensor_path, allow_pickle=True))
    b = d["b"]                        # (N, M, k)
    b_single = d["b_single"]          # (N, M)  = T=0.2 first draw
    ids = [str(x) for x in d["gold"]]
    meta = json.loads(str(d["meta"]))

    sub = {it["id"]: it["prompt"] for it in json.load(open(subset_path))}
    missing = [i for i in ids if i not in sub]
    if missing:
        raise KeyError(f"{len(missing)} tensor ids missing from subset.json, "
                       f"e.g. {missing[:3]}")
    prompts = [sub[i] for i in ids]
    assert len(prompts) == b.shape[0], (len(prompts), b.shape)

    X = tfidf(prompts)
    routed = knn_router(b_single, X, knn=a.knn, folds=a.folds, seed=a.router_seed)
    q = b_single[np.arange(len(routed)), routed].astype(float)

    d["q_router"] = q
    meta.update({
        "router": {"type": "tfidf_knn_5fold_oof", "knn": a.knn,
                   "folds": a.folds, "seed": a.router_seed},
        "has_labels": True,
        "temperature": a.temperature,
        "top_p": a.top_p,
        "max_tokens": a.max_tokens,
        "seed": a.gen_seed,
    })
    d["meta"] = json.dumps(meta)
    np.savez_compressed(tensor_path, **d)

    models = meta["models"]
    routed_hist = {models[m].split("/")[-1]: int((routed == m).sum())
                   for m in np.unique(routed)}
    print(f"[{a.bench}] q_router backfilled: learned_router acc = {q.mean():.3f} "
          f"(vocab={X.shape[1]}, N={b.shape[0]}, M={b.shape[1]}, k={b.shape[2]})")
    print(f"  routed-to histogram: {routed_hist}")
    print(f"  meta now carries: {sorted(meta.keys())}")
    print(f"  wrote {tensor_path}")


if __name__ == "__main__":
    main()
