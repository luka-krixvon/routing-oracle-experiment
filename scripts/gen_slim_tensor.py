"""Generate a `correctness_slim.npz` for an exact-match benchmark, REUSING the
same pipeline (models, T=0.2 k draws, answer extraction) as math500/gpqa so the
tensor is scientifically consistent with the existing artifacts.

RUN ON THE GPU VM. Example:
    python scripts/gen_slim_tensor.py --benchmark gsm8k --k 30
    python scripts/gen_slim_tensor.py --benchmark gsm8k --k 30 --n 500   # cap items

Output: artifacts/<benchmark>/correctness_slim.npz with the SAME keys as the
existing math500/gpqa tensors:
    b (N,M,k) int8, b_single (N,M) int8, greedy (N,M) int8,
    q_router (N,) float, gold (N,) object, meta (json str).

This is exactly what resample-or-reroute/experiments/run_pareto.py consumes.
"""
import argparse
import json
import os

import numpy as np

# same 11-model pool as the existing math500/gpqa tensors (keep consistent)
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "microsoft/phi-4",
    "allenai/OLMo-2-1124-7B-Instruct",
    "01-ai/Yi-1.5-9B-Chat",
    "ibm-granite/granite-3.3-8b-instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="gsm8k", help="gsm8k | math500 | gpqa | mmlu")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--n", type=int, default=None, help="cap #items (default: full split)")
    ap.add_argument("--backend", default="vllm", help="vllm | api")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=None, help="override model pool")
    args = ap.parse_args()

    # import here so --help works without the heavy deps
    from src.data import load_benchmark
    from src.generate import generate, GenConfig
    from src.score import exact_match

    models = args.models or MODELS
    recs = load_benchmark(args.benchmark, split=args.split, n=args.n, seed=args.seed)
    prompts = [r["prompt"] for r in recs]
    golds = [r["gold"] for r in recs]
    tasks = [r["task"] for r in recs]
    N, M, k = len(recs), len(models), args.k
    print(f"{args.benchmark}: N={N} x M={M} x k={k}  (backend={args.backend})")

    b = np.zeros((N, M, k), dtype=np.int8)
    greedy = np.zeros((N, M), dtype=np.int8)
    gcfg = GenConfig(k=k, temperature=0.2, top_p=1.0)  # match math500/gpqa; add max_tokens if your config sets it

    for mi, model_id in enumerate(models):
        print(f"[{mi+1}/{M}] {k} seed-aligned draws @T=0.2 for {model_id}")
        gens = generate(prompts, model_id, args.backend, gcfg)  # [{"samples":[k], "greedy":...}]
        for i, g in enumerate(gens):
            for d, s in enumerate(g["samples"][:k]):
                b[i, mi, d] = exact_match(s, golds[i], tasks[i])
            if g.get("greedy") is not None:
                greedy[i, mi] = exact_match(g["greedy"], golds[i], tasks[i])

    out = os.path.join(os.path.dirname(__file__), "..", "artifacts", args.benchmark)
    os.makedirs(out, exist_ok=True)
    meta = {"N": N, "M": M, "k": k, "models": list(models)}
    np.savez_compressed(
        os.path.join(out, "correctness_slim.npz"),
        b=b,
        b_single=b[:, :, 0].astype(np.int8),
        greedy=greedy,
        q_router=np.zeros(N, dtype=float),   # placeholder: fill with your learned router if desired
        gold=np.array(golds, dtype=object),
        meta=json.dumps(meta),
    )
    print("wrote", os.path.join(out, "correctness_slim.npz"))
    print("per-model reproducible p:", np.round(b.mean(axis=(0, 2)), 3))


if __name__ == "__main__":
    main()
