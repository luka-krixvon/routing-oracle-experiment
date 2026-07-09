"""Generate a code (pass@k) `correctness_slim.npz`, REUSING src.generate for
generation (same models / T=0.2 / k seed-aligned draws as math500/gpqa) and
EvalPlus for execution-based scoring. RUN ON THE GPU VM + IN A SANDBOX
(it executes model-generated code).

    pip install evalplus
    python scripts/gen_code_tensor.py --benchmark humanevalplus --k 20
    python scripts/gen_code_tensor.py --benchmark mbppplus      --k 20

Output: artifacts/<benchmark>/correctness_slim.npz  (same keys/format as the
math500/gpqa tensors) -> resample-or-reroute/experiments/run_pareto.py reads it
unchanged. Correctness = sample passes ALL unit tests (per-draw pass@1;
best-of-K over draws == pass@k, matching Thm. 2(b)).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# make `src` importable when run as `python scripts/gen_code_tensor.py` (mirrors 01/02_*.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# runtime-only GPU box (driver, no CUDA toolkit): force native sampler so vLLM does NOT
# JIT-compile FlashInfer (needs nvcc). Must be set BEFORE vllm is imported. (mirrors run_one_model.py)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODELS = [  # same 11-model pool as math500/gpqa
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


def load_problems(bench):
    if bench == "humanevalplus":
        from evalplus.data import get_human_eval_plus
        data = get_human_eval_plus()
    elif bench == "mbppplus":
        from evalplus.data import get_mbpp_plus
        data = get_mbpp_plus()
    else:
        raise ValueError(f"code bench must be humanevalplus | mbppplus, got {bench!r}")
    return [dict(task_id=t, prompt=p["prompt"], test=p.get("test", ""),
                 entry_point=p.get("entry_point", "")) for t, p in data.items()]


def passes_tests(prompt, completion, prob, timeout=10.0):
    """1 iff prompt+completion passes the problem's tests. SANDBOX THIS.

    For the paper, prefer EvalPlus's official evaluator; this minimal subprocess
    runner is a portable fallback.
    """
    program = prompt + completion + "\n" + prob["test"] + "\n"
    if prob["entry_point"]:
        program += f"\ncheck({prob['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return int(r.returncode == 0)
    except Exception:
        return 0
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="humanevalplus", help="humanevalplus | mbppplus")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    from src.generate import generate, GenConfig  # reuse the shared generation pipeline

    models = args.models or MODELS
    probs = load_problems(args.benchmark)
    if args.limit:
        probs = probs[: args.limit]
    prompts = [p["prompt"] for p in probs]
    N, M, k = len(probs), len(models), args.k
    print(f"{args.benchmark}: N={N} x M={M} x k={k}  (backend={args.backend})")

    b = np.zeros((N, M, k), dtype=np.int8)
    greedy = np.zeros((N, M), dtype=np.int8)
    gcfg = GenConfig(k=k, temperature=0.2, top_p=1.0)

    for mi, model_id in enumerate(models):
        print(f"[{mi+1}/{M}] {k} draws @T=0.2 for {model_id}")
        gens = generate(prompts, model_id, args.backend, gcfg)  # [{"samples":[k], "greedy":...}]
        for i, g in enumerate(gens):
            for d, s in enumerate(g["samples"][:k]):
                b[i, mi, d] = passes_tests(prompts[i], s, probs[i])
            if g.get("greedy") is not None:
                greedy[i, mi] = passes_tests(prompts[i], g["greedy"], probs[i])

    out = os.path.join(os.path.dirname(__file__), "..", "artifacts", args.benchmark)
    os.makedirs(out, exist_ok=True)
    meta = {"N": N, "M": M, "k": k, "models": list(models)}
    np.savez_compressed(
        os.path.join(out, "correctness_slim.npz"),
        b=b, b_single=b[:, :, 0].astype(np.int8), greedy=greedy,
        q_router=np.zeros(N, dtype=float),
        gold=np.array([p["task_id"] for p in probs], dtype=object),
        meta=json.dumps(meta),
    )
    print("wrote", os.path.join(out, "correctness_slim.npz"))
    print("per-model pass@1 (reproducible p):", np.round(b.mean(axis=(0, 2)), 3))


if __name__ == "__main__":
    main()
