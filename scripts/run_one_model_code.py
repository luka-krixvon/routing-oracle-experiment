"""Run ONE model on a code benchmark in its own process: load -> generate k
seed-aligned draws -> execute unit tests -> save this model's correctness
column -> exit (OS reclaims all VRAM). Mirrors run_one_model.py; launched
per-model by run_sequential_code.sh so only one model's weights are ever
resident. RUN IN A SANDBOX/CONTAINER — it executes model-generated code.

Saves data/per_model_code/m{idx}_{safe}.npz with: b_m (N,k) 0/1, ids, model.
Requires: pip install evalplus
"""
import argparse, os, sys, gc, json
import numpy as np

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # runtime-only box: no nvcc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.generate import GenConfig, generate


def load_problems(bench):
    if bench == "humanevalplus":
        from evalplus.data import get_human_eval_plus
        data = get_human_eval_plus()
    elif bench == "mbppplus":
        from evalplus.data import get_mbpp_plus
        data = get_mbpp_plus()
    else:
        raise ValueError("bench must be humanevalplus|mbppplus")
    return [dict(task_id=t, prompt=p["prompt"], test=p.get("test", ""),
                 entry_point=p.get("entry_point", "")) for t, p in data.items()]


def passes_tests(prompt, completion, prob, timeout=10.0):
    import subprocess, tempfile
    program = prompt + completion + "\n" + prob["test"] + "\n"
    if prob["entry_point"]:
        program += f"\ncheck({prob['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return int(r.returncode == 0)
    except Exception:
        return 0
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--bench", default="humanevalplus")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probs = load_problems(a.bench)
    prompts = [p["prompt"] for p in probs]
    safe = a.model.replace("/", "__")
    outdir = os.path.join(root, "data", "per_model_code")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"m{a.idx}_{safe}.npz")
    if os.path.exists(out):
        print("[skip] already done:", out); return

    cfg = GenConfig(k=a.k, temperature=a.temperature, top_p=a.top_p,
                    max_tokens=a.max_tokens, seed_alignment=True, root_seed=a.seed)
    from vllm import LLM
    max_model_len = a.max_model_len or min(4096, a.max_tokens + 2048)
    kw = {"gpu_memory_utilization": 0.92, "tensor_parallel_size": a.tensor_parallel_size,
          "max_model_len": max_model_len, "enforce_eager": True}
    if a.quantization:
        kw["quantization"] = a.quantization
    print(f"[load] {a.model}")
    llm = LLM(model=a.model, **kw)
    gens = generate(prompts, a.model, "vllm", cfg, llm=llm)
    del llm; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    N, k = len(probs), a.k
    b_m = np.zeros((N, k), dtype=np.int8)
    for i, g in enumerate(gens):
        for d, sample in enumerate(g["samples"][:k]):
            b_m[i, d] = passes_tests(prompts[i], sample, probs[i])
        if (i + 1) % 20 == 0:
            print(f"  scored {i+1}/{N}")
    np.savez(out, b_m=b_m, ids=np.array([p["task_id"] for p in probs], dtype=object),
             model=a.model, k=k, seed=a.seed, temperature=a.temperature)
    print("[done]", out, "pass@1 =", round(float(b_m[:, 0].mean()), 3),
          "p(mean) =", round(float(b_m.mean()), 3))


if __name__ == "__main__":
    main()
