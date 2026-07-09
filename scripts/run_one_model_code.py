"""Run ONE model on a code benchmark in its own process: load -> generate k
seed-aligned draws (chat-templated, like the math benchmarks) -> extract code
from the chat answer -> execute unit tests -> save this model's correctness
column AND the raw texts -> exit (OS reclaims all VRAM). Launched per-model by
run_sequential_code.sh. RUN IN A SANDBOX/CONTAINER — executes generated code.

v2: (a) saves RAW samples so scoring can be redone offline on CPU;
    (b) extracts code from chat-style answers (``` fences) before executing —
        src.generate uses llm.chat(), so completions are chat answers, not raw
        continuations (v1 scored everything 0 because of this).

Saves data/per_model_code/m{idx}_{safe}.npz with:
    b_m (N,k) int8, samples_m (N,k) object (raw text), ids, model.
Requires: pip install evalplus
"""
import argparse, os, sys, gc, re
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


FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(completion, entry_point):
    """Pull runnable code out of a chat-style answer.
    Returns (code, standalone): standalone=True if the code (re)defines the
    entry point itself; False if it reads as a continuation of the prompt."""
    blocks = FENCE.findall(completion)
    pat = re.compile(r"def\s+" + re.escape(entry_point) + r"\s*\(")
    for b in blocks:
        if pat.search(b):
            return b, True
    if blocks:
        b = blocks[0]
        return b, bool(re.search(r"def\s+\w+\s*\(", b))
    if pat.search(completion):
        return completion, True
    return completion, False


def header_imports(prompt):
    """Import lines from the problem prompt (needed by standalone snippets
    that assume e.g. `from typing import List` from the prompt)."""
    out = []
    for line in prompt.splitlines():
        ls = line.strip()
        if ls.startswith("import ") or ls.startswith("from "):
            out.append(line)
        if ls.startswith("def "):
            break
    return "\n".join(out)


def passes_tests(prompt, completion, prob, timeout=10.0):
    import subprocess, tempfile
    code, standalone = extract_code(completion, prob["entry_point"])
    if standalone:
        program = header_imports(prompt) + "\n" + code
    else:
        program = prompt + code
    program += "\n" + prob["test"] + "\n"
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


def _score_task(t):
    prompt, sample, prob = t
    return passes_tests(prompt, sample, prob, timeout=6.0)


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
    raw_out = out[:-4] + "_raw.npz"
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
    N, k = len(probs), a.k
    if os.path.exists(raw_out):
        print("[resume] raw generations found, skipping GPU:", raw_out)
        samples_m = np.load(raw_out, allow_pickle=True)["samples_m"]
    else:
        print(f"[load] {a.model}")
        llm = LLM(model=a.model, **kw)
        gens = generate(prompts, a.model, "vllm", cfg, llm=llm)
        del llm; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
        samples_m = np.empty((N, k), dtype=object)
        for i, g in enumerate(gens):
            for d, sample in enumerate(g["samples"][:k]):
                samples_m[i, d] = sample
        np.savez_compressed(raw_out, samples_m=samples_m,
                            ids=np.array([p["task_id"] for p in probs], dtype=object))
        print("[raw saved]", raw_out)

    # -------- parallel scoring on CPU (16 workers; hanging code hits per-test timeout) --------
    from concurrent.futures import ProcessPoolExecutor
    b_m = np.zeros((N, k), dtype=np.int8)
    task_args = [(prompts[i], samples_m[i, d], probs[i])
                 for i in range(N) for d in range(k)]
    workers = min(16, os.cpu_count() or 8)
    print(f"[score] {len(task_args)} tests on {workers} workers")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, ok in enumerate(ex.map(_score_task, task_args, chunksize=8)):
            i, d = divmod(idx, k)
            b_m[i, d] = ok
            done += 1
            if done % 400 == 0:
                print(f"  scored {done}/{len(task_args)}  running p={b_m.sum()/done:.3f}")
    np.savez_compressed(out, b_m=b_m, samples_m=samples_m,
                        ids=np.array([p["task_id"] for p in probs], dtype=object),
                        model=a.model, k=k, seed=a.seed, temperature=a.temperature)
    print("[done]", out, "pass@1 =", round(float(b_m[:, 0].mean()), 3),
          "p(mean) =", round(float(b_m.mean()), 3))


if __name__ == "__main__":
    main()
