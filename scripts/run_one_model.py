"""Run ONE model in its own process: load -> generate k seed-aligned draws -> score
-> save this model's correctness column -> free GPU. Because this runs as a separate
subprocess (launched by run_sequential.sh), all VRAM is reclaimed by the OS on exit;
the explicit del/empty_cache below also frees it within-process before we score.

Saves data/per_model/m{idx}_{safe}.npz with: b_m (N,k) 0/1, greedy_m (N,), ids, gold.
Never holds more than one model's weights; run_sequential.sh purges the HF snapshot after.
"""
import argparse, os, sys, json, gc
import numpy as np
# vLLM auto-selects FlashInfer for sampling, which JIT-compiles a CUDA kernel that needs nvcc
# (the CUDA *toolkit*). Runtime-only GPU boxes have the driver but no toolkit -> force the native
# PyTorch sampler. Must be set BEFORE vllm is imported. Override with VLLM_USE_FLASHINFER_SAMPLER=1.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.generate import GenConfig, generate
from src import score as scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)            # HF repo id
    ap.add_argument("--idx", type=int, required=True)    # column index in the pool
    ap.add_argument("--subset", required=True)           # subset.json (id, prompt, gold, task)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--quantization", default=None)      # e.g. "awq" | "awq_marlin" | None(fp16)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--max_model_len", type=int, default=None)  # per-model cap (e.g. 3072 for the tight 32B AWQ)
    ap.add_argument("--seed", type=int, default=42)            # root seed for the seed-aligned draws (A8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(__file__))
    subset = json.load(open(a.subset)); prompts = [q["prompt"] for q in subset]
    safe = a.model.replace("/", "__")
    out = a.out or os.path.join(root, "data", "per_model", f"m{a.idx}_{safe}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        print("[skip] already done:", out); return       # resumable

    cfg = GenConfig(k=a.k, temperature=a.temperature, top_p=a.top_p,
                    max_tokens=a.max_tokens, seed_alignment=True, root_seed=a.seed)
    # build the shared vLLM engine once with memory caps that fit a single 4090
    from vllm import LLM
    max_model_len = a.max_model_len or min(8192, a.max_tokens + 2048)
    llm_kw = {"gpu_memory_utilization": a.gpu_mem_util,
              "tensor_parallel_size": a.tensor_parallel_size,
              "max_model_len": max_model_len, "enforce_eager": True}
    if a.quantization:
        llm_kw["quantization"] = a.quantization
    print(f"[load] {a.model} (quant={a.quantization}, tp={a.tensor_parallel_size})")
    llm = LLM(model=a.model, **llm_kw)
    gens = generate(prompts, a.model, "vllm", cfg, llm=llm)   # seed-aligned k draws + greedy

    # free GPU BEFORE scoring (scoring is pure CPU)
    del llm; gc.collect()
    try:
        import torch
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass

    N, k = len(subset), a.k
    b_m = np.zeros((N, k), dtype=np.int8); greedy_m = np.zeros(N, dtype=np.int8)
    Y_m = np.empty((N, k), dtype=object); gold_m = np.empty(N, dtype=object)
    samples_m = np.empty((N, k), dtype=object); greedy_txt = np.empty(N, dtype=object)
    for i, (q, g) in enumerate(zip(subset, gens)):
        try:                                   # store the CANONICAL (normalized) gold so the
            gold_m[i] = scorer.extract_answer(str(q.get("gold")), q.get("task", "exact"))
        except Exception:                      # O^agg vote (Y labels are normalized) compares like-for-like
            gold_m[i] = str(q.get("gold"))
        for j, s in enumerate(g["samples"][:k]):
            samples_m[i, j] = s or ""
        greedy_txt[i] = g.get("greedy")
    ids = np.array([q["id"] for q in subset])
    # SAVE RAW FIRST: generation is expensive; a scoring bug must never throw it away.
    raw = out[:-4] + "_raw.npz" if out.endswith(".npz") else out + "_raw.npz"
    np.savez_compressed(raw, samples=samples_m, greedy=greedy_txt, ids=ids, gold=gold_m, model=a.model)
    # score (robust): a single bad draw scores 0 / label None, it can never crash the whole model.
    def _safe(fn, *aa):
        try:
            return fn(*aa)
        except Exception:
            return None
    for i, q in enumerate(subset):
        gold, task = q.get("gold"), q.get("task", "exact")
        for j in range(k):
            s = samples_m[i, j] or ""
            b_m[i, j] = _safe(scorer.exact_match, s, gold, task) or 0
            Y_m[i, j] = _safe(scorer.extract_answer, s, task)
        if greedy_txt[i] is not None:
            greedy_m[i] = _safe(scorer.exact_match, greedy_txt[i], gold, task) or 0
    # main (tiny) column: correctness + labels + run config -> combine.py / 04
    np.savez(out, b_m=b_m, greedy_m=greedy_m, Y_m=Y_m, gold=gold_m,
             ids=ids, model=a.model, k=k,
             seed=a.seed, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
             max_model_len=max_model_len, quantization=str(a.quantization),
             tensor_parallel_size=a.tensor_parallel_size)
    if b_m.mean() == 0:
        print(f"[WARN] {a.model}: ALL-ZERO correctness -- check scorer/prompt/extraction before trusting this run!")
    print(f"[done] {a.model}: mean single-draw acc={b_m[:,0].mean():.3f}, mean p_hat={b_m.mean():.3f} -> {out} (+raw)")


if __name__ == "__main__":
    main()
