"""k-sample generation per (query, model) under stochastic decoding.

Two backends:
  - vllm   : self-host the open 7-9B primary pool (cheap, GPU-only)
  - api    : OpenAI-compatible / frontier secondary pool (budgeted)
Both return, per prompt: {"samples": [str]*k, "greedy": str}.
Imports are lazy so this module loads without vllm/openai installed.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GenConfig:
    k: int = 20                  # samples per (query, model); >=20 overall, >=30 in the rare stratum
    temperature: float = 0.2     # MATCH the benchmark's own T=0.2 (re-drawing hotter manufactures noise)
    top_p: float = 1.0           # benchmark setting (top_p 1.0)
    add_greedy: bool = True
    max_tokens: int = 2048       # generous; avoid truncation artifacts
    seed_alignment: bool = True  # A8: draw index j uses a seed shared across models -> aligned K-tuples
    root_seed: int = 42


def generate(prompts: list[str], model_id: str, backend: str, cfg: GenConfig,
             **kw) -> list[dict]:
    if backend == "vllm":
        return _gen_vllm(prompts, model_id, cfg, **kw)
    if backend in ("api", "openai"):
        return _gen_api(prompts, model_id, cfg, **kw)
    raise ValueError(f"unknown backend {backend!r} (use 'vllm' or 'api')")


def _gen_vllm(prompts, model_id, cfg: GenConfig, llm=None, **kw):
    """Offline vLLM. Pass a shared `llm=LLM(model_id)` to avoid reloading.

    With cfg.seed_alignment, draw index j uses seed = root_seed + j, identical across
    ALL models (the seed depends only on j, not on model_id), so the resulting
    b[i,m,j] are seed-aligned K-tuples and the seed-aligned O^exp estimator is unbiased (A8).
    """
    from vllm import LLM, SamplingParams
    if llm is None:
        llm = LLM(model=model_id, **kw)
    # Apply each model's CHAT TEMPLATE via llm.chat() -- feeding raw strings to
    # llm.generate() skips the template, so template-sensitive instruct models (e.g.
    # phi-4) ignore the question and ramble off-topic. chat() wraps each prompt as a
    # user turn and applies the tokenizer's chat template (add_generation_prompt).
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    if cfg.seed_alignment:
        res = [{"samples": [None] * cfg.k, "greedy": None} for _ in prompts]
        for j in range(cfg.k):
            sp = SamplingParams(n=1, temperature=cfg.temperature, top_p=cfg.top_p,
                                max_tokens=cfg.max_tokens, seed=cfg.root_seed + j)
            for i, out in enumerate(llm.chat(msgs, sampling_params=sp, use_tqdm=True)):
                res[i]["samples"][j] = out.outputs[0].text
    else:
        sp = SamplingParams(n=cfg.k, temperature=cfg.temperature,
                            top_p=cfg.top_p, max_tokens=cfg.max_tokens)
        res = [{"samples": [o.text for o in out.outputs], "greedy": None}
               for out in llm.chat(msgs, sampling_params=sp, use_tqdm=True)]
    if cfg.add_greedy:
        gp = SamplingParams(n=1, temperature=0.0, max_tokens=cfg.max_tokens)
        for i, out in enumerate(llm.chat(msgs, sampling_params=gp, use_tqdm=True)):
            res[i]["greedy"] = out.outputs[0].text
    return res


def _gen_api(prompts, model_id, cfg: GenConfig, client=None, base_url=None, **kw):
    """OpenAI-compatible chat API (works for many frontier/open providers)."""
    from openai import OpenAI
    client = client or OpenAI(base_url=base_url)
    res = []
    for p in prompts:
        msg = [{"role": "user", "content": p}]
        c = client.chat.completions.create(model=model_id, messages=msg, n=cfg.k,
                                            temperature=cfg.temperature, top_p=cfg.top_p,
                                            max_tokens=cfg.max_tokens)
        item = {"samples": [ch.message.content for ch in c.choices], "greedy": None}
        if cfg.add_greedy:
            g = client.chat.completions.create(model=model_id, messages=msg, n=1,
                                               temperature=0.0, max_tokens=cfg.max_tokens)
            item["greedy"] = g.choices[0].message.content
        res.append(item)
    return res
