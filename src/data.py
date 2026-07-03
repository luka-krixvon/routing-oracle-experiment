"""Load benchmark correctness data and build the stratified subset
(oversample the rare-correct stratum that drives the gap).

`load_raw_correctness` is config-driven and supports the two verified formats:
  - mode: wide  (RouterBench 0shot/5shot pkl->parquet): one row per query, per-model
    columns. Models detected via `model_detect_pattern` (e.g. "{model}|model_response"),
    correctness read via `model_correctness_pattern` (e.g. bare "{model}").
  - mode: long  (LLMRouterBench): one row per (query, model); pivot to wide on load.
"""
from __future__ import annotations
import numpy as np


def load_benchmark(name: str = "gsm8k", split: str = "test", n: int | None = None,
                   seed: int = 0) -> list[dict]:
    """Load prompts + GOLD answers directly from a standard benchmark on HuggingFace.

    We generate FRESH samples from our own model pool, so we only need clean
    (prompt, gold, task) — not any pre-scored matrix. Standard benchmarks expose a
    clean gold field, unlike RouterBench's pickle. Returns [{id, prompt, gold, task}].

    Supported: gsm8k (numeric exact-match), mmlu (multiple-choice A-D),
    math500 (free-form boxed), gpqa (GPQA-Diamond, graduate-science multiple-choice
    A-D — the hard, unsaturated NON-math exact-match niche). Extend as needed.
    Requires the `datasets` library (see requirements.txt). Runs on the GPU VM.
    """
    from datasets import load_dataset
    import re as _re
    rng = np.random.default_rng(seed)
    recs = []
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for i, ex in enumerate(ds):
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            prompt = (ex["question"].strip() +
                      "\n\nSolve step by step and put the final numeric answer on the last line as "
                      "'#### <number>'.")
            recs.append({"id": f"gsm8k-{i}", "prompt": prompt, "gold": gold, "task": "gsm8k"})
    elif name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split=split)
        letters = ["A", "B", "C", "D"]
        for i, ex in enumerate(ds):
            ch = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(ex["choices"]))
            prompt = (f"{ex['question'].strip()}\n{ch}\n\nAnswer with the single letter "
                      "(A, B, C, or D) of the correct choice.")
            recs.append({"id": f"mmlu-{i}", "prompt": prompt, "gold": letters[ex["answer"]],
                         "task": "mmlu_pro"})
    elif name == "math500":
        # 500 hard competition-math problems (free-form). Much less saturated than GSM8K,
        # so the rare-correct / thin-support stratum is large -> shows the noise-dominant regime.
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")   # only split
        for i, ex in enumerate(ds):
            prompt = (ex["problem"].strip() +
                      "\n\nPlease reason step by step, and put your final answer inside \\boxed{}.")
            recs.append({"id": f"math500-{i}", "prompt": prompt, "gold": ex["answer"],
                         "task": "math500"})
    elif name == "gpqa":
        # GPQA-Diamond: 198 expert-written, Google-proof graduate-level questions in
        # biology/physics/chemistry. NON-math, 4-way multiple-choice, exact-match, and
        # deeply UNSATURATED for a 7-14B open pool (domain experts out-of-field score ~34%),
        # so the rare-correct / thin-support stratum is large -> the noise-dominant regime
        # the localization needs, on a benchmark that fits one 24GB GPU (198 items, k draws).
        #
        # Primary source is the canonical (gated) dataset: raw fields Question / Correct
        # Answer / Incorrect Answer 1..3. We shuffle the four options with a per-query,
        # seed-derived RNG (reproducible; A8-compatible) and record the gold LETTER, exactly
        # as lm-eval-harness / the GPQA paper build the prompt. If the gated repo is not
        # accessible (no HF login / license), we fall back to an ungated pre-formatted mirror
        # whose `problem` already embeds the choices and whose `answer` is the gold letter.
        letters = ["A", "B", "C", "D"]

        def _mc_prompt(question: str, choices: list[str]) -> str:
            ch = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices))
            return (f"{question.strip()}\n{ch}\n\nAnswer with the single letter "
                    "(A, B, C, or D) of the correct choice.")
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            for i, ex in enumerate(ds):
                q = str(ex["Question"]).strip()
                opts = [str(ex["Correct Answer"]).strip(),
                        str(ex["Incorrect Answer 1"]).strip(),
                        str(ex["Incorrect Answer 2"]).strip(),
                        str(ex["Incorrect Answer 3"]).strip()]
                # per-query deterministic shuffle so the correct option is not always 'A'
                perm = np.random.default_rng([seed, i]).permutation(4)
                shuffled = [opts[p] for p in perm]
                gold_letter = letters[int(np.where(perm == 0)[0][0])]
                recs.append({"id": f"gpqa-{i}", "prompt": _mc_prompt(q, shuffled),
                             "gold": gold_letter, "task": "gpqa"})
        except Exception as e:
            print(f"[gpqa] canonical Idavidrein/gpqa unavailable ({type(e).__name__}: {e});"
                  " falling back to ungated mirror aradhye/gpqa_diamond")
            ds = load_dataset("aradhye/gpqa_diamond", split="train")   # `problem` embeds
            for i, ex in enumerate(ds):                                # choices; `answer` is A-D
                prompt = (str(ex["problem"]).strip() + "\n\nAnswer with the single letter "
                          "(A, B, C, or D) of the correct choice.")
                recs.append({"id": f"gpqa-{i}", "prompt": prompt,
                             "gold": str(ex["answer"]).strip(), "task": "gpqa"})
    else:
        raise ValueError(f"unknown benchmark {name!r} (supported: gsm8k, mmlu, math500, gpqa; "
                         "for RouterBench/LLMRouterBench use load_raw_correctness)")
    if n is not None and n < len(recs):
        idx = rng.choice(len(recs), n, replace=False)
        recs = [recs[i] for i in sorted(idx)]
    return recs


def stratify_by_num_correct(b_single: np.ndarray) -> np.ndarray:
    """Per-query count of #models correct on a single draw (the stratum key)."""
    return (np.asarray(b_single) > 0).sum(axis=1).astype(int)


def make_subset_indices(num_correct: np.ndarray, subset_size: int,
                        rare_max: int = 3, oversample_rare_frac: float = 0.40,
                        seed: int = 0) -> np.ndarray:
    """Stratified subset oversampling the rare-correct stratum (#correct<=rare_max)."""
    rng = np.random.default_rng(seed)
    nc = np.asarray(num_correct)
    rare = np.where(nc <= rare_max)[0]
    common = np.where(nc > rare_max)[0]
    n_rare = min(len(rare), int(round(subset_size * oversample_rare_frac)))
    n_common = min(len(common), subset_size - n_rare)
    pick_rare = rng.choice(rare, n_rare, replace=False) if n_rare else np.array([], int)
    pick_common = rng.choice(common, n_common, replace=False) if n_common else np.array([], int)
    idx = np.concatenate([pick_rare, pick_common]); rng.shuffle(idx)
    return idx


def _read(schema):
    import pandas as pd
    fmt = schema.get("format", "parquet")
    return pd.read_parquet(schema["path"]) if fmt == "parquet" else pd.read_csv(schema["path"])


def _to_binary(perf: np.ndarray, threshold):
    return (perf >= threshold).astype(int) if threshold is not None else (perf > 0).astype(int)


def _finalize(b_single, models, dataset, query_ids, judge_datasets):
    judge = set(judge_datasets or [])
    is_judge = np.array([str(d) in judge for d in dataset], dtype=bool)
    return {"b_single": b_single, "model_ids": list(models),
            "dataset": np.asarray(dataset), "is_judge": is_judge,
            "query_ids": np.asarray(query_ids)}


def load_raw_correctness(schema: dict) -> dict:
    """Return dict(b_single (N,M), model_ids, dataset (N,), is_judge (N,), query_ids)."""
    mode = schema.get("mode", "wide")
    df = _read(schema)

    if mode == "wide":
        # detect models from a non-empty detect pattern, read correctness via corr pattern
        detect_pat = schema.get("model_detect_pattern") or schema["model_correctness_pattern"]
        if "{model}" not in detect_pat or detect_pat == "{model}":
            if not schema.get("models"):
                raise ValueError("bare/empty detect pattern needs explicit data_schema.models "
                                 "or a non-empty model_detect_pattern (e.g. '{model}|model_response')")
        pre, suf = detect_pat.split("{model}")
        models = schema.get("models") or [
            c[len(pre):len(c) - len(suf)] for c in df.columns
            if c.startswith(pre) and c.endswith(suf) and len(c) > len(pre) + len(suf)
        ]
        if not models:
            raise ValueError(f"no models matched detect pattern {detect_pat!r}")
        corr_pat = schema["model_correctness_pattern"]
        cols = [corr_pat.format(model=m) for m in models]
        perf = df[cols].to_numpy(dtype=float)
        b_single = _to_binary(perf, schema.get("correctness_threshold"))
        dcol = schema.get("dataset_col")
        dataset = df[dcol].astype(str).to_numpy() if dcol in df.columns else np.array([""] * len(df))
        qcol = schema.get("prompt_col")
        qids = df[qcol].astype(str).to_numpy() if qcol in df.columns else np.arange(len(df)).astype(str)
        return _finalize(b_single, models, dataset, qids, schema.get("judge_datasets"))

    elif mode == "long":
        import pandas as pd
        pcol, mcol, scol = schema["prompt_col"], schema["model_col"], schema["score_col"]
        piv = df.pivot_table(index=pcol, columns=mcol, values=scol, fill_value=0.0)
        models = list(piv.columns)
        b_single = _to_binary(piv.to_numpy(dtype=float), schema.get("correctness_threshold"))
        dcol = schema.get("dataset_col")
        if dcol in df.columns:
            dmap = df.groupby(pcol)[dcol].first()
            dataset = dmap.reindex(piv.index).astype(str).to_numpy()
        else:
            dataset = np.array([""] * len(piv))
        return _finalize(b_single, models, dataset, piv.index.to_numpy(), schema.get("judge_datasets"))

    raise ValueError(f"unknown data_schema.mode {mode!r} (use 'wide' or 'long')")
