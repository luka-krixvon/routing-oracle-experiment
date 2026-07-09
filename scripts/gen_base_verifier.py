#!/usr/bin/env python3
"""Score every saved HumanEval+ sample against the ORIGINAL HumanEval BASE tests
only (the 7 in-prompt examples) -> b_base tensor. This is the weak, gameable
"real verifier": base tests are visible in the prompt; the full EvalPlus suite
(our primary b tensor) is the held-out truth.

Reuses run_one_model_code's extraction so b_base is measured on the SAME code
that produced the full-suite b. Parallel; base_input is tiny so this is fast.

Run on the VM (evalplus + sandbox). Writes artifacts/humanevalplus/b_base.npz.
Usage: python gen_base_verifier.py [--max-models N] [--workers 8]
"""
import argparse, glob, os, re, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_one_model_code import extract_code, header_imports  # noqa: E402
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash  # noqa: E402
from evalplus.evaluate import get_groundtruth  # noqa: E402
from evalplus.eval import untrusted_check  # noqa: E402

PROBLEMS = get_human_eval_plus()
HASH = get_human_eval_plus_hash()
GT = get_groundtruth(PROBLEMS, HASH, [])  # {tid: {base, base_time, plus, plus_time}}


def build_program(prompt, completion, entry_point):
    """Mirror run_one_model_code.passes_tests's program assembly (minus the test)."""
    code, standalone = extract_code(completion, entry_point)
    return header_imports(prompt) + "\n" + code if standalone else prompt + code


def _base_pass(task):
    tid, completion = task
    p = PROBLEMS[tid]
    try:
        program = build_program(p["prompt"], completion, p["entry_point"])
        status, _ = untrusted_check(
            "humaneval", program, p["base_input"], p["entry_point"],
            expected=GT[tid]["base"], atol=p["atol"], ref_time=GT[tid]["base_time"],
            fast_check=True, min_time_limit=1.0, gt_time_limit_factor=4.0)
        return 1 if status == "pass" else 0
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-models", type=int, default=11)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    files = sorted([f for f in glob.glob(os.path.join(ROOT, "data/per_model_code/m*.npz"))
                    if not f.endswith("_raw.npz")],
                   key=lambda f: int(os.path.basename(f).split("_")[0][1:]))[: a.max_models]
    cols, models, ids_ref = [], [], None
    for f in files:
        d = np.load(f, allow_pickle=True)
        samples = d["samples_m"]              # (N, k) raw chat answers
        ids = [str(x) for x in d["ids"]]
        if ids_ref is None: ids_ref = ids
        assert ids == ids_ref, "id order mismatch across models"
        N, k = samples.shape
        tasks = [(ids[i], samples[i, dd]) for i in range(N) for dd in range(k)]
        bcol = np.zeros((N, k), dtype=np.int8)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for idx, ok in enumerate(ex.map(_base_pass, tasks, chunksize=16)):
                bcol[idx // k, idx % k] = ok
        cols.append(bcol); models.append(str(d["model"]))
        print(f"  {os.path.basename(f)[:34]:34s} base pass-rate = {bcol.mean():.3f}", flush=True)

    b_base = np.stack(cols, axis=1).astype(np.int8)   # (N, M, k)
    out = os.path.join(ROOT, "artifacts/humanevalplus/b_base.npz")
    np.savez_compressed(out, b_base=b_base, ids=np.array(ids_ref, dtype=object), models=models)
    print("wrote", out, "shape", b_base.shape)


if __name__ == "__main__":
    main()
