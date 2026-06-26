"""Stage 02 — produce the per-(query, model) draws.

  --simulate : NO GPU. Draw the (N,M,k) correctness tensor from a latent per-cell
               probability (src.simulate) and write data/processed/correctness_kxN.npz
               directly (scoring is identity for simulated correctness). Lets the whole
               02 -> 04 pipeline run on a laptop for validation.
  real       : loop the subset (from 01) x the model pool (from --config), call
               src.generate (vllm/api) with T=0.2, top_p=1.0, seed-aligned k draws,
               and save raw generations as JSONL for 03_score.py to score.
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true", help="no-GPU synthetic correctness")
    ap.add_argument("--N", type=int, default=200); ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--scenario", default="thin")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    ap.add_argument("--config", help="pool/sampling config (real run)")
    ap.add_argument("--subset", help="path to subset prompts .json from 01 (real run)")
    ap.add_argument("--backend", default="vllm")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(__file__))

    if a.simulate:
        from src import simulate
        out = a.out or os.path.join(root, "data", "processed", "correctness_kxN.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        sim = simulate.simulate_correctness(a.N, a.M, a.k, a.scenario, a.seed)
        np.savez(out, b=sim["b"], b_single=sim["b_single"], q_router=sim["q_router"],
                 meta=json.dumps(sim["meta"], ensure_ascii=False))
        print("SIMULATED correctness ->", out, "| shape", sim["b"].shape,
              "\nnext: python scripts/04_oracles_decompose.py --npz", out)
        return

    # ---------------- real GPU/API path ----------------
    import yaml
    from src.generate import GenConfig, generate
    cfg = yaml.safe_load(open(a.config))
    samp = cfg.get("sampling", cfg)
    gcfg = GenConfig(k=samp.get("k", 20), temperature=samp.get("temperature", 0.2),
                     top_p=samp.get("top_p", 1.0), root_seed=samp.get("root_seed", 42))
    models = [m["repo_id"] if isinstance(m, dict) else m for m in cfg["models"]]
    subset = json.load(open(a.subset))                 # [{"id":..., "prompt":..., "gold":...}, ...]
    prompts = [q["prompt"] for q in subset]
    rawdir = os.path.join(root, "data", "raw"); os.makedirs(rawdir, exist_ok=True)
    for mi, model_id in enumerate(models):             # one model loaded at a time (fits 2x4090)
        print(f"[{mi+1}/{len(models)}] {gcfg.k} seed-aligned draws @T={gcfg.temperature} for {model_id}")
        gens = generate(prompts, model_id, a.backend, gcfg)   # [{"samples":[k], "greedy":...}]
        fp = os.path.join(rawdir, f"gen_m{mi}.jsonl")
        with open(fp, "w") as f:
            for q, g in zip(subset, gens):
                f.write(json.dumps({"id": q["id"], "model": model_id, "gold": q.get("gold"),
                                    "samples": g["samples"], "greedy": g["greedy"]},
                                   ensure_ascii=False) + "\n")
        print("   ->", fp)
    print("done. next: python scripts/03_score.py --config", a.config, "--subset", a.subset)


if __name__ == "__main__":
    main()
