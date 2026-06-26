"""Stage 01 — build the query subset (prompts + gold) for the experiment.

Default: pull a standard benchmark straight from HuggingFace `datasets` (clean
prompts + gold), write data/subset.json = [{id, prompt, gold, task}] for 02/03.

  python scripts/01_make_subset.py --benchmark gsm8k --n 200
  python scripts/01_make_subset.py --benchmark mmlu  --n 500

(RouterBench / LLMRouterBench, which ship a pre-scored matrix without a clean gold
field, are handled by src.data.load_raw_correctness for stratification-only use.)
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="gsm8k", help="gsm8k | mmlu")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200, help="subset size (pilot ~200, full ~5000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(__file__))
    out = a.out or os.path.join(root, "data", "subset.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    recs = data.load_benchmark(a.benchmark, split=a.split, n=a.n, seed=a.seed)
    json.dump(recs, open(out, "w"), ensure_ascii=False, indent=0)
    tasks = {}
    for r in recs:
        tasks[r["task"]] = tasks.get(r["task"], 0) + 1
    print(f"subset: {len(recs)} queries from {a.benchmark} -> {out}")
    print("tasks:", tasks)
    print("next: python scripts/02_generate.py --config configs/pool_open8.yaml --subset", out)


if __name__ == "__main__":
    main()
