"""Rebuild the GPQA-Diamond subset (plaintext) locally from the gated source.

GPQA's authors ask that questions and answers never be posted online in plain
text, to keep the benchmark out of future training corpora. This repo therefore
ships only `artifacts/gpqa/subset_ids.json` (ids + SHA-256 hashes) and a
correctness tensor with the `gold` field stripped. Run this script to rebuild
the plaintext subset on your own machine:

  1. Accept the dataset terms at https://huggingface.co/datasets/Idavidrein/gpqa
  2. `huggingface-cli login`  (or set HF_TOKEN)
  3. python scripts/rebuild_gpqa_subset.py

It re-runs the exact deterministic pipeline (seed=42, per-query seed-derived
option shuffle — see src/data.py) and verifies every prompt/gold against the
committed hashes, then writes:

  artifacts/gpqa/subset.local.json           [{id, prompt, gold, task}]
  artifacts/gpqa/correctness_slim.local.npz  slim tensor with `gold` restored

Both outputs are gitignored; never commit them.

  --verify-only   only hash-check an existing subset.local.json (no HF access)
"""
import argparse, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPQA_DIR = os.path.join(ROOT, "artifacts", "gpqa")
IDS_PATH = os.path.join(GPQA_DIR, "subset_ids.json")
LOCAL_JSON = os.path.join(GPQA_DIR, "subset.local.json")
SLIM_NPZ = os.path.join(GPQA_DIR, "correctness_slim.npz")
LOCAL_NPZ = os.path.join(GPQA_DIR, "correctness_slim.local.npz")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def verify(recs, ids):
    assert len(recs) == len(ids), f"count mismatch: {len(recs)} vs {len(ids)}"
    bad = []
    for r, ref in zip(recs, ids):
        if r["id"] != ref["id"]:
            bad.append((ref["id"], "id order"))
        elif _sha(r["prompt"]) != ref["prompt_sha256"]:
            bad.append((ref["id"], "prompt hash"))
        elif _sha(r["id"] + "|" + r["gold"])[:16] != ref["gold_check"]:
            bad.append((ref["id"], "gold hash"))
    if bad:
        for qid, what in bad[:10]:
            print(f"  MISMATCH {qid}: {what}", file=sys.stderr)
        sys.exit(f"verification FAILED for {len(bad)}/{len(ids)} records")
    print(f"verified {len(ids)}/{len(ids)} records against subset_ids.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="hash-check existing subset.local.json without touching HF")
    a = ap.parse_args()

    ids = json.load(open(IDS_PATH))

    if a.verify_only:
        recs = json.load(open(LOCAL_JSON))
    else:
        from src import data  # heavy import (datasets); needs gated-HF access
        recs = data.load_benchmark("gpqa", split="train", n=198, seed=42)
    verify(recs, ids)

    if a.verify_only:
        return

    json.dump(recs, open(LOCAL_JSON, "w"), ensure_ascii=False, indent=0)
    print("wrote", LOCAL_JSON)

    if os.path.exists(SLIM_NPZ):
        import numpy as np
        z = dict(np.load(SLIM_NPZ, allow_pickle=True))
        z["gold"] = np.array([r["gold"] for r in recs], dtype=object)
        np.savez_compressed(LOCAL_NPZ, **z)
        print("wrote", LOCAL_NPZ, "(slim tensor + gold restored)")


if __name__ == "__main__":
    main()
