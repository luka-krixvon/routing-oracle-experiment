"""Safely delete ONE model's weights from the Hugging Face cache to reclaim disk.

Uses the official huggingface_hub.scan_cache_dir() + delete_revisions() API — it only
removes blobs/snapshots belonging to the named repo. It NEVER does `rm -rf` on a
directory and CANNOT touch your home dir or project code. Prints freed space.

  python scripts/cleanup_hf.py --model Qwen/Qwen2.5-32B-Instruct-AWQ
  python scripts/cleanup_hf.py --model X --dry-run     # show what WOULD be freed
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id to evict from cache")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    hits = [r for r in cache.repos if r.repo_id == a.model]
    if not hits:
        print(f"[cleanup] {a.model} not in cache (nothing to free).")
        print(f"[cache] total HF cache now: {cache.size_on_disk_str}")
        return
    revs = [rev.commit_hash for r in hits for rev in r.revisions]
    freed = sum(r.size_on_disk for r in hits)
    if a.dry_run:
        print(f"[dry-run] would free {freed/1e9:.2f} GB from {a.model} ({len(revs)} revs)")
        return
    strat = cache.delete_revisions(*revs)
    print(f"[cleanup] freeing ~{strat.expected_freed_size_str} from {a.model} ...")
    strat.execute()
    print(f"[cleanup] done. HF cache now: {scan_cache_dir().size_on_disk_str}")


if __name__ == "__main__":
    main()
