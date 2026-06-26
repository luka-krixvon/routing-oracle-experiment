"""Safely delete model weights from the Hugging Face cache to reclaim disk.

Uses the official huggingface_hub.scan_cache_dir() + delete_revisions() API — it only
removes blobs/snapshots belonging to the named repo(s). It NEVER does `rm -rf` on a
directory and CANNOT touch your home dir or project code. Prints freed space.

  python scripts/cleanup_hf.py --model Qwen/Qwen2.5-32B-Instruct-AWQ   # one repo
  python scripts/cleanup_hf.py --models configs/models.txt             # every repo in the pool
  python scripts/cleanup_hf.py --all                                   # entire HF hub cache
  python scripts/cleanup_hf.py --models configs/models.txt --dry-run   # show, don't delete
"""
import argparse


def repos_from_file(path):
    """Read repo ids from a models.txt-format file (`repo | quant | tp`, # comments)."""
    out = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        repo = line.split("|", 1)[0].strip()
        if repo:
            out.append(repo)
    return out


def purge(repos, dry):
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    present = {r.repo_id: r for r in cache.repos}
    revs, freed, named = [], 0, []
    for repo in repos:
        r = present.get(repo)
        if not r:
            print(f"[cleanup] {repo} not in cache (nothing to free).")
            continue
        revs += [rev.commit_hash for rev in r.revisions]
        freed += r.size_on_disk
        named.append(repo)
    if not revs:
        print(f"[cache] nothing to free; total HF cache now: {cache.size_on_disk_str}")
        return
    if dry:
        print(f"[dry-run] would free ~{freed/1e9:.2f} GB from {len(named)} repo(s): {named}")
        return
    strat = cache.delete_revisions(*revs)
    print(f"[cleanup] freeing ~{strat.expected_freed_size_str} from {len(named)} repo(s) ...")
    strat.execute()
    print(f"[cleanup] done. HF cache now: {scan_cache_dir().size_on_disk_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="single HF repo id to evict")
    ap.add_argument("--models", help="models.txt-format file: evict every repo listed")
    ap.add_argument("--all", action="store_true", help="evict the ENTIRE HF hub cache")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.all:
        from huggingface_hub import scan_cache_dir
        repos = [r.repo_id for r in scan_cache_dir().repos]
    elif a.models:
        repos = repos_from_file(a.models)
    elif a.model:
        repos = [a.model]
    else:
        ap.error("give one of --model REPO | --models FILE | --all")
    purge(repos, a.dry_run)


if __name__ == "__main__":
    main()
