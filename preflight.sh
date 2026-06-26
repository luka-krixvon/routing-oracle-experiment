#!/usr/bin/env bash
# PREFLIGHT — run this on the GPU server BEFORE any experiment. Read-only: it only
# checks disk / GPU / HF cache / project size / deps / per-model reachability and
# prints a GO / NO-GO. It does NOT download models or run anything.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"     # require this much free before starting

echo "================ DISK (df -h) ================"; df -h . / 2>/dev/null | sort -u
AVAIL_GB=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
echo "free on this filesystem: ${AVAIL_GB:-?} GB   (threshold ${MIN_FREE_GB} GB)"

echo "================ GPU (nvidia-smi) ============"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
else echo "!! nvidia-smi not found — no GPU? (smoke mode only)"; fi

echo "================ HF cache size ==============="
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
du -sh "$HF_HOME" 2>/dev/null || echo "(no HF cache yet at $HF_HOME)"
$PY - <<'PY' 2>/dev/null || true
from huggingface_hub import scan_cache_dir
try:
    c=scan_cache_dir(); print("cached repos:")
    for r in sorted(c.repos,key=lambda r:-r.size_on_disk):
        print(f"   {r.size_on_disk/1e9:6.2f} GB  {r.repo_id}")
    print(f"   total: {c.size_on_disk_str}")
except Exception as e: print("(scan_cache_dir:",e,")")
PY

echo "================ project size ================"; du -sh . 2>/dev/null
echo "  data/:    $(du -sh data 2>/dev/null | cut -f1 || echo -)"
echo "  results/: $(du -sh results 2>/dev/null | cut -f1 || echo -)"

echo "================ python deps ================="
$PY -c "import sys;print('python',sys.version.split()[0])"
$PY -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| gpus',torch.cuda.device_count())" 2>/dev/null || echo "!! torch missing"
$PY -c "import vllm;print('vllm',vllm.__version__)" 2>/dev/null || echo "!! vllm missing — pip install -r requirements.txt"

echo "================ model reachability =========="
$PY - <<'PY'
import os
from huggingface_hub import HfApi
api=HfApi()
ml="configs/models.txt"
if not os.path.exists(ml): print("!! configs/models.txt missing"); raise SystemExit
ok=bad=0
for ln in open(ml):
    ln=ln.strip()
    if not ln or ln.startswith("#"): continue
    repo=ln.split("|")[0].strip()
    try:
        info=api.model_info(repo)
        gated=getattr(info,"gated",False)
        sz=None
        try: sz=sum((s.size or 0) for s in (info.siblings or []))/1e9
        except Exception: pass
        flag="GATED(need login+accept)" if gated else "open"
        print(f"  OK   {repo:50s} {flag}" + (f"  ~{sz:.1f}GB(all files)" if sz else ""))
        ok+=1
    except Exception as e:
        print(f"  FAIL {repo:50s} {type(e).__name__}: {e}"); bad+=1
print(f"\n  reachable: {ok}  failed: {bad}")
PY

echo "================ VERDICT ====================="
if [ "${AVAIL_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "NO-GO: only ${AVAIL_GB}GB free (<${MIN_FREE_GB}GB). Free space or shrink pool/quant first."
else
  echo "Disk OK (${AVAIL_GB}GB >= ${MIN_FREE_GB}GB). Fix any FAIL/GATED above, then: bash run_sequential.sh"
fi
echo "(gated models: run 'huggingface-cli login' and accept the license on the model page first)"
