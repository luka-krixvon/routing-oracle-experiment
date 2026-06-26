#!/usr/bin/env bash
# Release everything our run used: purge pool model weights from the HF cache + free GPU.
# Safe: uses the huggingface_hub API (never rm -rf), and only kills OUR own GPU processes
# (matched by run_one_model.py) -- it never touches your home dir, code, or other GPU work.
#
#   bash release.sh                          # purge the pool's weights + free our GPU procs
#   MODELS=configs/models.txt bash release.sh
#   ALL=1 bash release.sh                    # purge the ENTIRE HF hub cache (everything)
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
MODELS="${MODELS:-configs/models.txt}"
free_gb(){ local v; v=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9'); \
  [ -n "$v" ] || v=$(df -Pk . 2>/dev/null | awk 'NR==2{print int($4/1048576)}'); echo "${v:-0}"; }

_is_ours(){           # $1=pid -> 0 if a leftover from a routing-oracle run (anchored; never false-matches)
  [ -d "/proc/$1" ] || return 1
  tr '\0' '\n' < "/proc/$1/cmdline" 2>/dev/null | grep -qE '(^|/)run_one_model\.py$' && return 0
  tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -qE '^ROE_RUN_TOKEN=roe-' && return 0
  return 1
}

echo "[release] freeing GPU left by our run ..."
if command -v nvidia-smi >/dev/null; then
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    _is_ours "$pid" || continue
    echo "[gpu] killing leftover run pid $pid"; kill "$pid" 2>/dev/null; sleep 1
    _is_ours "$pid" && kill -9 "$pid" 2>/dev/null
  done
fi

if [ "${ALL:-0}" = "1" ]; then
  echo "[release] purging the ENTIRE HF hub cache ..."
  "$PY" scripts/cleanup_hf.py --all
else
  echo "[release] purging the pool's model weights ($MODELS) ..."
  "$PY" scripts/cleanup_hf.py --models "$MODELS"
fi

echo "[gpu] final:"; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader || echo "  (no nvidia-smi)"
echo "[disk] free now: $(free_gb)GB"
echo "[release] done — weights purged, GPU released, VM left clean."
