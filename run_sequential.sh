#!/usr/bin/env bash
# SEQUENTIAL runner — disk-safe, one model at a time, purge between.
# For each model: disk guard -> generate+score in a SUBPROCESS (VRAM freed on exit)
# -> evict its weights from the HF cache -> re-check disk/GPU -> next. Then combine
# the tiny per-model columns and run the decomposition. Resumable (skips done models).
#
#   bash run_sequential.sh                 # benchmark=gsm8k, N=200 (edit below)
#   BENCH=mmlu N=1000 K=30 bash run_sequential.sh
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
BENCH="${BENCH:-gsm8k}"; N="${N:-200}"; K="${K:-20}"; SEED="${SEED:-42}"
MIN_FREE_GB="${MIN_FREE_GB:-35}"; MODELS="${MODELS:-configs/models.txt}"   # >= largest model (phi-4 ~29GB) + margin
mkdir -p logs data/per_model results/data reports/environment

echo "[env] recording hardware/NVIDIA/software environment -> reports/environment/"
"$PY" scripts/detect_environment.py --min-free-gb "$MIN_FREE_GB" 2>&1 | tail -4 || true

free_gb(){ local v; v=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9'); \
  [ -n "$v" ] || v=$(df -Pk . 2>/dev/null | awk 'NR==2{print int($4/1048576)}'); echo "${v:-0}"; }
gpu(){ command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader || echo "no nvidia-smi"; }

PURGE="${PURGE:-1}"   # 1 = at the very end, purge ALL pool weights + free GPU so the VM is left clean (0 = keep)
RUN_TOKEN="roe-$$"    # unique tag for THIS run; exported to each model so we can find+free only our procs

_is_ours(){           # $1=pid -> 0 if it's one of OUR run's GPU processes (anchored, can't false-match)
  [ -d "/proc/$1" ] || return 1
  tr '\0' '\n' < "/proc/$1/cmdline" 2>/dev/null | grep -qE '(^|/)run_one_model\.py$' && return 0
  tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -qE '^ROE_RUN_TOKEN=roe-' && return 0   # tp>1 vLLM workers inherit it
  return 1
}

free_gpu_orphans(){
  command -v nvidia-smi >/dev/null || return 0
  # auto-kill ONLY our own leftover GPU procs (argv token run_one_model.py, or the inherited
  # ROE_RUN_TOKEN=roe-* env tag matched line-anchored). Re-verify identity before SIGKILL so a
  # recycled PID can't be hit. Never matches JUPYTER_*, ROE_RUN=10, or anyone else's vllm.
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    _is_ours "$pid" || continue
    echo "[gpu] killing leftover run pid $pid"; kill "$pid" 2>/dev/null; sleep 1
    _is_ours "$pid" && kill -9 "$pid" 2>/dev/null
  done
  # report (do NOT kill) anything else still resident -- could be your other work
  rest=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  [ -n "$rest" ] && { echo "[gpu] other GPU processes still resident (NOT matched as ours, left untouched):"; \
    echo "$rest"; echo "[gpu] if any are leftovers from this run, free them with:  kill -9 <PID>"; }
}

teardown(){
  [ -n "${_TORN:-}" ] && return; _TORN=1            # run once (a signal exit then EXIT both fire)
  echo ""; echo "=============== TEARDOWN: leaving the VM clean ==============="
  free_gpu_orphans
  if [ "$PURGE" = "1" ]; then
    echo "[purge] removing ALL pool model weights from the HF cache (safe API, never rm -rf) ..."
    "$PY" scripts/cleanup_hf.py --models "$MODELS" || true
  else
    echo "[purge] PURGE=0 -> skipping the FINAL sweep (note: each model is still evicted right after it runs)"
  fi
  echo "[gpu] final:"; gpu; echo "[disk] free now: $(free_gb)GB"
  echo "============================================================="
}
_on_signal(){ echo ""; echo "[abort] interrupt received -> stopping and tearing down."; exit 130; }
trap _on_signal INT TERM   # Ctrl-C / kill actually STOPS the loop (then EXIT runs teardown once)...
trap teardown EXIT         # ...and teardown always runs exactly once on the way out (normal / abort / disk-guard)

echo "[subset] $BENCH x $N -> data/subset.json"
[ -f data/subset.json ] || "$PY" scripts/01_make_subset.py --benchmark "$BENCH" --n "$N" --seed "$SEED" | tee logs/01_subset.log

idx=0
# fields: repo | quant | tensor_parallel | [max_model_len]   (inline # comments are stripped)
while IFS='|' read -r repo quant tp mml; do
  repo="${repo%%#*}"; repo="$(echo "$repo" | xargs)"; [ -z "$repo" ] && continue   # skip blank/comment lines
  quant="${quant%%#*}"; quant="$(echo "${quant:-none}" | xargs)"
  tp="${tp%%#*}"; tp="$(echo "${tp:-1}" | xargs)"
  mml="${mml%%#*}"; mml="$(echo "${mml:-}" | xargs)"
  qarg=(); [ "$quant" != "none" ] && qarg=(--quantization "$quant")
  mmlarg=(); [ -n "$mml" ] && mmlarg=(--max_model_len "$mml")
  echo ""; echo "=================================================================="
  echo "MODEL #$idx : $repo  (quant=$quant tp=$tp${mml:+ max_model_len=$mml})"; echo "disk free: $(free_gb)GB | $(gpu | head -2)"
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    echo "!! NO-GO: only $(free_gb)GB free (<${MIN_FREE_GB}). Stopping safely."; exit 1; fi

  # 1-5) download + generate + score + save column + free GPU (all inside this subprocess).
  # ROE_RUN_TOKEN tags this process and every vLLM worker it spawns, so teardown can find+free them.
  if ROE_RUN_TOKEN="$RUN_TOKEN" "$PY" scripts/run_one_model.py --model "$repo" --idx "$idx" --subset data/subset.json \
        --k "$K" --seed "$SEED" --tensor_parallel_size "$tp" "${qarg[@]}" "${mmlarg[@]}" 2>&1 | tee "logs/m${idx}_$(echo "$repo"|tr / _).log"; then
    # 6-8) evict this model's weights from the HF cache, re-check
    "$PY" scripts/cleanup_hf.py --model "$repo" 2>&1 | tee -a "logs/m${idx}_$(echo "$repo"|tr / _).log"
    echo "after cleanup -> disk free: $(free_gb)GB | $(gpu | head -2)"
  else
    echo "!! model $repo FAILED — keeping its log, evicting weights, continuing to next."
    "$PY" scripts/cleanup_hf.py --model "$repo" || true
  fi
  idx=$((idx+1))
done < "$MODELS"

echo ""; echo "[combine] per-model columns -> correctness tensor"
"$PY" scripts/combine.py --subset data/subset.json | tee logs/combine.log
echo "[decompose] corrected oracles + gates + best-of-K"
"$PY" scripts/04_oracles_decompose.py --npz data/processed/correctness_kxN.npz --outdir results/data | tee logs/04_decompose.log
echo "[env] final environment snapshot -> reports/environment_final/"
"$PY" scripts/detect_environment.py --outdir reports/environment_final --min-free-gb "$MIN_FREE_GB" 2>&1 | tail -3 || true
echo ""; echo "ALL DONE. Send back:  results/   logs/   reports/   data/per_model/*.npz  (all small)."
echo "Final disk free: $(free_gb)GB"
