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
MIN_FREE_GB="${MIN_FREE_GB:-25}"; MODELS="${MODELS:-configs/models.txt}"
mkdir -p logs data/per_model results/data reports/environment

echo "[env] recording hardware/NVIDIA/software environment -> reports/environment/"
"$PY" scripts/detect_environment.py --min-free-gb "$MIN_FREE_GB" 2>&1 | tail -4 || true

free_gb(){ df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9'; }
gpu(){ command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader || echo "no nvidia-smi"; }

echo "[subset] $BENCH x $N -> data/subset.json"
[ -f data/subset.json ] || "$PY" scripts/01_make_subset.py --benchmark "$BENCH" --n "$N" --seed "$SEED" | tee logs/01_subset.log

idx=0
while IFS='|' read -r repo quant tp; do
  repo="$(echo "$repo" | xargs)"; [ -z "$repo" ] && continue; case "$repo" in \#*) continue;; esac
  quant="$(echo "${quant:-none}" | xargs)"; tp="$(echo "${tp:-1}" | xargs)"
  qarg=(); [ "$quant" != "none" ] && qarg=(--quantization "$quant")
  echo ""; echo "=================================================================="
  echo "MODEL #$idx : $repo  (quant=$quant tp=$tp)"; echo "disk free: $(free_gb)GB | $(gpu | head -2)"
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    echo "!! NO-GO: only $(free_gb)GB free (<${MIN_FREE_GB}). Stopping safely."; exit 1; fi

  # 1-5) download + generate + score + save column + free GPU (all inside this subprocess)
  if "$PY" scripts/run_one_model.py --model "$repo" --idx "$idx" --subset data/subset.json \
        --k "$K" --tensor_parallel_size "$tp" "${qarg[@]}" 2>&1 | tee "logs/m${idx}_$(echo "$repo"|tr / _).log"; then
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
