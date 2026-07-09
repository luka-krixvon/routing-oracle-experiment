#!/usr/bin/env bash
# SEQUENTIAL code-benchmark runner — one model at a time, purge between.
# Mirrors run_sequential.sh but for HumanEval+/MBPP+ (execution-scored, pass@k).
# RUN IN A SANDBOX/CONTAINER: run_one_model_code.py executes model-generated code.
#
#   BENCH=humanevalplus K=20 bash run_sequential_code.sh
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
BENCH="${BENCH:-humanevalplus}"; K="${K:-20}"; SEED="${SEED:-42}"; MAX_TOKENS="${MAX_TOKENS:-1024}"
MODELS="${MODELS:-configs/models.txt}"
mkdir -p logs data/per_model_code

"$PY" -c "import evalplus" 2>/dev/null || { echo "!! pip install evalplus first"; exit 1; }

idx=0
while IFS='|' read -r repo quant tp mml; do
  repo="${repo%%#*}"; repo="$(echo "$repo" | xargs)"; [ -z "$repo" ] && continue
  quant="${quant%%#*}"; quant="$(echo "${quant:-none}" | xargs)"
  tp="${tp%%#*}"; tp="$(echo "${tp:-1}" | xargs)"
  mml="${mml%%#*}"; mml="$(echo "${mml:-}" | xargs)"
  qarg=(); [ "$quant" != "none" ] && qarg=(--quantization "$quant")
  mmlarg=(); [ -n "$mml" ] && mmlarg=(--max_model_len "$mml")
  echo ""; echo "=============== CODE MODEL #$idx : $repo ==============="
  if "$PY" scripts/run_one_model_code.py --model "$repo" --idx "$idx" --bench "$BENCH" \
        --k "$K" --seed "$SEED" --max_tokens "$MAX_TOKENS" --tensor_parallel_size "$tp" \
        "${qarg[@]}" "${mmlarg[@]}" 2>&1 | tee "logs/code_m${idx}_$(echo "$repo"|tr / _).log"; then
    "$PY" scripts/cleanup_hf.py --model "$repo" || true
  else
    echo "!! $repo FAILED — evicting weights, continuing."
    "$PY" scripts/cleanup_hf.py --model "$repo" || true
  fi
  idx=$((idx+1))
done < "$MODELS"

echo ""; echo "[combine] per-model columns -> tensor"
"$PY" - <<'PYEOF'
import numpy as np, glob, os, json
fs = sorted([f for f in glob.glob("data/per_model_code/m*_*.npz") if not f.endswith("_raw.npz")],
            key=lambda f: int(os.path.basename(f).split("_")[0][1:]))
cols = [np.load(f, allow_pickle=True) for f in fs]
b = np.stack([c["b_m"] for c in cols], axis=1).astype(np.int8)
models = [str(c["model"]) for c in cols]
os.makedirs("artifacts/humanevalplus", exist_ok=True)
np.savez_compressed("artifacts/humanevalplus/correctness_slim.npz",
    b=b, b_single=b[:, :, 0].astype(np.int8), greedy=b[:, :, 0].astype(np.int8),
    q_router=np.zeros(b.shape[0]), gold=cols[0]["ids"],
    meta=json.dumps({"N": b.shape[0], "M": b.shape[1], "k": b.shape[2], "models": models}))
print("wrote artifacts/humanevalplus/correctness_slim.npz", b.shape)
print("per-model p:", np.round(b.mean(axis=(0, 2)), 3))
PYEOF
echo "ALL DONE. Send back: artifacts/humanevalplus/correctness_slim.npz (+ logs/)"
