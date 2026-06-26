#!/usr/bin/env bash
# One-command pipeline for the routing-oracle experiment.
#   bash run_all.sh [benchmark] [N]      # real run (needs GPU + vLLM)
#   bash run_all.sh --smoke              # no-GPU validation (simulated correctness)
# Outputs results/<mode>/decomposition.json  (+ mvp_decomposition.png for --smoke).
set -euo pipefail
cd "$(dirname "$0")"
CFG=configs/pool_open8.yaml
PY="${PYTHON:-$(command -v python3 || command -v python)}"
[ -n "$PY" ] || { echo "no python found; set PYTHON=/path/to/python"; exit 1; }

if [ "${1:-}" = "--smoke" ]; then
  echo "[smoke] no-GPU pipeline validation on simulated correctness ($PY)"
  "$PY" scripts/02_generate.py --simulate --N 200 --M 5 --k 10 --scenario thin
  "$PY" scripts/04_oracles_decompose.py --npz data/processed/correctness_kxN.npz --outdir results/mvp
  echo "DONE (smoke) -> results/mvp/decomposition.json"
  exit 0
fi

BENCH="${1:-gsm8k}"; N="${2:-200}"
echo "[1/4] build subset: $BENCH x $N"
"$PY" scripts/01_make_subset.py --benchmark "$BENCH" --n "$N" --seed 42
echo "[2/4] generate k seed-aligned draws @T=0.2 for the 8-model pool (vLLM)"
"$PY" scripts/02_generate.py --config "$CFG" --subset data/subset.json
echo "[3/4] score -> correctness tensor"
"$PY" scripts/03_score.py --subset data/subset.json
echo "[4/4] corrected oracles + gates + best-of-K + decomposition"
"$PY" scripts/04_oracles_decompose.py --npz data/processed/correctness_kxN.npz --outdir results/data
echo "DONE -> results/data/decomposition.json   (send this file + results/data/*.png back)"
