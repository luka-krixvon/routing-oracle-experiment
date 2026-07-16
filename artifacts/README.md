# Paper artifacts — reproduce the tables from committed data

These small files (≈250 KB total) reproduce the paper's headline decomposition,
pool-composition, and robustness tables **without a GPU and without re-generating
any model output**. The heavy generation step (11 models × N=500 × k=30 draws at
T=0.2) is upstream; here we ship its distilled outputs.

## Layout

```
artifacts/
  math500/
    correctness_slim.npz   # b (500,11,30) int8 + b_single, greedy, q_router, gold, ids, meta
                           #   (raw generated text `Y` is omitted to keep the file small;
                           #    it is only needed to recompute the majority-vote O^agg)
    decomposition.json     # O^exp/O^repro/router, G=G_rec+G_noise, noise share + bootstrap CI,
                           #   gates, best-of-K check, thin/dense strata, pool_definitions, family_correlation
    family_correlation.csv # 11×11 lineage-clustered error-correlation matrix
    robustness.json        # k-sweep, K-sweep (oracle S(K) vs gap share), leave-one-lineage-out
                           #   jackknife, tau reliability, and the learned kNN-router baseline
    subset.json            # the 500 MATH-500 prompts (id, prompt, gold, task)
  gsm8k/
    correctness_slim.npz   # same schema, GSM8K run
    decomposition.json
    family_correlation.csv
  gpqa/
    correctness_slim.npz   # b (198,11,30) int8 + b_single, greedy, q_router, meta
                           #   NOTE: no `gold` field — see "Licensing & GPQA" below
    subset_ids.json        # ids + SHA-256 hashes only (no question text);
                           #   rebuild plaintext with scripts/rebuild_gpqa_subset.py
    decomposition.json / family_correlation.csv / robustness.json
  humanevalplus/
    correctness_slim.npz   # HumanEval+ code run (EvalPlus base+extra tests)
    b_base.npz             # base-test-only correctness (the deployable verifier signal)
    subset.json            # the 164 HumanEval+ tasks (Apache-2.0; redistributable)
  environment_report.json  # auto-captured hardware/software stack (Table VI)
```

> GSM8K ships its `decomposition.json` + correlation only; its raw correctness
> tensor and robustness suite can be regenerated with the same pipeline (the
> MATH-500 tensor is included as the worked, unsaturated example).

## Licensing & GPQA redistribution

| Benchmark | Source | License | Redistributed here |
|---|---|---|---|
| GSM8K | `openai/gsm8k` | MIT | prompts + gold (subset.json) |
| MATH-500 | `HuggingFaceH4/MATH-500` | MIT | prompts + gold (subset.json) |
| HumanEval+ | EvalPlus | Apache-2.0 | tasks (subset.json) |
| GPQA-Diamond | `Idavidrein/gpqa` (gated) | CC-BY-4.0 | **hashes only** (subset_ids.json) |

GPQA's authors ask that questions and answers never be posted online in plain
text, to keep the benchmark out of future training corpora. We honor that:
this repo ships only correctness bits, query ids, and SHA-256 hashes for GPQA.
To rebuild the plaintext subset locally (requires accepting the dataset terms
on Hugging Face):

```bash
python scripts/rebuild_gpqa_subset.py        # writes artifacts/gpqa/subset.local.json
                                             # + correctness_slim.local.npz (gold restored)
```

The rebuild is deterministic (seed=42, per-query seed-derived option shuffle)
and hash-verified against `subset_ids.json`, so your local copy provably
matches the one used in the paper. `*.local.*` files are gitignored — never
commit them.

## Reproduce the decomposition (CPU, seconds)

```bash
python - <<'PY'
import numpy as np
z = np.load("artifacts/math500/correctness_slim.npz", allow_pickle=True)
b = z["b"].astype(float)                      # (500, 11, 30)
p_hat  = b.sum(2) / b.shape[2]
O_exp  = b.max(1).mean(1)                      # seed-aligned expected oracle
O_repro= p_hat.max(1)                          # reproducible ceiling
router = p_hat[:, int(np.argmax(p_hat.mean(0)))]  # best single model
G, noise = (O_exp - router).mean(), (O_exp - O_repro).mean()
print(f"O^exp={O_exp.mean():.3f}  O^repro={O_repro.mean():.3f}  gap={G:.3f}  noise_share={noise/G:.3f}")
# -> O^exp=0.873  O^repro=0.837  gap=0.100  noise_share=0.361   (Table V, MATH-500)
PY
```

## Reproduce the robustness suite

```bash
python scripts/06_robustness.py \
  --data artifacts/math500/correctness_slim.npz \
  --subset artifacts/math500/subset.json \
  --per_model_glob 'artifacts/math500/correctness_slim.npz' \
  --out artifacts/math500/robustness.json
```

(`06_robustness.py` reads `ids` from the per-model glob; the slim tensor carries
`ids`, so pointing the glob at it works.)
