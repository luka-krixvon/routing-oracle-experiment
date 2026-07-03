# How Much of the Routing Gap Is Real?

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Serving](https://img.shields.io/badge/serving-vLLM-orange.svg)
![Hardware](https://img.shields.io/badge/hardware-2%C3%97%20RTX%204090-lightgrey.svg)
![No API key](https://img.shields.io/badge/API%20key-not%20required-brightgreen.svg)

> Reproducibility companion for the paper *“How Much of the Routing Gap Is Real?
> Decomposing the Router-to-Oracle Gap into Reproducible Specialist Advantage and
> Single-Draw Label Noise.”* **Open-weight, local, no API key** — every model is
> served with `vLLM` at a fixed temperature; no closed endpoint, no LLM API.
>
> **What is proved vs. what is measured.** The paper's claims are *theorems* —
> they hold analytically and do **not** depend on any experiment. This code does
> one thing: it **measures the empirical magnitude** those theorems predict on a
> real open-weight pool, and runs the two assumption **gates** and the single
> **falsifiable** check. If a gate fails, it reports *no* magnitude — by design.

<p align="center">
  <img src="figures/fig_example.png" width="60%" alt="Worked example">
</p>

*Hand-checkable intuition: with 10 models each correct with probability `p = 0.1`,
the single-draw oracle reaches `1 − 0.9¹⁰ ≈ 65%` — yet no model is reliable and a
router that commits to one model can reach only `10%`. Of the apparent `65%`, only
the `10%` reproducible mass (grey) is reachable by routing; the remaining `55%`
(blue) is single-draw label noise no router can capture — only test-time
resampling can.*

---

## TL;DR

LLM-routing benchmarks report a large gap between deployed routers and a
**per-instance oracle** — the model that is best on each query, in hindsight. But
that oracle is computed from **one** correctness label per (query, model), and
under stochastic decoding (`T > 0`) that label is a single Bernoulli draw. So the
oracle is a *maximum over noisy single draws* and inherits the upward bias that
inflates best-of-`N` estimates. We re-estimate per-(query, model) correctness as a
**probability** from `k ≥ 20` fresh draws, and split the router-to-oracle gap
exactly into

```
G  =  G_rec           +  G_noise
      recoverable         single-draw label noise
      specialist          (no single-commit router can
      advantage           recover it — only resampling)
```

The headline: single-draw noise is a **substantial minority** of the gap — larger
on an unsaturated benchmark, approaching half on the hardest queries — while the
**majority is genuine, recoverable specialist advantage**. And the noise obeys a
**recoverability asymmetry**: it is *uncapturable by selection* (any single-commit
router is capped at the reproducible ceiling `O^repro`) yet *recovered by test-time
sampling* on the committed model.

| Benchmark | router→oracle gap | **single-draw noise share** | recoverable |
|---|---:|---:|---:|
| **GSM8K** (saturated)      | 3.3 pts  | **12 %**  `[6, 19]`  | 88 % |
| **MATH-500** (unsaturated) | 10.0 pts | **36 %**  `[31, 42]` | 64 % |

Evidence: **11 open-weight models, 8 distinct pretraining lineages**, `k = 30`
seed-aligned draws per cell at `T = 0.2`, on two exact-match benchmarks
(`N = 500` stratified each), with both assumption gates passing and the
best-of-`K` recoverability check passing.

---

## Result 1 — The gap is majority-recoverable; noise is a real minority

<p align="center"><img src="figures/fig_results_decomp.png" width="52%" alt="Gap decomposition"></p>

*The router-to-oracle gap (accuracy points) splits into recoverable specialist
advantage `G_rec` (grey) and single-draw label noise `G_noise` (blue). Noise is the
minority slice on both benchmarks — larger in absolute terms on MATH-500 but still
a minority of the gap.*

|                                   | GSM8K (sat.) | MATH-500 (unsat.) |
|---|---:|---:|
| Recorded oracle `O^exp`           | 0.993 | 0.873 |
| Reproducible ceiling `O^repro`    | 0.989 | 0.837 |
| Best-single router                | 0.960 | 0.773 |
| Gap `G`                           | 0.033 | 0.100 |
| &nbsp;&nbsp;recoverable `G_rec`   | 0.029 | 0.064 |
| &nbsp;&nbsp;noise `G_noise`       | 0.004 | 0.036 |
| **Noise share** `G_noise / G`     | **12 %** `[6, 19]` | **36 %** `[31, 42]` |

The noise term is statistically real on both (nested-bootstrap lower bound `> 0`).
Its router-independent floor `O^exp − O^repro` grows from 0.4 to 3.6 points as the
pool goes from saturated to unsaturated.

## Result 2 — The noise concentrates exactly where no model is reliable

<p align="center"><img src="figures/fig_results_thin.png" width="52%" alt="Noise share by support stratum"></p>

*Single-draw noise share by support stratum. Where no model is reliable
(thin support, ≤ 3 of 11 correct) the share is highest — **43 % on MATH-500**. But
MATH-500 is noise-heavier chiefly because it places far more queries there
(28 % vs 3 % of queries), a ~9× larger no-model-reliable population — the
`Θ(1 − p̄)` mechanism the theory predicts. Even in the worst stratum the share
stays below one half.*

## Result 3 — Correlated pools inflate the apparent gap

<p align="center"><img src="figures/fig_results_pool.png" width="52%" alt="Noise share by pool composition"></p>

*Redundancy depresses the recoverable component while leaving decoding variance
untouched, so the noise share is highest for the most homogeneous pool and lowest
for the most decorrelated one — the direction the theory predicts.*

| Pool | `K` | GSM8K | MATH-500 |
|---|---:|---:|---:|
| Full                           | 11 | 12 % | 36 % |
| One-per-lineage (diverse)      | 8  | 9 %  | 34 % |
| Qwen size-sweep (homogeneous)  | 3  | 24 % | 44 % |

Cross-model error correlation is high (within-lineage exceeds cross-lineage), so
the **effective pool size is only 2.1–3.5 of 11**: eleven models behave like two to
three independent ones. The one-per-lineage pool is the conservative headline.

## Recoverability check (the single falsifiable prediction)

The prediction that test-time **sampling** recovers what **selection** cannot
passes on both benchmarks: best-of-`K` on the per-query committed best model
reaches **0.946** on MATH-500, exceeding the matched-budget independent-pool oracle
`O^exp,⊥ = 0.874`. Recovery *scope* is verifier-gated: a single-round majority vote
reaches only **0.824** — below the reproducible ceiling **0.837** — so the
sampling-recoverable mass is dominated by the guessing residual, reclaimable only
with a deploy-time verifier, not by aggregation alone.

## Why it happens

<p align="center"><img src="figures/fig_asymmetry.png" width="72%" alt="Recoverability asymmetry"></p>

*The recoverability asymmetry. On the **selection** axis every single-commit
router — deterministic, randomized, or any mixture — is capped at `O^repro`; the
hatched mass above it is unreachable by choosing a model in advance, and this cap
needs no cross-model independence. On the **sampling** axis the same per-query draw
budget, spent as best-of-`K` on the committed model, provably lifts performance
through the verifier-free aggregation ceiling toward `O^exp`.*

---

## Repository layout

```
routing-oracle-experiment/
├── src/                          # library (imported by the scripts)
│   ├── oracles.py                    # O^exp / O^repro / O^agg + Fréchet bracket
│   ├── decompose.py                  # exact gap decomposition  G = G_rec + G_noise
│   ├── generate.py                   # vLLM generation; seed-aligned draws (A7)
│   ├── score.py                      # exact-match scoring
│   ├── stats.py                      # bootstrap CIs, winner's-curse radius, tests
│   ├── simulate.py                   # known-p simulation (smoke run + KS gate)
│   └── data.py                       # dataset loaders
├── scripts/                      # pipeline stages (run from the repo root)
│   ├── 01_make_subset.py             # stratified query subset (oversamples thin support)
│   ├── 02_generate.py                # k seed-aligned draws @ T=0.2  (or --simulate, no GPU)
│   ├── 03_score.py                   # → correctness tensor  (N, M, k)
│   ├── 04_oracles_decompose.py       # oracles + gates + best-of-K + decomposition
│   ├── run_one_model.py              # disk-safe unit: download → generate → score → evict
│   ├── cleanup_hf.py                 # evict weights from the HF cache (safe hub API)
│   ├── combine.py · rescore.py       # assemble / re-score per-model columns
│   └── detect_environment.py         # hardware / CUDA / version snapshot for the paper
├── configs/
│   ├── models.txt                    # the 11-model open-weight pool (edit here)
│   └── pool_open8.yaml               # sampling / gate / estimator parameters
├── figures/                      # the paper figures shown in this README
├── run_sequential.sh             # disk-safe end-to-end runner (one model at a time)
├── run_all.sh                    # simple chain · --smoke (no-GPU pipeline validation)
├── preflight.sh                  # GO/NO-GO checks (disk, GPU, model reachability)
└── tests/test_oracles.py
```

## Install

```bash
pip install -r requirements.txt          # install vLLM first; it pulls a matching torch
```

## Reproduce (fast — no GPU, ~30 s)

Validates the whole pipeline on simulated correctness:

```bash
bash run_all.sh --smoke                  # → results/mvp/decomposition.json
```

## Reproduce the results from scratch (GPU)

```bash
bash preflight.sh                        # checks only; prints GO / NO-GO
bash run_sequential.sh                   # disk-safe, one model at a time
BENCH=gsm8k N=1000 K=30 bash run_sequential.sh   # fuller run
```

`run_sequential.sh` processes **one model at a time** — download → `k` seed-aligned
draws → exact-match score → save the tiny per-model column → **evict the weights**
(safe hub API, never `rm -rf`) → free the GPU → next. Peak disk ≈ one model, and it
is **resumable**. It ends by assembling the `(N, M, k)` tensor and running
`04_oracles_decompose.py` (corrected oracles + gates + best-of-`K` + decomposition
→ `results/data/decomposition.json`). `detect_environment.py` records the exact
hardware/CUDA/library stack (masking secrets) for the paper's system-configuration
appendix.

## The model pool & data

**Pool** (`configs/models.txt`, edit freely): 11 open-weight **text-only**
instruction models across **8 pretraining lineages** — Mistral-7B,
DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-{7,14,32}B, Phi-4, OLMo-2-7B, Yi-1.5-9B,
Granite-3.3-8B, Gemma-2-9B, Llama-3.1-8B. Text-only by design: the one measured
noise source must be decoding stochasticity, not modality / answer-extraction
mismatch. OLMo-2, Yi-1.5 and Granite were added so no single lineage dominates.

**Benchmarks**: GSM8K (saturated) and MATH-500 (unsaturated), exact-match, clean
gold. Provider caching / per-draw dependence is guarded by an independence gate;
the known-`p` simulation gate checks the #-correct-models distribution before any
magnitude is reported.

## What is *not* in this repository (by design)

- **LLMRouterBench's per-cell correctness matrix.** It is the paper's *primary
  audit target* (33 models, 391,645 instances; its ~20-point oracle gap is, by
  construction, a union of single `T = 0.2` draws), but its raw matrix is not
  publicly released, and `O^repro` is non-identifiable from a `k = 1` matrix. So
  this repository runs a **controlled open-pool re-generation** on standard
  benchmarks instead; connect LLMRouterBench via `src/data.load_raw_correctness`
  when its data becomes available.
- **Closed-model endpoints.** Recovering `O^exp` and `O^repro` needs *new,
  seed-pinned* per-cell generation at a fixed temperature, which closed APIs
  neither guarantee nor expose — hence open weights, and **no API key**.
- **Vision-language models.** Admitting them on text benchmarks would make input
  modality and answer extraction heterogeneous and inject non-decoding variance
  outside the seed-aligned model.

## Citation

See [`CITATION.cff`](CITATION.cff). The arXiv identifier will be added once the
preprint is posted.

## License

[MIT](LICENSE). © 2026 Teng-Ruei Chen.
