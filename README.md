# What Does a Routing Oracle Measure Under Stochastic Decoding?

## Coupling, Scorer Choice, and Single-Commit Ceilings

**Current manuscript and evidence: 27 September 2026.** The revised anonymous
manuscript was submitted to TMLR on this date and remains **under review, not
accepted**. This repository provides the corresponding author-named preprint
and numerical evidence. An [arXiv replacement](https://arxiv.org/abs/2607.03436)
is **being prepared**; this page does not announce a published arXiv v3.

> **Start with the dated materials below.** The top-level `src/`, `scripts/`,
> `configs/`, `tests/`, figures, and runner scripts are preserved legacy
> material. They are **not the implementation or reproduction entry point for
> the current manuscript**. The earlier README's scientific framing and
> headline claims have been superseded.

## Current materials

| File | Purpose |
|---|---|
| [Author-named preprint](releases/2026-09-27/routing-oracle-preprint.pdf) | Current manuscript, including the retrospective policy illustration and limited human check. |
| [Manuscript source](releases/2026-09-27/routing-oracle-source.zip) | LaTeX source, bibliography, and manuscript assets; **not analysis source code**. |
| [Numerical evidence](releases/2026-09-27/routing-oracle-evidence.zip) | Byte-identical copy of the anonymous supplementary evidence submitted with the revision. |
| [SHA-256 checksums](releases/2026-09-27/SHA256SUMS) | File-integrity checks for the dated artifacts. |
| [Release notes and evidence boundaries](releases/2026-09-27/README.md) | Contents, preserved-version status, verification, and limitations. |

The evidence ZIP has SHA-256
`8cc507cb7eef5b4b8559ba11f17889ee9921ac4bf3c5a42adadf05faa6c3badf`.
Its original documentation and manifests are intentionally preserved.
Preparation-time status labels and the anonymous manuscript's PDF binding
inside that ZIP describe the archived submission package, not the current
author-named PDF. Use the dated release notes and outer checksums for the
public artifacts.

## What the current paper contributes

- **A coupling-aware measurement contract.** A policy that commits to one
  model before seeing its response and a hindsight any-correct union are
  different decision classes. Marginal model success probabilities identify
  a sharp interval for the union; a product union additionally names a
  cross-model independence assumption. The contract separates the
  single-commit ceiling, coupling, scorer, model pool, response budget, and
  finite-draw estimation.
- **A fixed-archive audit.** Frozen correctness tensors cover 11 open models,
  30 archived responses per query–model cell, 500 GSM8K queries, 500 MATH-500
  queries, and 198 GPQA-Diamond queries. The audit reports every retained
  scorer channel, all 2,047 nonempty model subpools, fixed-pool checks, and
  eight frozen finite-draw paths. These are sensitivity analyses of a fixed
  archive, not confidence intervals or population estimates.
- **A retrospective held-out policy illustration.** A separate MATH-500
  300/100/100 train/tune/evaluation procedure instantiates the matched
  policy-specific decomposition. Evaluation scores were excluded from
  fitting, tuning, and prediction, but the archive had been inspected
  historically. The illustration does not establish new-data
  generalization, a routing improvement, or a deployable policy.
- **A limited reference-based human check.** Two initial judgments were
  obtained for each of 30 GSM8K and 30 GPQA-Diamond response cells. Definite
  two-person consensus covered 26/30 and 11/30 cells, respectively, and
  agreed with the frozen scorer on those subsets. The remaining 4 and 19
  cells stay undetermined. This is not archive-wide scorer validation,
  independent verification of benchmark references, or validation of other
  scorer channels.

The study does not re-evaluate external routing benchmarks' published
numbers. Its results are conditional on the declared archive, scorer,
pool, coupling, and decision class. They are not a current leaderboard,
an equal-cost routing gain, or a guarantee of attainable routing headroom.
The GSM8K and GPQA display scorers were developed after limited output
inspection; the later human check does not remove that exposure.

## What can be checked, and what is not released

The evidence contains frozen score tensors and their identities, numerical
contracts, canonical Run-A results, hash records, separate revision outputs,
and aggregate human-check material. These support independent checks of
reported score-tensor-to-result arithmetic and evidence bindings.

This is **not a complete end-to-end reproducibility release**. It does not
provide the full analysis implementation, internal release-control files,
historical raw model responses, a runtime environment, or a reusable
generation-to-scoring-to-analysis pipeline. The LaTeX source ZIP builds the
manuscript; it does not fill these implementation gaps. The preserved
top-level legacy code is not a substitute.

Individual human returns, explanations, participant identities, and consent
records are not publicly released. Only aggregate counts, documentation,
and the historical judgment rules are included; they cannot reproduce
case-level human judgments or independently establish contributor expertise
or independence.

## Legacy material

Earlier code, figures, and documentation remain available for historical
reference. Existing Git history and tags are retained. For the previous
documentation and its original context, use the
[legacy README at commit `0304028`](https://github.com/luka-krixvon/routing-oracle-experiment/blob/0304028d5ad721af96f4cc97fca10f8490126133/README.md).

Do not use that legacy README's claims, figures, or execution instructions
as documentation for the September 2026 manuscript.

## Citation

Please cite the dated preprint for the current manuscript:

> Teng-Ruei Chen. 2026. *What Does a Routing Oracle Measure Under Stochastic
> Decoding? Coupling, Scorer Choice, and Single-Commit Ceilings.*
> Preprint, 27 September 2026. Under review.

[CITATION.cff](CITATION.cff) points to this dated GitHub preprint.
The [arXiv record](https://arxiv.org/abs/2607.03436) remains the preprint-series
entry; its replacement is being prepared and is not represented here as
already published.

## Licensing

The existing [MIT license](LICENSE) covers the repository's legacy code;
it is not a blanket license for the manuscript, evidence, model weights,
or third-party benchmark content. This update does not relicense
third-party material. Applicable upstream rights and terms remain in force.
