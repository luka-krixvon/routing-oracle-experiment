# Dated manuscript and evidence — 27 September 2026

**Paper:** *What Does a Routing Oracle Measure Under Stochastic Decoding?
Coupling, Scorer Choice, and Single-Commit Ceilings*  
**Author:** Teng-Ruei Chen

The anonymous revision was submitted to TMLR on 27 September 2026.
It remains **under review, not accepted**. The PDF here is the corresponding
author-named preprint, using the earlier arXiv preprint layout.
The [arXiv replacement](https://arxiv.org/abs/2607.03436) has been **submitted
but not yet publicly announced by arXiv**; these materials do not claim
that arXiv v3 is publicly available.

## Files

| File | Contents and scope |
|---|---|
| [routing-oracle-preprint.pdf](routing-oracle-preprint.pdf) | Current author-named manuscript with appendices. |
| [routing-oracle-source.zip](routing-oracle-source.zip) | LaTeX manuscript source, bibliography, and assets. This is **not the analysis implementation**. |
| [routing-oracle-evidence.zip](routing-oracle-evidence.zip) | Byte-identical copy of the anonymous supplementary ZIP submitted with the revision. |
| [SHA256SUMS](SHA256SUMS) | SHA-256 digests of the three files above. |

The evidence ZIP's SHA-256 is:

```text
8cc507cb7eef5b4b8559ba11f17889ee9921ac4bf3c5a42adadf05faa6c3badf
```

From this directory, verify the downloaded files with:

```sh
shasum -a 256 -c SHA256SUMS
```

Hashes establish file identity, not scientific validity or completeness.

## Reading the evidence

The outer evidence ZIP preserves these components:

- `original/routing-oracle-supplement.zip`: original frozen correctness
  tensors and ordered identities, numerical contract, canonical Run-A
  outputs, analysis receipt, and member digests.
- `revision/routing-oracle-revision-addendum-20260922.zip`: separate
  querywise/descriptive outputs and the historical-archive policy
  illustration, with their own evidence bindings.
- `human-audit/aggregate_summary.json`, `human-audit/README.md`, and
  `human-audit/RUBRIC.md`: aggregate human-check results, their definitions
  and limits, and the historical judgment rules.
- `revision_manifest.json`: roles, sizes, and hashes of the preserved
  members.

Extract the two inner ZIPs into **separate directories**; do not overlay
their contents. Each inner manifest uses its own original relative paths.
The original Run-A record's absence of policy outputs does not describe the
later policy illustration in the revision addendum.

### Preserved documentation and status

The submitted evidence has not been rewritten for this public release.
Some internal documentation still calls the package a local candidate,
refers to the anonymous manuscript's pagination and hash, or says the
human material has not yet been uploaded. Those are preserved
pre-submission statements, not the current release status.

The manuscript was submitted on 27 September 2026, and the evidence ZIP
here is byte-identical to the supplementary file submitted with it.
Internal PDF hashes bind the anonymous submission manuscript, **not** the
author-named preprint in this directory. The outer `SHA256SUMS` identifies
the public PDF and source ZIP. Historical statements that human evidence
was absent apply only to the earlier packages; the combined ZIP includes
the later, closed aggregate check.

## Scientific and reproducibility boundaries

The current contribution is a coupling-aware measurement contract and an
audit of fixed archived tensors, with a separate retrospective held-out
policy illustration and a limited reference-based human check. Neither
the fixed-archive audit nor the policy illustration establishes population
effects, new-data generalization, an equal-cost routing gain, or general
deployability. Scorer, subpool, and finite-draw ranges are sensitivity
summaries, not confidence intervals.

The policy illustration excludes evaluation scores from its current
fitting, tuning, and prediction stages but reuses a historically inspected
archive. The human check has definite two-person consensus on 26/30 GSM8K
and 11/30 GPQA-Diamond cells, all agreeing with the frozen scorer; 4 and 19
cells remain undetermined. It does not validate the full archive, other
scorer channels, or benchmark references. No original scorer or audit
result was changed by that check.

The package supports checks of the frozen score-tensor-to-result numbers
and their recorded bindings. It **does not include** the full analysis
source, internal release-control files, raw model responses, runtime, or
an end-to-end generation/scoring/analysis pipeline. The manuscript source
ZIP is LaTeX, not a replacement for the missing analysis implementation.
The top-level legacy `src/`, `scripts/`, and runners are not the current
manuscript's reproduction entry point.

Individual human annotations and explanations, identifying information,
invitations, and consent records are not released. The aggregate alone
cannot reproduce case-level judgments or independently verify participant
identity, independence, expertise, or adherence to blinding.

## Earlier versions and licensing

The previous public README is preserved at
[commit `0304028`](https://github.com/luka-krixvon/routing-oracle-experiment/blob/0304028d5ad721af96f4cc97fca10f8490126133/README.md).
Its headline claims and pipeline instructions are superseded for the
current manuscript; repository history and existing tags are retained.

The repository's MIT license covers legacy code. It does not relicense
the manuscript, all evidence contents, model weights, or third-party
benchmark material. Applicable upstream rights and terms remain in force.

For the current citation, use the
[dated preprint entry in CITATION.cff](../../CITATION.cff).
