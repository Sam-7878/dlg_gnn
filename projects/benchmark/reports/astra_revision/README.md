# Astra pre-submission revision audit (2026-10-09)

## Current decision

**Integrity and arithmetic can be verified; submission readiness is not cleared.** No neural campaign was retrained, no frozen metric/score was overwritten, and no repository push or submission was performed. The author requests that unsubmitted manuscript LaTeX sources stay local; only code and supporting evidence are prepared for GitHub.

## Review disposition

| Review issue | Action and evidence | Remaining condition |
|---|---|---|
| Public repository/release mismatch | Current 13+LANL / seven / 80 of 91 / 435 scope; explicit verify/tables/local-paper modes; 641-payload public derivative; per-run/cell audit CSV | Current revision still needs an immutable published commit/tag; previous f21 identifies historical frozen evidence only |
| Graph, label and feature provenance | Every one of 24,169 crypto labels matches original labels.csv Category0-positive; A03 partial and A05 full hashes verified; archived 269-file patch audited; global CSV address alignment and core pipeline tracing; provider/injection rules documented | Original hybrid generator and sole nonconstant feature definition still missing; all 43,140 / 22,427 / 6,896 added edges join same-label nodes; unsupervised construction validity not cleared |
| Ethereum high AP but low F1 | Same-run metric JSON IDs/hashes checked; repaired Aug uses percentiles80–99.5 val-F1 grid versus comparator val-prevalence threshold; all three crypto Aug F1 cells marked diagnostic | Repaired score arrays, exact thresholds/confusion matrices absent; retrospective uniform threshold/alert re-evaluation impossible from scalar metrics |
| Exactness | Archived error/tolerance CSV; new 12-case CPU loss/score/gradient/one-Adam-update qualification covering directed, weighted duplicate/self-loop Gram and GCN 2/4 layers; all PASS | CPU qualification is limited to declared fixtures; it is not full-run bitwise equivalence |
| Memory versus execution | Eight matched cap/full measurements exposed; AnomalyDAE unmatched 8-vs24 claim removed; row-block update order described; projected runtime and resumed segment time separated | No matched AnomalyDAE cap/full measurement; hardware timing is device-specific |
| Baselines/config/scores | Pinned constructor signatures plus caller settings/provenance; A03 three-seed sensitivity table; low ROC/near-prevalence AP explicitly discussed; no test-informed inversion; CONAD remains diagnostic | Historical score arrays/loss curves/run-bound defaults incomplete; new audited training would be needed to certify suspect cells |
| Aug causal mechanism | Numeric Elliptic Base-70/Permuted/Zero controls; multi-factor comparison acknowledged; alignment/property-selection claims removed | Controls do not isolate a causal augmentation mechanism; no universal selection rule established |
| Statistics | S2 DOMINANT/Aug mean-rank tie fixed; 200,000 dataset-block permutations per S1–S4; fixed seed20261008; source/native/injected descriptive groups; SD split/optimization caveat | Small-block inference remains conditional; no significant S3 omnibus result; zero MC exceedances indicate simulation resolution |
| Novelty/editorial | BOND/PyGOD/GADWild scope comparison and direct GADAM contrast; campaign terms defined; no unsupported nonlinear Gram impossibility theorem; PREM formal ICDM DOI; variant names reduced; readable metric panels | Paper remains a candidate pending scientific provenance resolution |

## Evidence locations

- `../../evidence/public_numeric_evidence.zip`: public numeric derivative. SHA-256 `b1caab1124404eb6ac4dcd909007c69ceab2410e02f84363d53b0f8465b0c7f5`. Four manuscript writers/packagers removed; all retained payloads byte-identical.
- Original frozen ZIP remains local, unchanged, SHA-256 `459a3721efd1c0e17e49315ca31e92331ae754ae1683068303bdc793a5feab08`. Existing Git history contains this old ZIP/writer; current index removals cannot erase it.
- `../../evidence/astra_revision/MANIFEST.json`: exact hashes of public derived audit/statistic files.
- `../../DATASET_CONSTRUCTION_AUDIT.md`: source global graph, labels and pipeline call paths; JSON source hashes in evidence.
- `exact_cpu_qualification.json`, `source_newline_audit.json`, `fresh_public_export_verification.json`: qualifications and separate directory verification.
- `Final_PDF_Review.md`: candidate PDF hashes and every-page review. PDFs and TeX stay under local `paper/current/`.

## Next scientific decision

Provide the code/record that produced the exact frozen hybrid tensors. If labels informed graph edges or the remaining feature, the crypto unsupervised evaluation requires a label-independent construction and a new separately identified campaign; frozen results must be preserved and not relabeled as a corrected run. If the producer is unavailable, the author must decide whether to restrict primary conclusions to the auditable subset and treat these artifacts as diagnostics. Current evidence alone does not resolve that decision.

`python scripts/reproduce_project.py --project benchmark --mode verify` checks integrity. `--mode tables` additionally recomputes public arithmetic with NumPy/SciPy. Neither establishes full neural reproducibility or submission readiness. A public checkout intentionally cannot regenerate the withheld manuscript.
