# TDS submission evidence layout (2026-10-29)

## Current publication revision 316

The active private paper is `projects/tds/paper/submission_316/`: one shared source,
neutral preprint and MDPI anomaly-SI wrappers. Source ZIPs/platform metadata are
in private `submission/{preprints,applsci}/316/`; numeric identities/state history
are in `results/316_dual_submission/`; public reports and candidate manifests are
in `reports/316/` and `publication/316/`. R314/R315 artifacts remain frozen. No
result CSVs or author records are placed in paper/. The315 layout below is history.
Title/data parity, local builds, author approval, public release and actual
submitted/posted platform events remain separate assertions.

This amendment covers the temporal-distribution-shift paper, independently of
the historical Transaction Decomposition / Micro-RAG subsystem in document 06.
Shared algorithms remain under `src/`. TDS audit/replay tools belong to
`projects/tds/scripts/revision315/` and do not become dependencies of shared code.

| Asset | Location | Visibility |
|---|---|---|
| Work order and development discussion | `docs/work_reports/tds/315_applsci/` | Private |
| Preserved R314 manuscript | `projects/tds/paper/applsci/` | Private |
| Preserved R314 experiments and numeric inputs | `projects/tds/results/314_applsci/` | Numeric subset Git-eligible; raw identifiers/checkpoints remain local |
| Common scientific manuscript source | `projects/tds/paper/common/315/` | Private |
| Applied Sciences source and PDF | `projects/tds/paper/applsci/submission_315/` | Private |
| Preprints.org source and PDF | `projects/tds/paper/preprints/submission_315/` | Private |
| Source ZIPs, delivery PDFs and author records | `projects/tds/submission/` | Private except README indexes and venue plan |
| Frozen historical panel | `projects/tds/results/main_final_v2/` | Existing identity unchanged |
| Task 315 numeric results and manifests | `projects/tds/results/315_applsci/` | Reviewable; raw-data restrictions apply |
| Audit reports | `projects/tds/reports/315/` | Public-safe reports |
| Candidate inventory, licenses, replay commands | `projects/tds/publication/315/` | Public-safe metadata |
| Submission index | `projects/tds/submission/applsci/315/` | Public-safe pointer only |

The old `paper_applsci/` path is a historical Git/ZIP prefix. Its alias is recorded
in `canonical_source_tree.json`; no competing manuscript tree is created.
The latest author instruction supersedes the earlier `projects/results/` convention:
all TDS task numerics now live in `projects/tds/results/<task>/`. The old paths in
sealed manifests are resolved through `publication/layout/path_migrations.json`,
without rewriting frozen manifests. Paper folders contain TeX/Bib/styles/figures,
compiled PDFs and private front-matter metadata, but no result CSVs/checkpoints or
audit bundles. Rollback snapshots and QA renders are private outside paper/.
Benchmark and StreamMC evidence are untouched. The venue sequence is Preprints.org
followed by Applied Sciences, Special Issue “Machine Learning and Its Application
for Anomaly Detection” (https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt).
Local staging, anonymous public release verification, final author approval,
and actual journal submission are separate states. The 315 gate evaluator fails
closed when author facts or a verified immutable release are absent.
