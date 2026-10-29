# TDS submission evidence layout (2026-10-29)

This amendment covers the temporal-distribution-shift paper, independently of
the historical Transaction Decomposition / Micro-RAG subsystem in document 06.
Shared algorithms remain under `src/`. TDS audit/replay tools belong to
`projects/tds/scripts/revision315/` and do not become dependencies of shared code.

| Asset | Location | Visibility |
|---|---|---|
| Work order and development discussion | `docs/work_reports/tds/315_applsci/` | Private |
| Preserved R314 manuscript and experiments | `projects/tds/paper/applsci/` | Private |
| Working 315 manuscript, PDF, source ZIP and author records | `projects/tds/paper/applsci/submission_315/` | Private |
| Frozen historical panel | `projects/tds/results/main_final_v2/` | Existing identity unchanged |
| Task 315 numeric results and manifests | `projects/results/tds/315_applsci/` | Reviewable; raw-data restrictions apply |
| Audit reports | `projects/tds/reports/315/` | Public-safe reports |
| Candidate inventory, licenses, replay commands | `projects/tds/publication/315/` | Public-safe metadata |
| Submission index | `projects/tds/submission/applsci/315/` | Public-safe pointer only |

The old `paper_applsci/` path is a historical Git/ZIP prefix. Its alias is recorded
in `canonical_source_tree.json`; no competing manuscript tree is created.
Numerical outputs have their own task namespace under the author's requested
`projects/results/`, without moving or rewriting Benchmark or StreamMC evidence.
Local staging, anonymous public release verification, final author approval,
and actual journal submission are separate states. The 315 gate evaluator fails
closed when author facts or a verified immutable release are absent.
