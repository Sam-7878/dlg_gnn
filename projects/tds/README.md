# TDS facade

For Applied Sciences work order 315, see [reports/315](reports/315/) and
[publication/315](publication/315/). Reproduce the prediction audit with
`python projects/tds/scripts/revision315/audit.py` from the repository root.
New task-specific numeric outputs live at `projects/results/tds/315_applsci/`.
Shared scientific implementations remain in `src/`; project scripts stay here.
Unsubmitted manuscript sources, PDFs, author records and source ZIPs stay local
under `paper/applsci/submission_315/`. No public deposition is implied by a local path.

`./projects/tds/reproduce.sh verify` checks the graph construction and data loader source hashes. With NetworkX and Matplotlib installed it also parses a fixed entity/context fixture, builds the graph, checks local risk and compares a frozen JSON hash; otherwise the scientific smoke is reported as skipped. Live LLM integration is separate and may require credentials. This A07 facade does not claim to reproduce TDS paper results.

The working manuscript and format wrappers are in [paper/README.md](paper/README.md).
