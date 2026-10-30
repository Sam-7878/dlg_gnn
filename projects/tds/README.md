# TDS facade

Current publication closure: [R318 reports](reports/318/README.md). The private
canonical manuscript is `paper/submission_318/`; venue candidates/signoffs live in
`submission/{preprints,applsci,author_records}/318/`. R318 reconciles earlier and
post-declaration R317 identities, cold-builds both generations, and preserves all
R317 scientific results. It reflows the checklist and clarifies pseudonymous IDs
and reproduction scope, with no new training. Current ledgers are in `results/318/`;
the approved-in-wording author authority remains the user-designated R317 JSON.
Science/candidate integrity PASS is distinct from pending publication rights,
exact-file signoff, an actual public release, and platform submission/posting.
The [R317 scientific campaign](reports/317/README.md), [R316](reports/316/README.md)
and R315 links below remain preserved histories.

Run `python projects/tds/scripts/revision317/replay.py` for CPU replay of 110 test
arrays and 95 validation-only calibrators. Full source reconstruction and bounded
training require the upstream GoG archives and the environment recorded in
`results/317/environment.json`; see `scripts/revision317/README.md`.

For Applied Sciences work order 315, see [reports/315](reports/315/) and
[publication/315](publication/315/). Reproduce the prediction audit with
`python projects/tds/scripts/revision315/audit.py` from the repository root.
Task-specific numeric outputs live at `projects/tds/results/315_applsci/`;
preserved R314 experiments live at `results/314_applsci/`.
Shared scientific implementations remain in `src/`; project scripts stay here.
Unsubmitted manuscript sources, PDFs, author records and source ZIPs stay local
under `paper/`. The common scientific source is `paper/common/315/`; generated
venue sources are `paper/preprints/submission_315/` and `paper/applsci/submission_315/`.
Build both with `python projects/tds/scripts/revision315/build.py`. Edit the common
source, then regenerate both formats; do not independently edit generated bodies.
Source ZIPs and private author records are in `submission/`, QA/rollback in
`private/`, numeric evidence in `results/`, and public reports in `reports/`.
See [the current venue/layout report](reports/315/venue_split/README.md).
The intended sequence is Preprints.org first, then Applied Sciences Special Issue
“Machine Learning and Its Application for Anomaly Detection”. Neither posting nor
submission is implied by local build success.

`./projects/tds/reproduce.sh verify` checks the graph construction and data loader source hashes. With NetworkX and Matplotlib installed it also parses a fixed entity/context fixture, builds the graph, checks local risk and compares a frozen JSON hash; otherwise the scientific smoke is reported as skipped. Live LLM integration is separate and may require credentials. This A07 facade does not claim to reproduce TDS paper results.

The working manuscript and format wrappers are in [paper/README.md](paper/README.md).
