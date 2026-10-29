# Worktree and visibility scope

The user had already relocated `projects/tds/paper_applsci/` to
`projects/tds/paper/applsci/` before 315 began. The resulting tracked old-path
deletions were present at the input Git HEAD and are preserved as user changes.
No reset, restoration, cleanup of those inputs or Git history rewrite occurred.

315 changed only TDS paths, the project artifact layout and the new TDS architecture
amendment. Benchmark and StreamMC source/results were not modified. New tools,
numeric outputs, provenance and reports are eligible for Git tracking. The one
curated manuscript-free numeric ZIP has a scoped ignore exception, matching the
Benchmark evidence approach. This makes it reviewable locally; it does not publish
the data or constitute redistribution approval.

All active TDS paper paths, private author records, source ZIPs and the full
submission candidate remain ignored. Prior public Git history may contain old
manuscript sources; adding ignore rules does not erase history. No history rewrite,
commit, push, tag creation, release publication or journal submission was performed.
