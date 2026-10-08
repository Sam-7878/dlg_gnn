# Project artifact layout migration — 2026-10-06

## Scope and ownership

The canonical reviewer path is now `projects/<project>/`. The benchmark A07
candidate paper, generated tables/statistics, bibliography, figure, PDFs and
final A06/A07 reports moved into `projects/benchmark/`. StreamMC's current
manuscript and TDS's current manuscript/wrappers moved into their project paper
folders. A hash-indexed copy of the published DLG-GNN predecessor source/PDF
is under `projects/dlg_gnn/paper/published_v1/`; the frozen original remains
unchanged. `projects/ARTIFACT_LAYOUT.md` maps every old location and exception.

The A05/A06 scientific evidence ZIP is unchanged (SHA-256
`459a3721efd1c0e17e49315ca31e92331ae754ae1683068303bdc793a5feab08`).
Frozen A05/A06 source snapshots, graphs, scores, approved registry, prior
round runners and A06 rollback remain at their historical locations. No
training or metric change was made.

## Local binary storage

The 2,926,837,760-byte `venv-before-rename.tar.gz` backup moved to ignored
`local_storage/benchmark/backups/`; SHA-256 before and after the move was
`5ecd07e207e9a070410601e2ed9bd9e4fd85336ff350c197daf0f64ca862b094`.
It is removed from the *current Git tree*, but old Git history still contains
the object. `environment/wheelhouse-a04/` (about 3 GB) remains the offline
wheel cache and is explicitly ignored. The 12 GB `outputs/` tree remains
Git-ignored local run storage; large frozen graph/checkpoint inputs were not
deleted or moved. The older 24 MB A07 review ZIP moved to ignored local backup
storage with SHA-256 `562981c32263f76234e95eae6a2e9ead7a358cbf7dc7cd7cf2a7d05c7632876a`.

Five byte-identical A07 legacy TeX/Bib/PDF duplicates were removed from
`docs/papers/_42_01_Benchmark_PrePrints/` after comparing SHA-256 with the
new canonical copies. That directory and `_42_Benchmark/` retain their older
frozen inputs because historical tests/scripts refer to them. The active
StreamMC and TDS papers are no longer under `docs/papers/`.

## Reproduction and PDF checks

- New shallow, no-local clone of commit
  `2ad775672a51e3b832fe004a55be107366b16994` occupied 118 MB at
  `/tmp/dlg-layout-dGLKgD/repo` on WSL2. It contained all 35 curated benchmark
  paper artifacts and the 645-payload frozen evidence ZIP, without the local
  wheelhouse, backup tarball, raw graph workspace or build caches.
- In that checkout, all four `verify` modes passed under `.venv_cuda` Python
  3.14.4; DLG, StreamMC and TDS scientific fixtures also passed. Benchmark
  `paper` passed in 2.93 s: 44 references, four S1/S2 Holm rows, 14 claim links,
  consistency gate PASS, and exact expected TeX/Bib/statistics hashes.
- The relocated MDPI, neutral and F1 TeX compiled. The MDPI PDF is 14 A4 pages
  with no overfull or undefined-reference warnings. Text extracted from all
  three rebuilt PDFs matches the prior reviewed PDFs after normalizing only
  the automatically printed October 5/6 date. The 14 MDPI pages and one F1
  page were rendered and inspected. Current PDF hashes and page notes are in
  `A07_Final_PDF_Review.md`.
- `projects/stream_mc/paper/current/audit_latex.py` passed after replacing an
  absolute old manuscript path. TDS `make preprint` produced a 14-page PDF
  at the new path; its existing overfull-box warnings remain a TDS paper issue,
  outside this repository-layout migration.

## Public-release gate

The A07 article still has a future release URL statement. This migration is a
local reviewer candidate, not a deposited preprint or journal submission.
The A07 work order requires an actual public immutable tag/commit/release URL
in Data Availability, followed by another PDF build and review. Public GitHub
Release/Zenodo/Preprints.org/MDPI steps have not been claimed here.
