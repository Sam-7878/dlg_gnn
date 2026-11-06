# A28 public audit and submission evidence

This record corresponds to the final A28 submission files. The [PDF quality assurance](A28_PDF_QA.md) identifies the two PDF hashes and page maps. The [submission manifest](A28_SUBMISSION_HASHES.sha256) also identifies the two TeX files and source ZIPs held by the authors. The [scope review](SCORE_REDISTRIBUTION_SCOPE.md) explains which artifacts are retained internally.

The scientific archive remains [the A24 release](https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a24-supplementary-2026-10-30-r1). The original audit implementation is pinned at commit `71ffdbf9f7c75fc16163d17fd787f509dd6e2923`.

- [README.md](https://github.com/Sam-7878/dlg_gnn/blob/71ffdbf9f7c75fc16163d17fd787f509dd6e2923/projects/benchmark/publication/a27/README.md)
- [statistics.py](https://github.com/Sam-7878/dlg_gnn/blob/71ffdbf9f7c75fc16163d17fd787f509dd6e2923/projects/benchmark/publication/a27/statistics.py)
- [exact_null.cpp](https://github.com/Sam-7878/dlg_gnn/blob/71ffdbf9f7c75fc16163d17fd787f509dd6e2923/projects/benchmark/publication/a27/exact_null.cpp)
- [exact_input.txt](https://github.com/Sam-7878/dlg_gnn/blob/71ffdbf9f7c75fc16163d17fd787f509dd6e2923/projects/benchmark/publication/a27/exact_input.txt)
- [provenance.py](https://github.com/Sam-7878/dlg_gnn/blob/71ffdbf9f7c75fc16163d17fd787f509dd6e2923/projects/benchmark/publication/a27/provenance.py)
- [scores.py](https://github.com/Sam-7878/dlg_gnn/blob/71ffdbf9f7c75fc16163d17fd787f509dd6e2923/projects/benchmark/publication/a27/scores.py)

## A28 verification

- [Elliptic ROC-AUC](elliptic_roc.json) records the distinct historical and independently refitted five-seed results.
- [Reddit-Syn phase timing](reddit_dlg_aug_phase_timing.json) records an additional full-budget timing run. The stages inside fit sum to 201.604 seconds. Input loading and scoring are outside that sum. [Instrumentation source](measure_reddit.py)
- [GADNR diagnostic status](https://github.com/Sam-7878/dlg_gnn/blob/benchmark-a28-evidence-2026-11-06-r1/projects/benchmark/publication/a28/gadnr_diagnostic_status.json) preserves the three confirmed capped runs and the Reddit-Syn incomplete record.
