# Draft release notes — benchmark v2 current evidence

**Proposed public tag:** `benchmark-v2.0.0-preprint` (create only after author approval and final release review). The existing `v1.0.0-preprint` is historical and must remain untouched. Local review tag: `benchmark-v2.0.0-a07-candidate-r2` at commit `2d51c162c7f72d89de19b4f78fb1b3059ca92999`.

## Release summary

Frozen support-aware graph anomaly benchmark with 13 primary graphs and LANL external validation. Seven primary inferential models produce 80/91 supported primary pairs and 435 approved primary/external successful records. CONAD reference behavior is retained as an implementation diagnostic only. The evidence asset `frozen_a05_a06_evidence.zip` carries 645 SHA-256-listed files including approved registry, canonical dataset manifest, source/environment provenance, exact sparse/fused/AnomalyDAE equivalence reports, statistics and raw-score evidence where retained.

## Reviewer commands

```bash
python scripts/reproduce_project.py --project benchmark --mode verify
python scripts/reproduce_project.py --project benchmark --mode paper
```

The first command needs no scientific packages or third-party graphs. The second needs SciPy and regenerates paper inputs from frozen evidence. Full benchmark training remains a manual, GPU-intensive campaign and is not launched by CI.

## Important limitations

Ethereum, BSC and Polygon are hashed locally constructed hybrid graphs without a documented public raw download or redistribution permission. Full independent reruns of these cases require separate data access. Historical legacy runs do not all retain raw predictions or run-bound package locks, so bitwise replay of every historical run is not claimed. F1 is provided in a one-page supplement in the A07 manuscript package.

## Publication gate

After the GitHub Release asset has a public URL, verify the tag resolves to the intended commit and rerun `projects/benchmark/scripts/a07_build_manuscript.py` with `--release-tag`, `--release-commit`, `--release-url`, and DOI if issued. Rebuild and review PDF hashes. Do not deposit the current candidate PDF, which still contains a future release-URL sentence.
