# A26 evidence and reviewer access

The public scientific release and commit are unchanged. The selected registry still has 455 successful runs (420 primary and 35 external) and 84 of 91 complete primary pairs. Public `verify` and `tables` remain the integrity and numeric replay entrypoints for those results. A26 adds the audit outputs here. [Scientific release](https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a24-supplementary-2026-10-30-r1)

## Per-node score coverage

| Collection | Retained arrays | Coverage |
|---|---:|---|
| Primary contract graphs | 105 | 35 each for Ethereum, BSC and Polygon |
| Earlier BitcoinOTC reruns | 35 | Seven models × five seeds, including ten DLG-Base/Aug vectors |
| Earlier DGraphFin AnomalyDAE reruns | 5 | Five seeds |
| Matched supplementary studies and resource checks | 113 | Separate from the 455 selected runs |

The selected registry records no retained array for 310 runs across 10 datasets (275 earlier primary runs and 35 historical LANL runs). This supersedes the count of 350 in the A25 access record. The 35 BitcoinOTC files and the five DGraphFin AnomalyDAE files were verified against both original file and raw array hashes. The BitcoinOTC finding confirms the earlier rerun report. A24 did not produce additional BitcoinOTC score sets. [Verification inventory](provenance.json)

For controlled access, request the dataset, model, seed and artifact from the corresponding author, Ki-Hyung Kim (kkim86@ajou.ac.kr), and explain the verification purpose. Anonymous reviewers can route the request through the handling editor. Third-party dataset and score access follow the applicable provider terms. Complete retraining also depends on local inputs, software and hardware.
