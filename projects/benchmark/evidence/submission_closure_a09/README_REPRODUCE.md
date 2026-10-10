# A09 evidence handoff and reproduction

A09 is an editorial/evidence closure of A08 execution revision2. No corrected input, production setting, seed, score or scientific source was changed. Original A08 acceptance/PDFs remain authoritative historical identities. The frozen numeric packages and scientific checkout are publicly released; current closure records separately document explicit user-reported coauthor approval and independent retrieval.

## Packages

1. `projects/benchmark/evidence/public_numeric_evidence.zip`: immutable historical archive, SHA256 `b1caab1124404eb6ac4dcd909007c69ceab2410e02f84363d53b0f8465b0c7f5`.
2. `projects/benchmark/evidence/a08_public_numeric_evidence.zip`: actual 226-payload corrected numeric package plus release_manifest.json (227 ZIP files), SHA256 `b1ae519a058784ed1d8306d2986748de6832e142209502e984030ff35b18e100`.
3. `projects/benchmark/evidence/a09_submission_closure_companion.zip`: A09 companion with original gate records, 105 sanitized scalar metric JSONs and flat CSV, actual 15 test-support rows, final dependency map, per-quantity qualification, seven unavailable-cell source records, source origins, historical identity recheck and byte-identical statistical replay. Contract IDs, new raw scores, arrays, checkpoints and private manuscript writer are excluded. Original private metric hashes identify the redacted originals; removing `selected_node_ids` never changes scalar metrics or alert counts.

The existing A08 public checkout contains the two original ZIPs and the required facades/source/locks. The companion supplements it; it does not independently supply raw training data or replace that checkout.

## Commands actually executed in a new local public checkout

```bash
PYTHON=/mnt/d/_work/goat_bank/.venv_cuda/bin/python bash projects/benchmark/reproduce.sh verify --revision a08
PYTHON=/mnt/d/_work/goat_bank/.venv_cuda/bin/python bash projects/benchmark/reproduce.sh tables --revision a08
```

Actual checkout: `local_storage/benchmark/a09_submission_closure/clean_public_checkout`. Both exit codes are zero. Original `--revision` choices are `auto`, `legacy-a07`, `a08`; no invented A09 training revision is required. Full help is `python scripts/reproduce_project.py --help`.

The Python environment remains `.venv_cuda`; run-bound A08 source/config/environment hashes are in `execution_manifest.json` and the existing 226-file package. New editorial helper source and package identities are separate from scientific execution identity.

## Additional author-local CPU audit

```bash
/mnt/d/_work/goat_bank/.venv_cuda/bin/python projects/benchmark/scripts/a09_collect_evidence.py
/mnt/d/_work/goat_bank/.venv_cuda/bin/python projects/benchmark/scripts/a09_summarize_evidence.py
```

The first command requires preserved private `local_storage/benchmark/a08_data_repair/runs/*/scores.npz` and `frozen/*/contract_graph.npz`. It checks node/label/mask alignment, score/metric hashes, validation grouped-score maximum-F1 selection and largest-threshold tie rule, test `>=`, and ceil alert budgets with stable-ID ties. All 105 recomputed metric dictionaries exactly equal originals. Public scalar/table replay does not claim to reconstruct those private vectors.

Statistical replay uses unrounded approved five-seed means, population SD, NumPy `default_rng(20261008)` as one stream across S1–S4, 200000 row/block permutations, batches of 2000, average tie ranks, the original tie correction, `>= observed - 1e-12` exceedance and `(extreme+1)/(B+1)`. Holm adjusts each entire declared Aug-versus-each family (18 rows total). Replayed output bytes equal A08. The exact artifact hashes and exceedance counts are in the companion, not inferred from PDF rounding.

All 351 preserved historical metric members are independently hash-checked: 350 successes and one failure; missing historical raw-score evidence is not upgraded. Archived initial corrected-campaign successes/failure and qualification runs are excluded from the active 105/455 counts.

## Access and approval boundary

GoG source: Luo et al., NeurIPS2024, DOI10.52202/079017-0894; repository `https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs` links the original ZIPs/labels. The local provider README SHA matches the frozen manifest, including Category0=fraud. Upstream LICENSE downloaded on2026-10-10 identifies CC BY-NC-SA4.0; no new redistribution permission is inferred. Derived mappings/raw scores and unsubmitted manuscript are withheld from GitHub by author policy, not automatically because of a legal prohibition. The user confirmed coauthor approval of the source/funding/conflict/AI disclosures and the controlled reviewer-access scope. Any actual private delivery still requires recipient/channel and delivered-file records.

Public `paper` deliberately rejects absent private manuscript sources. Author-local neutral/preprint and MDPI PDFs/source ZIPs are separate editorial-system candidates. Commit/push and immutable release publication are authorized and completed. Actual journal/Preprints.org upload or submission has not been performed. Final C5/C6 status and current private PDF identities are in submission_closure_manifest.json.

## Qualification recovery and final local packaging

The original CUDA records did not retain relative tensor observations or the AnomalyDAE mean-loss observation. The identical tiny fixtures were rerun solely to recover these missing observations, without production training. Original absolute records and tolerances remain separate. The recovered relative definition uses max(|reference|, atol/rtol), with floor0.1 for model float32 and10 for scalar float64. All tolerance fractions are below1.

```bash
CUDA_VISIBLE_DEVICES=GPU-ab53069a-3217-ee09-80f8-748b5fdbd156 CUBLAS_WORKSPACE_CONFIG=:4096:8 /mnt/d/_work/goat_bank/.venv_cuda/bin/python projects/benchmark/scripts/a09_recover_qualification.py
/mnt/d/_work/goat_bank/.venv_cuda/bin/python projects/benchmark/scripts/a09_pdf_review.py --finalize
/mnt/d/_work/goat_bank/.venv_cuda/bin/python projects/benchmark/scripts/a09_package_closure.py --private-only
```

The PDF finalizer requires actual every-page manual observations bound to current PDF/PNG hashes; it cannot generate author approval. After approval, --private-only compiles both exact private source ZIPs in fresh directories and requires byte-identical reviewed PDFs. Both existing numeric ZIPs are preserved. The companion inventory excludes its own two index files, which are themselves in the ZIP; the outer closure manifest and response report are separately delivered and not recursively inside that ZIP. A09 closure helpers are delivered under `closure_helpers/` as source snapshots; their runtime imports still require the repository checkout and preserved private inputs where stated.

## Actual public identity and independent replay

Scientific/source tag: [benchmark-a09-2026-10-10](https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a09-2026-10-10); commit [2cc6f85afc44830eb7eee92d709c6eb8d567ba2d](https://github.com/Sam-7878/dlg_gnn/commit/2cc6f85afc44830eb7eee92d709c6eb8d567ba2d). GitHub reports immutable=true. Both ZIPs plus SHA256SUMS were downloaded without credentials, their hashes matched, and an actual tagged clone passed verify/tables with byte-identical numeric outputs. See public_retrieval_verification.json for commands, hashes and scope.

The companion ZIP retains the reviewed pre-publication snapshot (including its then-current PDF review). Current authorization/publication/retrieval and final PDF reviews are later outer documents, separately committed; they were not retroactively inserted in the immutable companion. evidence_inventory.csv indexes the frozen ZIP payload, not every subsequently updated file in this directory. Current source helpers can differ from the snapshots in closure_helpers; scientific execution source hashes remain unchanged.
