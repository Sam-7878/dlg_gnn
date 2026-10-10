# A08 sampled GADNR numerical execution revision 2

The authoritative scientific amendment is `projects/benchmark/protocols/A08_NUMERICAL_REPAIR_AMENDMENT_2026-10-10.md`. It supplements the raw-input amendment without changing frozen raw arrays, stable identities, masks, metrics, caller defaults or the fixed `.venv_cuda` environment.

## Observed failure and repair boundary

The initial corrected-input campaign preserved50 successful primary records before BSC GADNRseed42 failed with a negative float32 covariance determinant ratio. This is a numerical execution error, not OOM, timeout or a measured WSL/eGPU failure. The complete old source/qualification/run identities remain in `projects/benchmark/evidence/a08_data_repair/{execution_history,run_history}/pre_gadnr_numerical_repair_20261010/`; local scores/checkpoints are preserved in `local_storage/benchmark/a08_data_repair/campaign_history/`.

`src/gog_fraud/models/a08_gadnr_numerics.py` is an A08-only subclass and rank-one float64 evaluation of the existing detached sampled KL expression. It preserves the original implementation's log-ratio orientation, identity regularizer, sampling, loss weights and output dtype/device. It does not silently substitute canonical Gaussian KL, introduce jitter/clipping, or change gradient detachment. Nonfinite input/output remains a numerical failure. The historical shared detector in `models/pygod/gadnr.py` is unchanged.

All105 current primary runs restart from fresh initialization under the new `execution_manifest.json` hash. Earlier successful seeds are excluded from the current registry to prevent mixed backend cells. The durable campaign visits BSC first to exercise the previously failed configuration. Ethereum30, BSC/Polygon40 global epochs, Aug20 local epochs and the24-hour/model-dataset-seed guard remain in force.

## Actual qualification and acceptance scope

Sixteen arithmetic/upstream/ill-conditioned boundary tests pass. Actual sampled CPU and RTX3090 model forward, scores/loss, parameter gradients and one Adam update also pass at atol1e-5/rtol1e-4 under matched deterministic fixture conditions. An initial nondeterministic GPU fixture update exceeded tolerance and was rejected; fixture determinism was then controlled without relaxing tolerances. Production determinism settings are unchanged. Actual whole-Polygon seven-model and failed-chain BSC GADNR one-epoch qualifications are recorded in G5.

Final validation checks actual training loss-call counts against declared epochs, all raw-score/target/split/source bindings, approved numeric values in both PDFs, every rendered page, the current public payload, and actual clean public/local reproduction. Integrity, numerical qualification and full scientific/package acceptance have separate scopes. `SUBMISSION_READY` additionally requires author approval and a real immutable public evidence identity.

## Porting

The existing graph-level target API and historical Stream/TDS paths do not import the A08 numerical subclass or static node adapter. Porting sampled GADNR arithmetic requires the receiving project's own sample-size/objective/precision qualification and result lineage. Static all-population scaling and random node splits must not be substituted for temporal fit/split contracts. The Benchmark regression results do not clear other projects' empirical results or manuscripts.

## Public/private boundary

Current `verify` selects A08 and requires its curated numeric bundle; `--revision legacy-a07` explicitly selects preserved historical integrity. Unsubmitted TeX/Bib/style/class/PDF and manuscript writers remain local. Current raw node IDs/scores/checkpoints remain restricted. Preserved public paper-folder READMEs, already-published foundation figures and the exact historical public ZIP's numeric vectors/supporting plot remain public under their recorded identities. No commit, push, history rewrite or submission is performed by these facades.
