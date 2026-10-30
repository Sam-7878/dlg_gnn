# R318 final evidence reconciliation

Use the shared Ubuntu environment identified in `/mnt/d/_Work/goat_bank/dev_env.md`.
From repository root:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /mnt/d/_Work/goat_bank/.venv_cuda/bin/python projects/tds/scripts/revision318/numeric_audit.py
/mnt/d/_Work/goat_bank/.venv_cuda/bin/python projects/tds/scripts/revision318/candidates.py
/mnt/d/_Work/goat_bank/.venv_cuda/bin/python projects/tds/scripts/revision318/closure.py
```

The first command reads frozen numerical results and writes R318 audits. The
second preserves R317 and creates private R318 manuscripts, reflows the checklist,
clarifies Data Availability, and independently builds both source-ZIP generations.
The last prepares ledgers, rights/release documentation and unsigned exact-file
approval forms. It does not push, tag, publish, submit, or represent a signature.
Current supplied author facts come from `submission/author_records/317/author_confirmation_from_a16.json`.

Public-safe code/results/reports remain separate from ignored paper sources,
venue files, author records, private mappings and QA renders. No detector training
entry point is invoked. Old manifests are historical evidence, not current ledgers.
