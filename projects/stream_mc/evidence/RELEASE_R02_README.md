# DLG-SelectiveStream R02 scientific source/evidence

This prerelease publishes scientific code, the unchanged pseudonymous R01 raw
campaign and R02 verification/reaggregation. It does **not** publish the private
manuscript, approve it for submission, or claim peer review/production safety.
All-author exact paper-version approval and final AI wording remain OPEN.

Extract `r02_scientific_source.zip` into an empty directory. Place the three
`r01_numeric_evidence.zip.partNNN` files and `r01_release.json` in the extracted
`projects/stream_mc/evidence/`. From its root run:

```bash
python3 -m venv .verify-venv
.verify-venv/bin/python projects/stream_mc/scripts/r02_public.py
export PYTHONPATH="$PWD/src"
# Recorded scientific dependencies required for the following full-policy check:
python projects/stream_mc/scripts/r02_public.py --full-policy
```

The verifier SHA-checks/assembles the exact 258,850,451-byte R01 ZIP and checks
every member/source. Full mode verifies all 25 original model identities,
6,065 policy/control metric rows and 40 MC rows, without original addresses or
private Parquet. Outputs are separate; no retraining/retiming is implied.
R02 verification ZIP contains actual numeric receipts/statistical reaggregation,
18 replay-record checks, active 540 timing and 180 selected-execution checks,
the distinct new state-boundary probe and actual tests/negative fixtures.

Check `r02_release_checksums.sha256` before unpacking. The R02 ZIP's internal
manifest records all payload hashes and added R02 source hashes. External
access receipts are maintained separately in the repository after publication
to avoid self-referential archive hashes. Source ZIP includes frozen training
and runtime-v1/v2 sources; amendments do not relabel original timed code.

Dataset-derived inputs retain provider CC BY-NC-SA 4.0 attribution:
https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs.
Aliases are linkable, not anonymization. Provider labels are research contract
targets, not transaction-fraud truth. Unknown edge/label discovery times prevent
strict as-of claims. No universal calibration/risk/MC superiority is asserted.

The approximately 247 MiB raw evidence belongs here, not inside a journal's
120 MB total submission package. Private exact 939-target handoff, paper
sources/PDFs, address mappings and provider original archives are not assets.
