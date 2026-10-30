# A24 reproducibility contract

A24's annotated evidence tag is `benchmark-a24-supplementary-2026-10-30-r1`; the actual resolved scientific commit and asset retrieval hashes are recorded in `publication/a24/release_retrieval.json`. A23's tag remains pinned to `94b0abbb4c89405da3f3c4fab7121f62bbe0b827`. Git tags can be moved; use the recorded commit and SHA-256 identities. No DOI is asserted.

Public standard-Python commands:

```
python scripts/reproduce_project.py --project benchmark --mode verify
python scripts/reproduce_project.py --project benchmark --mode tables
```

Verification checks public payload/source hashes; 455 selected successful run identities and 84 supported primary pairs; statistics n/k/Q/W/MC identities; every primary metric mean and sample SD from five stored scalar seeds; every supplementary mean and sample SD; 60 feature-only, 25 control, 30 alert and 4 full-resource records. It does not recreate third-party data or rederive private raw scores. Those were audited author-locally, including 113 new vector sets and 105 actual A08 bindings. ECOD's repeated seed evaluations are not independent model fits.

`paper` in a public checkout must exit explicitly with `A24 PRIVATE_INPUT_REQUIRED`. This is a checked availability boundary, not a paper reproduction PASS. The corresponding public path is `tables`; original per-seed scalars remain accessible. Author-local `paper` requires the ignored writer and original A23 snapshot, and actually compiles the two A24 PDFs. A24 source bundles are privately supplied to platforms; their clean builds, hashes and all-page visual/native validation are recorded separately. Existing older tracked source files are not A24 regeneration inputs.

Raw construction/full experiments require provider acquisition, the declared Python/CUDA stack and local path configuration. `a24_g1_inputs.py` rebuilds the three crypto graphs and compares private A08 score/input bindings; the public JSON records completed identities. `a24_run_studies.py`, `a24_followup.py`, `a24_cora_exact.py` and preregistered YAMLs describe the separate supplementary studies; these are never invoked by integrity/table verification. `full` is a guard, not an implicit GPU job.

No platform deposit, new coauthor approval or DOI assignment is implied. The work-order SI deadline is 2026-12-31; the official MDPI page returned HTTP429 during this audit, so current deadline and submission formalities require an author-side platform check.

## Release publication correction

The first A24 release was published before assets were attached; repository-level release immutability then prevented asset additions (GitHub HTTP422). That original tag/release is preserved without assets. The current `-r1` release uses the identical scientific payload and uploads assets while draft, verifies them, then publishes. The current pinned commit additionally corrects these release references. No scientific metric, source objective, preregistration or primary result changes.
