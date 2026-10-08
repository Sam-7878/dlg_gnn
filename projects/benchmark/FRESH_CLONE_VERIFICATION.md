# Public checkout qualification — 2026-10-09

The current qualification uses a **clean export of the prospective public file set**, not a claim that the revisions have already been committed/pushed. It copies tracked and non-ignored candidate files into a separate directory, with no unsubmitted manuscript sources or private manuscript inputs. The base Git HEAD is `f21ebd6348f1a781476d07823f01566e43d9fb8f`; output is `/tmp/dlg-astra-public-reviewed-_1mo7fzg`. The tested export contains 1028 files (137.01 MB).

| Command | Result |
|---|---|
| Standard-Python `verify`, four projects (DLG / benchmark / StreamMC / TDS) | PASS each |
| `.venv_cuda` benchmark `tables` | PASS: numeric tables, run audits and 200,000 permutations per view regenerated; resulting evidence hashes match |
| Standard-Python benchmark `paper` | Expected explicit private-input guard; author withheld the unsubmitted manuscript |

Benchmark verification checks 641 public ZIP payloads, 13 primary graphs, seven models, 80 supported pairs, 435 successes, archived equivalence and 39 supplementary evidence hashes. It does not retrain detectors. Optional scientific fixtures are separate recorded `.venv_cuda` checks.

The first export exposed four expected source hashes computed on CRLF while the committed code had LF. Each old hash exactly equals the current source converted to CRLF; no model code changed. The current-byte hashes and prior hashes are both retained. See `reports/astra_revision/source_newline_audit.json` and first/final export JSON reports.

Historical checkout claims for earlier repository history are retained in `reports/a07/Historical_Fresh_Clone_Verification.md`; they are not the current publication identity. After upload, the public immutable commit/tag needs a fresh checkout test. Candidate layout/integrity PASS does not clear unresolved hybrid construction or missing historical score provenance.
