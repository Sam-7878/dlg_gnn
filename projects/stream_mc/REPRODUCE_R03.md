# R03 author-local submission closeout

Target: **Applied Sciences — Graph Neural Networks: Theory, Methods and
Applications**, special issue `C80IXAF9V4`, announced20November2026 deadline.
User's final instruction: after author approval, Preprints.org first, journal
second. A temporarily considered Electronics issue was explicitly reversed.
No actual posting, editor contact or journal submission is authorized by these
tools. Exact target/config and unverified final portal items are in
`configs/submission_target_r03.json`.

The original R01 campaign and public R02 release stay immutable. New R03 tools
write **ignored private** `handoff/r03/` and `paper/current/r03/`, not new raw
measurements in R01. The user will supply final AI-use/CRediT facts later;
all-author exact-version approval is missing, so both publication gates HOLD.
No fictitious author approval file, DOI, receipt or approved-final ZIP is made.

Use the root CUDA venv under **Ubuntu** (actual environment26.04.1LTS,
Python3.14.4). R03 uses `CUDA_VISIBLE_DEVICES=-1`; no GPU fits/long replay.
Run from repository root, in order, only if preparing a **new** author-local
handoff (do not overwrite an existing reviewed candidate):

```bash
source /mnt/d/_Work/goat_bank/.venv_cuda/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=-1
python projects/stream_mc/scripts/r03_audit.py baseline
python projects/stream_mc/scripts/r03_audit.py diagnose
python projects/stream_mc/scripts/r03_audit.py access
python projects/stream_mc/scripts/r03_paper.py --run-label candidate1
# Actually inspect every PDF page/contact sheet; supply an honest
# handoff/r03/visual_review_decision.json bound to those PDF identities.
python projects/stream_mc/scripts/r03_finalize.py
python -m unittest discover -s projects/stream_mc/tests -p test_r03_closeout.py -v
```

`baseline` needs the author-local late R02 paper, successful083831Z receipt and
original939 inputs. It checks33 paper source inputs,19 generated inputs
(18tables+numeric macros), final16/16/7 PDFs/renders/package memberships and1103
preserved old files. It does **not** imply an external person received939 files.
The freshly downloaded public derivative instead has1001ZIP members.

`diagnose` loads saved degree-logistic coefficients/scaling and retained test
inputs, **without fitting or neural forwards**. Canonical API/expit gives
AP.505262786955955; manual float32 sigmoid gives.5066910654043318, matching
recorded timing scalars. Raw max error1.1921e-7/calibrated2.8711e-7, labels
unchanged and distinct-score count3621→3620. This is a new arithmetic diagnostic,
not recovered historical per-repeat vectors or new performance timing. It also
checks90 achieved-budget rows and comparator-minus-margin sign/matching scope.

`access` really downloads all8 versioned public assets without credentials,
checks full bytes/SHA/payloads and executes the downloaded source in a fresh
stdlib venv. The previous R02 full-policy6065/MC40 verification is **reused,
not rerun**. Same operator/host isolation is not external scientific peer review.

`r03_paper` needs actual TeX/BibTeX/Poppler and PIL. Fresh source-only stages,
actual subprocess exits and strict reference/overflow checks are used; PDF-hash
namespaces prevent stale-page mixing. It preserves every numeric table body
and changes only three caption lines. A repeated build needs a new run label,
and new exact PDF identities must be visually reviewed; approval never carries
over automatically. `finalize` creates candidate source/PDF/letter packages,
checks member hashes, independently clean-builds package sources and checks
identical extracted text, records conservative full-upload bytes, and locks the
new identities separately from old R02. It intentionally refuses to overwrite
existing packages or infer approval from build success.

Private handoff includes scientific-scope/rights/assessment/cover/checklist,
late R02 post-host receipts and logs, manifests and honest GateP/J statuses.
No paper/approval/original address mapping is added to public Git. Technical
access can pass while rights, AI/CRediT, final author approval and live portal
policy remain OPEN.120MB is a planning budget, **not newly verified exact
Applied Sciences policy**. CodeMIT, upstream dataCCBYNCSA4.0 and manuscript
posting license are distinct; technical inventory is not legal clearance.

Before actual posting: authors supply/confirm facts and rights, remove internal
candidate wording only then, rebuild/inspect/package under new identities and
approve those bytes. After actual preprint posting add its real DOI/status to
the journal cover/checklist. No automatic full25-model retraining or100k replay
is scheduled by R03.
