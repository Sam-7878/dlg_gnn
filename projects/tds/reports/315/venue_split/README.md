# Current TDS venue split and result separation

Date: 2026-10-29. This follow-up supersedes the earlier 315 directory convention
and journal target. Earlier 315 artifacts remain historical evidence.

| Purpose | Canonical location |
|---|---|
| Shared scientific text, bibliography and paper figure/table sources | `projects/tds/paper/common/315/` (private) |
| Preprints.org manuscript | `projects/tds/paper/preprints/submission_315/` (private) |
| Applied Sciences manuscript | `projects/tds/paper/applsci/submission_315/` (private) |
| Venue source ZIPs, delivery PDFs and metadata | `projects/tds/submission/{preprints,applsci}/315/` (private except README) |
| Author records | `projects/tds/submission/author_records/315/` (private) |
| R314 numeric experiments | `projects/tds/results/314_applsci/` |
| R315 numeric derivatives and results | `projects/tds/results/315_applsci/` |
| Older reviewed-draft figure CSVs | `projects/tds/results/historical_reviewed_draft/data/` |
| Public reports and publication inventory | `projects/tds/reports/315/`, `publication/315/` |
| Private rollback, manuscript QA and original mixed archives | `projects/tds/private/` |

The author intends Preprints.org posting first, then Applied Sciences, Special
Issue [Machine Learning and Its Application for Anomaly Detection](https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt).
That exact target is user-specified. Direct official-page access returned HTTP429,
so its live deadline and portal acceptance are not verified. The former target's
deadline is not reused. No DOI, posting or submission is invented.

Both generated manuscripts use byte-identical scientific bodies and bibliography.
The preprint uses a publisher-neutral article class and removes journal branding
from article and supplement, consistent with [Preprints.org's formatting guidance](https://www.preprints.org/instructions-for-authors).
The same scientific results, limitations and negative finding are retained.
AI drafting assistance is also described in Methods. Remaining author facts stay
explicit; neither format is represented as approved for posting.

Build: `python projects/tds/scripts/revision315/build.py`.
Audit moved scientific data: `python projects/tds/scripts/revision315/audit.py`.
Check claims after build: `python projects/tds/scripts/revision315/claims.py`.
Refresh local numeric candidate: `python projects/tds/scripts/revision315/stage.py`.
Record venue readiness: `python projects/tds/scripts/revision315/close.py`.
The private common source is required for paper builds, while public numeric
replay remains independent of manuscript files.

Historical manifests retain exact bytes. Use `resolve_frozen()` and the migration
registry to locate relocated payloads; do not rewrite old seals or create second
numeric copies in paper/. The eight R314 scientific tests remain supported.
Historical R314 manuscript packaging/sealing commands are superseded by the 315
venue builder and must not be used to regenerate frozen submission archives.

Numeric outputs are Git-eligible, including the reduced prediction arrays and
curated candidate ZIP. Original identifier-bearing vectors, raw-source audit CSVs,
checkpoints, private reports/author records and all paper sources remain excluded.
This local organization does not publish a release or override upstream licenses.
