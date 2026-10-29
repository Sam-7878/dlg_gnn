"""Public evidence reports; personal confirmations remain in private records."""
import json
import platform
import subprocess
from pathlib import Path
import pandas as pd
from common import ROOT,PROJECT,OUT,REPORTS,PRIVATE,sha,write


def put(name,text):
    REPORTS.mkdir(exist_ok=True);(REPORTS/name).write_text(text.strip()+'\n')


def main():
    d=json.loads((OUT/'selection_decision.json').read_text());s=pd.read_csv(OUT/'model_summary.csv')
    c=pd.read_csv(OUT/'modelwise_calibration.csv');b=pd.read_csv(OUT/'time_block_sensitivity.csv').iloc[0]
    extra=[]
    for folder in ['submission/preprints/316','submission/applsci/316','publication/316']:
        for p in (PROJECT/folder).glob('*.zip'):extra.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    dest=OUT/'additional_316_identities.json'
    if dest.exists():assert json.loads(dest.read_text())==dict(files=extra)
    else:write(dest,dict(files=extra))
    snapshot=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(Path(__file__).parent.glob('*.py'))]
    write(OUT/'source_manifest.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        new_code_committed=False,files=snapshot,scope='317 working-tree source identity, not a fabricated release commit'))
    from importlib.metadata import version
    write(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        executable=__import__('sys').executable,numpy=version('numpy'),scipy=version('scipy'),sklearn=version('scikit-learn'),
        torch=version('torch'),torch_geometric=version('torch-geometric'),matplotlib=version('matplotlib'),
        extra_dependency='project-local LightGBM4.6.0; shared venv not upgraded',
        gpu='cuda:1 RTX3090 24GB; batch and optimizer held fixed for comparability',
        observed_os=Path('/etc/os-release').read_text(),date='2026-10-30'))
    registry=pd.read_csv(OUT/'run_registry.csv');registry['base_git_commit']=json.loads((OUT/'source_manifest.json').read_text())['base_commit']
    registry['working_tree_source_manifest_sha256']=sha(OUT/'source_manifest.json')
    registry.to_csv(OUT/'run_registry.csv',index=False)
    put('collection_end_selection_audit.md',f'''# Collection-end and sample-time audit

**Decision: {d['decision']}.** Claims are restricted to sampling-associated
composition in a retrospective archive. No identified censoring mechanism or
prospective label availability is asserted.

All 24,316 contracts were independently streamed from the hash-matched three ZIPs:
268,282,924 rows, all valid under the original endpoint check. Recovered last times
match every packed cutoff; source label Category0 joins one-to-one. First-time
sorting uses the same full population and count sizes, before any new model fitting.
The preregistered descriptive trigger was an absolute >=5 percentage-point
prevalence change in any split; it is not a significance test.

| Sorting | Train | Validation | Test |
|---|---:|---:|---:|
| Last transfer |39.8508%|8.4453%|2.9331%|
| First transfer |14.7935%|71.3189%|56.9901%|

Pooled median observed lifetimes: positive0.670625days; other608.447361days.
Ethereum medians0.524583/717.142940days. These are observed spans, not survival
estimates under a known censoring model. First-time class quantiles, split-specific
first/last/activity/chain summaries and endpoint proximity are in the linked CSVs.

The pinned official README documents End Month2024-02. Empirical maximum is
1709251199, 2024-02-29T23:59:59Z. The inspected collector's generic end_block and
API termination loop do not establish a precise archive stopping policy.
`collection_boundary_time` therefore remains unknown; the observed maximum is
stored separately. Fraud discovery time is unknown for every contract.

Seven-day eligibility retains24,179 and removes137 (112positive/25other).
Thirty-day eligibility retains23,739 and removes577 (422positive/155other).
Inclusion does not require later activity. The new `GoG-317-first7d-v1` rebuilds
all feature degrees and tail128 topology <=cutoff; every sample has >=1 valid
transfer and all24,179 maximum input timestamps satisfy cutoff. Its split is
16,925/3,627/3,627, positives2,507/2,542/2,037. Test prevalence56.1621%.
New and old tests are not paired and have distinct identities.

Public `sample_time_lineage.csv` is identifier-reduced, not formally anonymous.
CSV is used instead of Parquet because PyArrow is absent; no environment upgrade
was required. The address-bearing mapping and source snapshots remain private.
Zero chain-qualified contract overlaps do not establish wallet independence;
cross-chain address-string overlaps are24/30/36 for train--val/train--test/val--test.

Evidence: `results/317/selection_decision.json`, `sample_time_lineage.csv`,
`first_last_ordering_diagnostics.csv`, `lifespan_by_class.csv`,
`pooled_time_quantiles.csv`, `fixed_horizon_eligibility.csv`, `horizon7_*`.
''')
    put('comparator_and_primary_panel_audit.md','''# Comparator and primary panel audit

The primary comparison uses the original matched GIN+BCE, GraphSAGE+BCE, summary
LR and MLP; new GATv2 time/no-time, LightGBM and IF controls use the same test IDs.
Actual feature and loss differences are declared rather than hidden. The separate
archival GIN/GAT/chain-recurrent/focal-SAGE and averaging panel is moved to the
Supplement. In particular, archivalGIN AP0.4648 is never substituted for matched
GIN+BCE AP0.4524 in the paired MLP comparison (AP0.4751).

New GATv2 meanAP time0.311875 vs no-time0.389731; five seeds each, same weightedBCE,
60-epoch ceiling/patience5, hidden48, batch64 and AdamW settings. Removing time also
omits its16 inputs from the classifier head; this is a branch/head ablation, not a
claim to isolate extrapolation causality. Original test normalized float32 time
is[1.0036826,1.0040067], all outside train[0,0.99999976]; initial maximum harmonic
frequency gives about4radians of phase extension.

The former TGN-style model is a **chain-level recurrent-state comparator**:
three48-dimensional states indexed by chain; reset at every epoch and partition;
logit from current summary+previous state; then GRU update from summary+time;
no label input; frozen weights in held-out traversal; no cross-partition warm-start.
It is not original TGN node memory. The executable label-independence test is retained.

All optional lightweight items were EXECUTED: one LightGBM4.6.0 configuration,
unlabeled mixed-train IsolationForest with fixed negative-score orientation,
95 modelwise validation-only TS calibrators, chain-specific AP, train-chain prior,
fixed budgets/tie bounds, and 2,000-draw hour-block sensitivity. No full original
TGN/DyGLib retraining, second horizon grid or test-directed tuning is added.
IF AP0.016989 is reported without reversing scores after seeing labels; probability
metrics are inapplicable. Deterministic LR/LightGBM repetitions have no seed spread
and do not remove sampling uncertainty. All45 new fit records retain test_access_count1.

The finite-seven-day panel has LR AP0.906827, MLP0.958154, GIN0.635791,
time-GATv20.612597 at test prevalence0.561621. This reinforces competitive simple
controls on a different retrospective task, not a paired gain over the old test.

Evidence: `panel_comparability.csv`, `run_registry.csv`, `experiment_preregistration.json`,
`temporal_input_ranges.csv`, `temporal_input_ablation.csv`, `state_semantics_audit.json`.
''')
    put('panel_comparability_audit.md',(REPORTS/'comparator_and_primary_panel_audit.md').read_text())
    put('statistics_claim_consistency.md',f'''# Statistics and claim consistency

The preserved seven archival and six exploratory families remain unchanged and
separate; no new317 result is added after observing it. AP groups score ties and
matches scikit-learn, including duplicated bootstrap observations. Finite swap
correction is(1+extreme)/(10000+1), minimum1/10001. Original Holm p0.0920 and0.1279
are retained; matched GIN-minus-MLP exploratory Holm p0.118388 is not equivalence.
The swap unit is each sample within each fixed seed; point statistic is mean seed
AP difference. Event bootstrap does not resample seeds as independent observations.

Observed rank invariance under scalar calibration does not imply score-swap null
invariance: score swaps mix magnitudes across two vectors. Thus differing MC-vs-raw
and MC-vs-TS p-values are not alone a bug. A separate rank-swap family was not added.

Hour-block sensitivity:19 complete UTC-hour blocks,2000/2000valid draws;
conditional GIN-minus-MLP interval[{b.ci95_low:.6f},{b.ci95_high:.6f}]. No latent
wallet dependence or future-deployment coverage guarantee is established.
Ethereum/BSC chain bootstrap intervals use2000draws conditional on fits. Polygon
one positive has a defined AP point but insufficient comparative support.

CPU replay identified a new317 calibrated-ROC discrepancy <=3.959e-6 caused by
clipping probabilities before ranking. Both AP and ROC now use unmodified scores;
the corrected new317 calibrated ROC table is independent of untouched R314--316
artifacts. CSV replay uses round-trip float parsing to preserve IEEE score ties.
The full Supplement includes F1@0.5 and adaptiveECE; no metric is merely promised.

Main title/abstract narrow to sampling-associated prevalence shift. Checklist
consequences include only measured effects; no invented leaked-input inflation,
right-censoring causality, long-term shift or prospective early-warning result.
''')
    put('A16_to_TDS_declarations_audit.md','''# A16 to TDS declaration audit

Actual public source was fetched through connected GitHub, not inferred from an old
local copy: [A16 source](https://github.com/Sam-7878/dlg_gnn/blob/main/projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.tex),
Git blob f285b88dd63d0dcc9f931f72e884a9cddb7517cf. Snapshot is author-local; metadata
is `results/317/online_source_provenance.json`. The user's existing authorization
to apply A16 workflow wording to TDS is preserved. It is not a signature on317 PDFs.

| Section | TDS application | Remaining fact check |
|---|---|---|
| CRediT | Same requested division of roles | Both authors approve exact317 version |
| Funding | IITP--ITRC, Korea MSIT, IITP-2026-RS-2021-II211835 | Official award and funder role absent |
| IRB/Consent | Not applicable for existing public contracts and no recruited participants | TDS author declaration confirmation |
| AI | Same named tools, expanded to actual coding/analysis assistance | Earlier exact model versions not retained; not invented |
| Data availability | Shared Sam-7878/dlg_gnn, TDS paths and local replay scope | TDS immutable release/rights or publisher-accepted restriction |
| COI | No automatic inheritance of A16 no-conflict assertion | TDS-specific declarations absent |

The A16 final published-version agreement sentence is not copied. A16 Seong-Su Park
and current TDS SeongSu Park require author choice; department, corresponding role,
ORCIDs/contact identity remain unconfirmed. The same shared declarations.tex is
materialized into both versions, with Methods AI disclosure and back-matter product
details. No fake independently-verified/finalized assertion is added for human approval.
Benchmark A20 release identifiers,105vectors and350-limitations are not transplanted.
''')
    put('tds_public_release_availability_audit.md','''# TDS availability and redistribution audit

**HOLD_PUBLIC_RELEASE.** Anonymous GitHub read returned three releases and no TDS
release name/tag. This is a bounded public inventory, not an assertion that private
or other unpublished artifacts do not exist. No tag/push/release/post action occurred.

| Asset | Local disposition | Public rights |
|---|---|---|
| Shared/TDS code | Repository scripts, source hashes | Repository code license; no automatic data relicensing |
|110 numerical arrays, lineage, tables/statistics | Identifier-reduced local candidate | Derivative redistribution approval missing |
| Raw GoG ZIPs/labels | External author-local source | Upstream CC BY-NC-SA4.0 and access terms |
| Contract mappings | projects/tds/private/317 | Excluded |
|55 original+45new checkpoints | Ignored author-local paths | Excluded pending rights review |
| Manuscripts/source ZIPs | Ignored paper/submission317 and venue delivery folders | Private author review only |
| Author identity/signatures/funding proof | submission/author_records/317 | Private |

Local numerical replay is L2. New training actually ran under the declared source
and environment, but no clean independent full-retraining/L3 claim is made. An approved
TDS tag, commit, downloadable asset, downloaded SHA and anonymous replay are still
required, or a documented restriction accepted by both target venues. Candidate ZIP
and manifest are reviewable; preparing them is not authorization to publish them.
''')
    put('preprints_mdpi_policy_audit.md','''# Venue and policy audit

Checked2026-10-29/30. [Official SI](https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt)
indexed text identifies Machine Learning and Its Application for Anomaly Detection,
deadline31December2026, guest editors Naeem Ayoub and Amira Mouakher. Direct page
fetch returned429; actual authenticated portal selection is NOT_VERIFIED.

[Preprints instructions](https://www.preprints.org/instructions-for-authors) were
read in full: all-author posting consent, durable DOI/withdrawal consequences,
first-page contacts, neutral publisher presentation, available data/rights and
Methods AI disclosure apply. An actual posting receipt/DOI is absent.
[MDPI ethics](https://www.mdpi.com/ethics) official indexed text requires AI use in
Methods and product details in Acknowledgments; direct429 limits live verification.
No unrecorded model versions are invented. The inspected official template ZIP
download succeeded; its MDPI class hash62979fe0... matches the preserved18June2025
class. Both venues share science and declaration facts, with formatting only differing.

Preprint uses standard article class without MDPI class/logo/issue footer. MDPI uses
the actual submit template with publisher-assigned fields empty and no custom
TDS-revision/prepared-author-pending/DOI-not-assigned headers or footers. Standard
class submit wording is template boilerplate and does not establish a portal receipt.
Current draft still explicitly lacks confirmed correspondence, funding evidence and
COI; those author-review statements must be resolved before final public submission.

Intended order remains approved preprint posting -> real DOI -> journal disclosure.
SUBMITTED_PREPRINT, POSTED_PREPRINT, SUBMITTED_MDPI remain false; DOI remains null.
''')
    put('reference_selection_audit.md','''# Reference verification

Added only literature used in the argument: Pang/Ruff anomaly reviews; Kapoor--Narayanan
sampling/leakage; Lipton etal. assumptions of label shift; LightGBM and IsolationForest
for the executed controls; Brody etal. for the actual GATv2 encoder. Primary author
preprints, NeurIPS proceedings, PMLR and publisher-registered Crossref metadata were
checked. `reference_verification.json` preserves DOI metadata for the journal/IF
items; official source URLs appear in BibTeX. Poursafaei and Saerens already existed
and are retained for the actual temporal-evaluation/prior-shift argument. No claim
that class-conditional invariance holds is borrowed from label-shift literature.
Unused older BibTeX entries do not produce uncited numbered references.
''')
    rows=[('Last-time selection','ACCEPT','Full raw lineage; reversed prevalence; title/claims narrowed'),
          ('Right-censoring cause','PARTIAL','Exact policy unknown; association only, no causal declaration'),
          ('Fixed horizon early warning','REJECT INTERPRETATION','7-day task executed; label time unknown'),
          ('Time input','ACCEPT','5seed GATv2 with/without branch; ranges and phase extent'),
          ('TGN naming','ACCEPT','Chain-level recurrent-state; exact state semantics'),
          ('One primary panel','ACCEPT VIA ALLOWED SEPARATION','Matched primary; independent archival Supplement'),
          ('GBDT/IF','ACCEPT','LightGBM and fixed-orientation unlabeled IF executed'),
          ('Modelwise calibration','ACCEPT','95 val-only calibrators; both raw and calibrated losses'),
          ('Chain/AP/budgets','ACCEPT','ETH/BSC intervals; Polygon support caveat; tie bounds'),
          ('Reusable contribution','ACCEPT','12-row main checklist plus vector decision tree'),
          ('AP randomization bug claim','PARTIAL','AP ties tested; score-swap null not monotone invariant'),
          ('Dependence','ACCEPT WITH LIMIT','19 hour blocks; wallet dependence unresolved'),
          ('Literature/title/style','ACCEPT','Verified sources; sampling-associated title; no QA jargon in main Methods/Results'),
          ('A16 declarations','APPLIED; HUMAN CLOSURE HOLD','Public original checked; same shared facts, real coding disclosure'),
          ('Publish/submit','HOLD','Exact-file author signatures, rights, official funding and platform proof absent')]
    put('claude_review_response_matrix.md','# Claude review response matrix\n\n|Issue|Disposition|Evidence/action|\n|---|---|---|\n'+'\n'.join('|'+ '|'.join(r)+'|' for r in rows))
    print('Public evidence reports written')


if __name__=='__main__':main()
