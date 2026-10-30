# A24 final PDF visual and numerical QA

Scientific release: https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a24-supplementary-2026-10-30-r1

Review method: actual95dpi PNG renders were inspected with view_image, every page via labelled contact sheets and critical control tables individually. This is agent visual QA, not a claim of new coauthor sign-off. Native text/x-y cell extraction and glyph bounds supplement the visual check.

## applsci

PDF SHA256: `6d07e6291923bdb7417f870eeb325c0fbe6118ea7b742ef863ed8a124d87a571`; pages: 25.

| Page | Observation | Result |
|---:|---|---|
| 1 | Title/abstract give benchmark motivation, DLG full name, frozen counts and matched-A24 LANL scope; author lines clear. | PASS |
| 2 | Introduction and support-aware contributions readable; adverse and historical-evidence limitations retained. | PASS |
| 3 | Related-work descriptions distinguish CoLA/CONAD/PREM/TAM/GADAM/ADA-GAD; citations resolve. | PASS |
| 4 | Detector families and DLG architecture definitions readable; no new-theorem claim. | PASS |
| 5 | Architecture figure and dense/Gram equations fit within page; caption legible. | PASS |
| 6 | Row residual/loss and AnomalyDAE all-pairs limitation remain readable; protocol begins cleanly. | PASS |
| 7 | Table1 raw/input/test populations and corrected crypto N/E are visible; all rightmost descriptions fit. | PASS |
| 8 | S1 n8/k7/Q29.0357/W0.6049 and all four views agree with source; label-use and zero-residual convention explicit. | PASS |
| 9 | Support matrix shows84/91primary plus7external; missing-run and OOM evidence are distinguished. | PASS |
| 10 | Main AP table complete; uncertainty links to Supplement; below-prevalence and crypto comparator limitations explicit. | PASS |
| 11 | Alert tables have readable model panels; paired-delta figure now uses sample SD and retains negative effects. | PASS |
| 12 | Paired-delta table uses sample SD; Friedman table and S4 tail interpretation agree with recomputed values. | PASS |
| 13 | Theoretical memory profiles are labelled illustrative; separate actual measurements are distinguished. | PASS |
| 14 | Decision table distinguishes feasibility/quality; LANL historical preprocessing limit and matched-A24 comparison are explicit. | PASS |
| 15 | Limitations and new supplementary findings visible; conclusion starts below; no lost paragraph at page continuation. | PASS |
| 16 | Measured resource table and conclusion continuation readable; actual r1 tag/commit/URL, data scope, AI and funding backmatter visible. | PASS |
| 17 | References1--22 visible; foundation DOI202609.0848 agrees with supplied citation and neutral URL. | PASS |
| 18 | Remaining references including IForest/ECOD fit; DOI/URL line breaks remain within page. | PASS |
| 19 | ROC panels and historical A5 controls complete; individually zoomed A5 header and all seven final ROC-AUC mean/SD cells readable; no ValF1 column. | PASS |
| 20 | Primary test-F1 panels complete; source-policy limitation and sample SD stated. | PASS |
| 21 | Supplement AP panels retain all primary means and show sample SD; unsupported cells remain N/A. | PASS |
| 22 | Feature-only, matched controls and alert tables complete; LANL raw-input qualification and P/R/lift headers visible. | PASS |
| 23 | Seedwise lift table fits; Polygon K5 and sample-SD uncertainty noted; no ratio-of-means substitution. | PASS |
| 24 | All14dataset split/prediction-unit rows visible; full observable fitting and label masks distinguished. | PASS |
| 25 | Complete-case membership table fits; S5 external-descriptive label scoped; final model names present. | PASS |

## preprints

PDF SHA256: `092369953dfb16da91110e21e3fa6529fb532c6923220c0f0b95a1fa512c365c`; pages: 28.

| Page | Observation | Result |
|---:|---|---|
| 1 | Publisher-neutral title/abstract/authors and ORCID footnotes fit; DLG expanded and matched-A24 LANL scope clear. | PASS |
| 2 | Introduction/benchmark contributions readable; fixed counts and evidence limits retained. | PASS |
| 3 | Related-work mechanisms and citation context readable without clipping. | PASS |
| 4 | Scaling/benchmark rigor and detector-family descriptions fit. | PASS |
| 5 | DLG formulation, architecture figure/caption and dense loss readable. | PASS |
| 6 | Gram identity, node loss and AnomalyDAE arithmetic boundary fit. | PASS |
| 7 | All Table1 populations, corrected crypto dimensions and final provenance column visible. | PASS |
| 8 | Protocol/statistics show n8/Q29.0357/W0.6049 and other views; label and numerical conventions readable. | PASS |
| 9 | Support figure and execution status distinctions complete. | PASS |
| 10 | Primary AP table and comparator/prevalence interpretation fit; sample-SD supplement link present. | PASS |
| 11 | Both alert-model panels and paired-delta table visible, with sample-SD definition. | PASS |
| 12 | Paired-delta SD figure, Friedman table and MC interpretation readable. | PASS |
| 13 | MC tails and illustrative resource profiles distinguished from measurements. | PASS |
| 14 | Decision table includes DGraph below-prevalence warning and qualified LANL comparison. | PASS |
| 15 | Historical/new experiment distinctions and LANL preprocessing mismatch explicit; adverse control results retained. | PASS |
| 16 | Four measured resource cases, conclusions and r1 data availability fit; long release URL wraps after xurl fix. | PASS |
| 17 | Actual AI-assisted development/orchestration/analysis scope visible; conflicts and first references readable. | PASS |
| 18 | Related-work reference URLs and bibliographic fields fit. | PASS |
| 19 | Remaining methodology/dataset references legible with safe URL wrapping. | PASS |
| 20 | Last references including IForest/ECOD complete; no unresolved citation markers. | PASS |
| 21 | Both primary ROC model panels visible with sample SD. | PASS |
| 22 | Both primary test-F1 panels visible; historical policy limitation retained. | PASS |
| 23 | Historical A5 control table individually zoomed; all seven ROC-AUC mean/SD last cells and header visible; F1 omitted. | PASS |
| 24 | Primary AP uncertainty panels complete, with unchanged means and sample SD. | PASS |
| 25 | Feature-only/controls/alerts fit; explicit LANL A24-only input match; adverse means retained. | PASS |
| 26 | Seedwise lift table complete, with discrete Polygon K5 caveat. | PASS |
| 27 | All14 split/prediction-unit rows fit, with validation/test roles stated. | PASS |
| 28 | Complete-case membership table and final model column fit; no extra inference for external LANL. | PASS |

Original A23 Table A3 is renumbered A5 after readable panel splitting. Its historical validation-F1 column is omitted; ROC-AUC is the actual last column. All seven last-column means/SDs are present at the expected PDF x/y positions in both versions. Two native last-cell deletion mutations and18 section/table source mutations were rejected.

No final overfull hbox or unresolved references/citations. The Preprints long release URL was repaired with xurl. Paired-delta Figure3 now uses sample SD, not archived SE bars. Mean-only main AP links to sample-SD supplementary panels.
