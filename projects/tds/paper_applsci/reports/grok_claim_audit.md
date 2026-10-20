# Grok review claim audit

The executable work order in `docs/work_reports/tds/314_grok_review/` governs this revision. The directory is named `314_grok_review`, not `314_grok_review3`.

| Review point | Evidence-based disposition |
|---|---|
| Architecture novelty | Reframe as an evaluation methodology; GIN is a reference encoder. |
| Incomparable losses | Confirmed. Historical BCE also has inverse-frequency `pos_weight`; it is not unweighted BCE. Supplemental matched 2×2 matrix covers GIN/GraphSAGE and weighted BCE/focal. |
| Sample meaning | Confirmed manuscript error: one graph per token/contract at its last valid recorded transaction time, not 24,316 transactions. |
| Three input features | Full-history log in/out/total degree, not transaction count; features are computed before the edge cap. Correct this additional source-level finding. |
| Local k-hop parameter | No k-hop extraction exists in the packed builder. All contract endpoints are considered, then 128 recent edges are retained. Remove the unsupported k-hop formulation. |
| GIN batch normalization | No BatchNorm module exists in historical model source or checkpoint. Remove the manuscript's batch-normalization claim. |
| TGAT time encoding | Normalized absolute contract cutoff is harmonically encoded; not relative edge-time differences. Describe the actual controlled implementation. |
| TGN online behavior | Per-chain memory is reset at each train epoch and each validation/test pass. Prediction uses previous state; allowed current graph/time information updates state for the next sample. No held-out labels enter state updates or gradients. |
| Timestamp ties | 3,910/24,316 samples (16.08%) share a cutoff timestamp. Source has block number and transaction hash, but no transaction index. Hash sorting is deterministic, not a verified execution order. Strict earlier-time source reconstruction and checkpoint sensitivity are provided separately. |
| p-values must match after temperature scaling | Rejected as a general mathematical assertion: observed ranking invariance does not imply identical score-swap nulls. Audit uses the same frozen MC vectors and IDs. |
| Statistical implementation | Additional real bug found: historical AP ignored score ties, which bootstrap duplication creates. Recompute all seven protocols using threshold-grouped AP checked against sklearn; preserve original results. |
| Calibration interpretation | All fitted temperatures are below one; logits sharpen. Inspect positive-probability reliability bins; use mixed miscalibration, rather than a universal confidence-direction claim. |
| Temporal-model fragility explanation | Unsupported mechanistic attribution. Replace with benchmark- and implementation-specific results. |
| Counterfactual leakage claim | No leakage stress test supports its magnitude. Replace with a construction-level information-admissibility statement. |
| Operational uncertainty monitoring | Remove unused variance and epistemic-monitoring recommendation. |
| Training convergence | Original selected epochs exist but full trajectories were not retained. Any inferred stop epochs are identified as inferred. New supplemental runs record full histories and use a predeclared extended ceiling. |
| Figure and citation refresh | Official current template downloaded; original source was already modular and includes four figures. Regenerate vectors and add SD bars/support; verify curated recent sources. |

All findings are scoped to this manuscript. Historical Gate M v8 is retained as a historical evidence-completeness decision, not treated as proof of the stronger new submission requirements.
