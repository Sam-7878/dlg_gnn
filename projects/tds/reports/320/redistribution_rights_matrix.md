# Redistribution Rights and Privacy Matrix — R320

Assessment of upstream licenses, redistribution boundaries, and privacy protection for the public release.

| Asset | Upstream Source | Upstream License | Distribution Category | Justification & Safeguards |
|---|---|---|---|---|
| Evaluation scripts & model code | Original work (`dlg_gnn`) | MIT / Apache 2.0 compatible | `allowed_public` | Fully auditable code with no proprietary dependencies. |
| Aggregated performance tables & CSVs | Original calculations | Open Data | `allowed_public` | High-level summary metrics containing no raw address identities. |
| Raw contract graph tensors (`.pt`) | Derived from GoG | CC BY-NC-SA 4.0 | `restricted_local` / `rebuild_recipe` | GoG requires non-commercial attribution. We provide deterministic rebuild recipes rather than redistributing raw data dumps. |
| Raw transaction hashes and wallet addresses | Blockchain public ledgers | Public ledger | `excluded_from_payload` | To protect pseudonymity and avoid linkable deanonymization, raw addresses and hashes are omitted from public release payloads. |
| Model checkpoints (`.pt`) | Local training runs | Original work | `restricted_local` | Available upon reasonable request; excluded from minimal replay bundle to keep repository lightweight. |
| Manuscript source & LaTeX files | Authors | Author copyright | `submission_bundle_only` | Managed in submission archives; published via Preprints.org and MDPI open access. |
