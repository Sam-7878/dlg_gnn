# StreamMC paper workspace

`current/r01/` is the active DLG-SelectiveStream R01 scientific content,
neutral preprint/free-format journal wrappers, bibliography, generated tables,
figures and standalone supplement. Root submission/supplement entry points are
compatibility wrappers pointing to that content after closure. The original
reviewed directory is preserved in the immutable R01 input snapshot ZIP.
`current/audit_latex.py` delegates to actual clean TeX/BibTeX validation, not
the historical superficial string checks.
`history/` is a Git-ignored local archive of the earlier `_41_01_Stream`
draft, presentation and figures; it is not the current paper. The StreamMC
`verify` facade recomputes raw prediction metrics and exact paired statistics;
private `paper` compilation and all-page visual checks remain separate.
