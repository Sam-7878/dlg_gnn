# P1 Audit Report 04: Preprints.org Bundle Compilation Audit

## 1. Executive Summary
- **Objective**: Verify that the self-contained Preprints.org archive compiles cleanly without network dependencies, external packages, or TeX compilation errors.
- **Target Archive**: `publication/benchmark/preprints/DLG_Benchmark_Preprints_Submission.zip` (769.3 KB).
- **Compiled PDF**: `publication/benchmark/preprints/DLG-Benchmark-Preprint.pdf` (487.0 KB, 32 pages).
- **Status**: **100% PASSED** (4-pass compilation clean with 0 errors).

---

## 2. Compilation Log Summary

```text
Compilation Sequence:
  Pass 1: pdflatex -interaction=nonstopmode -halt-on-error DLG-Benchmark-Preprint.tex (Exit Code 0)
  Pass 2: bibtex DLG-Benchmark-Preprint (Exit Code 0, 48 references resolved)
  Pass 3: pdflatex -interaction=nonstopmode -halt-on-error DLG-Benchmark-Preprint.tex (Exit Code 0)
  Pass 4: pdflatex -interaction=nonstopmode -halt-on-error DLG-Benchmark-Preprint.tex (Exit Code 0)
Final Page Count: 32
Output PDF Size: 487.0 KB
PDF SHA-256: 6189771e004536702d59a2fad112df3c1285d146010babacd860187cc4b4b839
```

---

## 3. Bundle Asset Inventory
The submission zip includes:
- `DLG-Benchmark-Preprint.tex` (Clean publisher-neutral LaTeX manuscript)
- `references.bib` (Complete bibliography with DOIs)
- `graphical_abstract.png` (300 DPI high-resolution graphical abstract)
- `generated/appendix_performance_tables.tex`
- `generated/table_capacity_controls.tex`
- `generated/table_capacity_controls_m3.tex`
- `generated/table_capacity_controls_paired_deltas.tex`
- `generated/table_dlg_architecture_budget.tex`

All inputs resolve relatively from the root of the unpacked bundle.
