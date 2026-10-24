#!/usr/bin/env python3
"""A13 Setup and Inventory Script.
Fulfills Gate A13-G1:
- Freezes and inventories actual A12 artifacts and submission files.
- Computes SHA-256 directly from bytes on disk.
- Generates publication/a13/a12_input_sha256_manifest.csv and artifact_inventory_a13.csv.
- Sets up paper/a13/ and publication/a13/ environments.
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
BENCH = ROOT / 'projects/benchmark'

A12_PAPER = BENCH / 'paper/a12'
A12_MDPI = A12_PAPER / 'mdpi'
A12_PUB = BENCH / 'publication/a12'
A12_SUB_PRE = BENCH / 'submission/preprints'
A12_SUB_MDPI = BENCH / 'submission/applsci'

A13_PAPER = BENCH / 'paper/a13'
A13_MDPI = A13_PAPER / 'mdpi'
A13_FIGS = A13_PAPER / 'figures'
A13_TABLES = A13_PAPER / 'tables'
A13_PUB = BENCH / 'publication/a13'

# 1. Create A13 directories
A13_PAPER.mkdir(parents=True, exist_ok=True)
A13_MDPI.mkdir(parents=True, exist_ok=True)
A13_FIGS.mkdir(parents=True, exist_ok=True)
(A13_MDPI / 'figures').mkdir(parents=True, exist_ok=True)
A13_TABLES.mkdir(parents=True, exist_ok=True)
A13_PUB.mkdir(parents=True, exist_ok=True)

# Copy Definitions and styles to A13
shutil.copytree(A12_MDPI / 'Definitions', A13_MDPI / 'Definitions', dirs_exist_ok=True)
for sty in ['soul.sty', 'soul-ori.sty']:
    if (A12_MDPI / sty).exists():
        shutil.copy2(A12_MDPI / sty, A13_MDPI / sty)
if (A12_PAPER / 'references.bib').exists():
    shutil.copy2(A12_PAPER / 'references.bib', A13_PAPER / 'references.bib')
    shutil.copy2(A12_PAPER / 'references.bib', A13_MDPI / 'references.bib')

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# 2. Inventory A12 Artifacts
a12_files_to_hash = [
    # Paper A12 files
    ('paper_preprints_tex', A12_PAPER / 'DLG-Benchmark_A12_Preprints.tex'),
    ('paper_preprints_pdf', A12_PAPER / 'DLG-Benchmark_A12_Preprints.pdf'),
    ('paper_references_bib', A12_PAPER / 'references.bib'),
    ('paper_fig_conditional_delta', A12_PAPER / 'figures/fig_conditional_delta.pdf'),
    ('paper_fig_dlg_architecture', A12_PAPER / 'figures/fig_dlg_architecture.pdf'),
    ('paper_fig_support_matrix', A12_PAPER / 'figures/fig_support_matrix.pdf'),
    ('paper_table_dataset_properties', A12_PAPER / 'tables/table_dataset_properties.tex'),
    ('paper_table_friedman', A12_PAPER / 'tables/table_friedman_s1_s4.tex'),
    ('paper_mdpi_tex', A12_MDPI / 'DLG-Benchmark_A12_AppliedSciences.tex'),
    ('paper_mdpi_pdf', A12_MDPI / 'DLG-Benchmark_A12_AppliedSciences.pdf'),
    # Publication A12
    ('pub_dataset_identity_csv', A12_PUB / 'dataset_identity_reconciliation.csv'),
    ('pub_dataset_identity_md', A12_PUB / 'dataset_identity_reconciliation.md'),
    ('pub_claims_csv', A12_PUB / 'claims_to_evidence_a12.csv'),
    ('pub_numeric_gate_json', A12_PUB / 'A12_MANUSCRIPT_NUMERIC_GATE.json'),
    # Analysis & Diagnostics
    ('analysis_statistics_a12', BENCH / 'analysis/statistics_a12.json'),
    ('analysis_asymptotic_mc_csv', BENCH / 'analysis/a12_asymptotic_mc_comparison.csv'),
    ('analysis_permutation_unit_tests', BENCH / 'analysis/a12_permutation_unit_tests.md'),
    ('diagnostics_score_alignment_json', BENCH / 'diagnostics/a12_score_alignment_empirical.json'),
    ('diagnostics_hub_bias_hypothesis', BENCH / 'diagnostics/a12_hub_bias_hypothesis.md'),
    # Submissions
    ('sub_preprints_tex', A12_SUB_PRE / 'DLG-Benchmark_A12_Preprints.tex'),
    ('sub_preprints_pdf', A12_SUB_PRE / 'DLG-Benchmark_A12_Preprints.pdf'),
    ('sub_preprints_zip', A12_SUB_PRE / 'source_bundle.zip'),
    ('sub_preprints_gate_report', A12_SUB_PRE / 'PREPRINT_GATE_REPORT.md'),
    ('sub_mdpi_tex', A12_SUB_MDPI / 'DLG-Benchmark_A12_AppliedSciences.tex'),
    ('sub_mdpi_pdf', A12_SUB_MDPI / 'DLG-Benchmark_A12_AppliedSciences.pdf'),
    ('sub_mdpi_zip', A12_SUB_MDPI / 'source_bundle.zip'),
    ('sub_mdpi_cover_letter', A12_SUB_MDPI / 'COVER_LETTER.md'),
    ('sub_mdpi_data_availability', A12_SUB_MDPI / 'DATA_AVAILABILITY.md'),
    ('sub_mdpi_submission_gate_report', A12_SUB_MDPI / 'SUBMISSION_GATE_REPORT.md'),
    # Scripts
    ('script_dataset_contract', BENCH / 'scripts/a12_dataset_contract_gate.py'),
    ('script_validate_numbers', BENCH / 'scripts/a12_validate_manuscript_numbers.py'),
    ('script_audit_score', BENCH / 'scripts/a12_audit_score_direction.py'),
    ('script_audit_friedman', BENCH / 'scripts/a12_audit_friedman_statistics.py'),
    ('script_generate_figures', BENCH / 'scripts/a12_generate_figures.py'),
    ('script_build_manuscripts', BENCH / 'scripts/a12_build_manuscripts.py'),
    ('script_compile_manuscripts', BENCH / 'scripts/a12_compile_manuscripts.py'),
    ('script_package_submission', BENCH / 'scripts/a12_package_submission.py'),
    # Execution Report
    ('execution_report_a12', ROOT / 'docs/work_reports/benchmark/237_A12/DLG_Benchmark_A12_Execution_Report_2026-10-24.md')
]

manifest_rows = []
for label, p in a12_files_to_hash:
    if p.exists():
        sha = compute_sha256(p)
        size = p.stat().st_size
        manifest_rows.append({
            'artifact_id': label,
            'relpath': str(p.relative_to(ROOT)),
            'exists': True,
            'size_bytes': size,
            'sha256': sha
        })
    else:
        manifest_rows.append({
            'artifact_id': label,
            'relpath': str(p.relative_to(ROOT)),
            'exists': False,
            'size_bytes': 0,
            'sha256': 'FILE_NOT_FOUND'
        })

manifest_csv_path = A13_PUB / 'a12_input_sha256_manifest.csv'
with open(manifest_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['artifact_id', 'relpath', 'exists', 'size_bytes', 'sha256'])
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Wrote {manifest_csv_path} with {len(manifest_rows)} entries.")

# 3. Write artifact_inventory_a13.csv
inv_rows = [
    {
        'phase': 'A13_FREEZE',
        'artifact': 'a12_input_sha256_manifest.csv',
        'path': 'projects/benchmark/publication/a13/a12_input_sha256_manifest.csv',
        'status': 'VERIFIED_ON_DISK',
        'description': 'SHA-256 manifest of all A12 inputs recomputed from actual bytes.'
    },
    {
        'phase': 'A13_DIAGNOSTICS',
        'artifact': 'score_availability_by_campaign.csv',
        'path': 'projects/benchmark/publication/a13/score_availability_by_campaign.csv',
        'status': 'TO_BE_GENERATED',
        'description': 'Truthful breakdown of which campaigns have per-node score arrays vs not stored.'
    },
    {
        'phase': 'A13_VALIDATION',
        'artifact': 'manuscript_numeric_contract.json',
        'path': 'projects/benchmark/publication/a13/manuscript_numeric_contract.json',
        'status': 'TO_BE_GENERATED',
        'description': 'Literal parsing contract for both Preprints and MDPI .tex files.'
    },
    {
        'phase': 'A13_VALIDATION',
        'artifact': 'manuscript_numeric_validation.json',
        'path': 'projects/benchmark/publication/a13/manuscript_numeric_validation.json',
        'status': 'TO_BE_GENERATED',
        'description': 'Literal validation outcome parsing actual .tex files and asserting against registry.'
    },
    {
        'phase': 'A13_MANUSCRIPTS',
        'artifact': 'DLG-Benchmark_A13_Preprints.tex',
        'path': 'projects/benchmark/paper/a13/DLG-Benchmark_A13_Preprints.tex',
        'status': 'TO_BE_GENERATED',
        'description': 'Publisher-neutral A13 preprint manuscript with corrected causal claims, memory, and hardware.'
    },
    {
        'phase': 'A13_MANUSCRIPTS',
        'artifact': 'DLG-Benchmark_A13_AppliedSciences.tex',
        'path': 'projects/benchmark/paper/a13/mdpi/DLG-Benchmark_A13_AppliedSciences.tex',
        'status': 'TO_BE_GENERATED',
        'description': 'MDPI Applied Sciences A13 manuscript with corrected causal claims, memory, and hardware.'
    },
    {
        'phase': 'A13_FIGURES',
        'artifact': 'fig3_paired_delta_data.csv',
        'path': 'projects/benchmark/publication/a13/fig3_paired_delta_data.csv',
        'status': 'TO_BE_GENERATED',
        'description': 'Explicitly seed-paired DLG-Aug vs DLG-Base delta values and paired SE.'
    },
    {
        'phase': 'A13_RELEASE',
        'artifact': 'release_candidate_manifest.json',
        'path': 'projects/benchmark/publication/a13/release_candidate_manifest.json',
        'status': 'TO_BE_GENERATED',
        'description': 'Full A13 release candidate manifest with all file hashes and git commit binding.'
    }
]

inv_csv_path = A13_PUB / 'artifact_inventory_a13.csv'
with open(inv_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['phase', 'artifact', 'path', 'status', 'description'])
    writer.writeheader()
    writer.writerows(inv_rows)

print(f"Wrote {inv_csv_path} with {len(inv_rows)} entries.")
print("A13 Setup and Initial Inventory Complete.")
