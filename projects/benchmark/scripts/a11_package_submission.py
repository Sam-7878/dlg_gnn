#!/usr/bin/env python3
"""Phase P0-G & Phase J: Packaging Submission Bundles."""
import shutil
import zipfile
from pathlib import Path

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
BENCH = ROOT / 'projects/benchmark'
PAPER = BENCH / 'paper/a11'
SUB = BENCH / 'submission'

PRE_DIR = SUB / 'preprints'
APP_DIR = SUB / 'applsci'
PRE_DIR.mkdir(parents=True, exist_ok=True)
APP_DIR.mkdir(parents=True, exist_ok=True)

# 1. Preprints.org Bundle
shutil.copy2(PAPER / 'DLG-Benchmark_A11_Preprints.tex', PRE_DIR / 'DLG-Benchmark_A11_Preprints.tex')
shutil.copy2(PAPER / 'DLG-Benchmark_A11_Preprints.pdf', PRE_DIR / 'DLG-Benchmark_A11_Preprints.pdf')

pre_zip = PRE_DIR / 'source_bundle.zip'
with zipfile.ZipFile(pre_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(PAPER / 'DLG-Benchmark_A11_Preprints.tex', arcname='DLG-Benchmark_A11_Preprints.tex')
    z.write(PAPER / 'references.bib', arcname='references.bib')
    for fig in (PAPER / 'figures').glob('*.*'):
        z.write(fig, arcname=f'figures/{fig.name}')

print(f'Created {pre_zip} ({pre_zip.stat().st_size} bytes)')

# 2. MDPI Applied Sciences Bundle
MDPI_PAPER = PAPER / 'mdpi'
shutil.copy2(MDPI_PAPER / 'DLG-Benchmark_A11_AppliedSciences.tex', APP_DIR / 'DLG-Benchmark_A11_AppliedSciences.tex')
shutil.copy2(MDPI_PAPER / 'DLG-Benchmark_A11_AppliedSciences.pdf', APP_DIR / 'DLG-Benchmark_A11_AppliedSciences.pdf')

app_zip = APP_DIR / 'source_bundle.zip'
with zipfile.ZipFile(app_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(MDPI_PAPER / 'DLG-Benchmark_A11_AppliedSciences.tex', arcname='DLG-Benchmark_A11_AppliedSciences.tex')
    z.write(MDPI_PAPER / 'references.bib', arcname='references.bib')
    for sty in MDPI_PAPER.glob('*.sty'):
        z.write(sty, arcname=sty.name)
    for def_file in (MDPI_PAPER / 'Definitions').rglob('*'):
        if def_file.is_file():
            rel = def_file.relative_to(MDPI_PAPER)
            z.write(def_file, arcname=str(rel).replace('\\', '/'))
    for fig in (MDPI_PAPER / 'figures').glob('*.*'):
        z.write(fig, arcname=f'figures/{fig.name}')

print(f'Created {app_zip} ({app_zip.stat().st_size} bytes)')
