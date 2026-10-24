import shutil
from pathlib import Path

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
BENCH = ROOT / 'projects/benchmark'

A11_PAPER = BENCH / 'paper/a11'
A12_PAPER = BENCH / 'paper/a12'
A12_MDPI = A12_PAPER / 'mdpi'
A12_FIGS = A12_PAPER / 'figures'

A12_PAPER.mkdir(parents=True, exist_ok=True)
A12_MDPI.mkdir(parents=True, exist_ok=True)
A12_FIGS.mkdir(parents=True, exist_ok=True)
(A12_MDPI / 'figures').mkdir(parents=True, exist_ok=True)

# Copy figures to mdpi/figures
for fig_file in A12_FIGS.glob('*'):
    shutil.copy2(fig_file, A12_MDPI / 'figures' / fig_file.name)

# Copy Definitions and sty from a11/mdpi to a12/mdpi
shutil.copytree(A11_PAPER / 'mdpi/Definitions', A12_MDPI / 'Definitions', dirs_exist_ok=True)
for sty in ['soul.sty', 'soul-ori.sty']:
    shutil.copy2(A11_PAPER / 'mdpi' / sty, A12_MDPI / sty)

# Read original references.bib
orig_bib = (A11_PAPER / 'references.bib').read_text(encoding='utf-8')

# Append the 3 verified Applied Sciences references
applied_sciences_refs = """
@article{dilini2026comprehensive,
  title        = {A Comprehensive Review on Graph-Based Anomaly Detection: Approaches for Intrusion Detection},
  author       = {Dilini, N. and Sun, N. and Miao, Y. and Moustafa, N.},
  journal      = {Applied Sciences},
  volume       = {16},
  number       = {4},
  pages        = {1906},
  year         = {2026},
  publisher    = {MDPI},
  doi          = {10.3390/app16041906}
}

@article{heidrich2026systematic,
  title        = {A Systematic Evaluation Method of Graph-Derived Signals for Tabular Machine Learning},
  author       = {Heidrich, M. and Heidemann, J. and Buchkremer, R. and {Wandosell Fern{\'a}ndez de Bobadilla}, G.},
  journal      = {Applied Sciences},
  volume       = {16},
  number       = {5},
  pages        = {2624},
  year         = {2026},
  publisher    = {MDPI},
  doi          = {10.3390/app16052624}
}

@article{jiang2025mvcg,
  title        = {{MVCG-SPS}: A Multi-View Contrastive Graph Neural Network for Smart Ponzi Scheme Detection},
  author       = {Jiang, X. and Tsai, W.-T.},
  journal      = {Applied Sciences},
  volume       = {15},
  number       = {6},
  pages        = {3281},
  year         = {2025},
  publisher    = {MDPI},
  doi          = {10.3390/app15063281}
}
"""

a12_bib = orig_bib.strip() + "\n" + applied_sciences_refs.strip() + "\n"

(A12_PAPER / 'references.bib').write_text(a12_bib, encoding='utf-8')
(A12_MDPI / 'references.bib').write_text(a12_bib, encoding='utf-8')
print("Successfully set up A12 directory, figures, Definitions, and references.bib with 3 Applied Sciences papers.")
