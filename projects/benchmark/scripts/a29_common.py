"""A29 shared helpers.

Rules carried over from A24 (see A29 work order section 6):
  * the 455 primary runs, the 91-pair support matrix and primary ranks are frozen;
  * every new result is an *extension* and is written under results/a29 only;
  * labels never enter fit(); validation labels may select thresholds/configs,
    test labels are read once, after scores are fixed;
  * missing measurements are recorded as missing, never as zero.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a24_common import ROOT, sha, dump, rows, writecsv  # noqa: E402,F401

RESULTS = ROOT / 'projects/benchmark/results/a29'          # public, small, tracked
LOCAL = ROOT / 'local_storage/benchmark/a29'                # internal: scores, logs
PUB24 = ROOT / 'projects/benchmark/publication/a24'
REG = ROOT / 'projects/benchmark/publication/a16/canonical_selected_run_registry.csv'
GPU_3090 = 'GPU-ab53069a-3217-ee09-80f8-748b5fdbd156'
SEEDS = [42, 43, 44, 45, 46]
MODELS7 = ['DOMINANT', 'AnomalyDAE', 'CoLA', 'GADNR', 'OCGNN', 'DLG-Base', 'DLG-Aug']

# Manuscript release (A30): text, build and sign-off only; the A29 data in results/a29 is unchanged.
# FREEZE_DATE fixes the PDF date (SOURCE_DATE_EPOCH), zip timestamps and the evidence tag name; change it on the day of a new freeze.
REV = 'a30'
FREEZE_DATE = '2026-11-26'
EVIDENCE_TAG = f'benchmark-{REV}-evidence-{FREEZE_DATE}-r1'
PAPER_OUT = ROOT / 'projects/benchmark/paper' / REV            # built manuscripts of this release
PUB_OUT = ROOT / 'projects/benchmark/publication' / REV        # QA, hash lists, sign-off of this release

for p in (RESULTS, LOCAL):
    p.mkdir(parents=True, exist_ok=True)


def pin_3090() -> None:
    """Pin the process to the RTX 3090 by UUID before torch initialises CUDA.

    A29_GPU_UUID may name another device for accuracy-only studies (never for timing/memory claims).
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('A29_GPU_UUID', GPU_3090)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'local_storage/benchmark/a24/deps'))


def environment() -> dict:
    import torch
    import torch_geometric
    import pygod
    import numpy
    out = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'pyg': torch_geometric.__version__,
        'pygod': pygod.__version__, 'numpy': numpy.__version__,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'host': platform.node(), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    if torch.cuda.is_available():
        out['gpu_name'] = torch.cuda.get_device_name(0)
        out['gpu_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
    return out


def read_json(p):
    return json.loads(Path(p).read_text())
