import json
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCH_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m2" / "architecture"
SRC_DIR = REPO_ROOT / "src"

import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gog_fraud.models.pygod.dlg_base import DLGBase
from gog_fraud.models.pygod.dlg_full_base import DLGFullBase

def test_architecture_source_trace_json():
    trace_path = ARCH_DIR / "dlg_architecture_source_trace.json"
    assert trace_path.exists(), f"Missing trace: {trace_path}"
    
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    formulas = data["formulas"]
    assert "129*F + 16705" in formulas["DLG-Base"]["total_active_formula"]
    assert "129*F + 12480" in formulas["DLG-Aug"]["stage2_l2_global"]["active_formula"]
    assert "258*F + 16704" in formulas["DLG-Aug"]["total_pipeline_formula"]

def test_pytorch_parameter_exact_formulas():
    test_dims = [13, 17, 67, 165, 300, 500, 602, 767, 1433, 3703]
    for F_dim in test_dims:
        # DLGBase
        base = DLGBase(in_dim=F_dim, hid_dim=64, num_layers=4)
        base_p = sum(p.numel() for p in base.parameters())
        expected_base = 129 * F_dim + 16705
        assert base_p == expected_base, f"DLG-Base param mismatch for F={F_dim}: got {base_p}, expected {expected_base}"
        
        # DLGBase scalar gate
        assert isinstance(base.alpha, torch.nn.Parameter)
        assert base.alpha.numel() == 1
        
        # DLGFullBase (Stage 2)
        full_base = DLGFullBase(in_dim=F_dim + 64, orig_dim=F_dim, hid_dim=64, num_layers=4)
        full_p = sum(p.numel() for p in full_base.parameters())
        expected_full = 129 * F_dim + 12480
        assert full_p == expected_full, f"DLG-Aug Stage 2 param mismatch for F={F_dim}: got {full_p}, expected {expected_full}"
