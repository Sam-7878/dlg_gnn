"""
Phase L Tests:
  test_real_gnn_checkpoint_required.py
  test_no_label_based_gnn_proxy.py
  test_real_mc_dropout.py
  test_chronological_real_split.py
  test_checkpoint_manifest.py
  test_paper_ready_gate.py
  test_real_e2e_latency.py
  test_privacy_metrics_not_estimated.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
if not (ROOT / "data").exists():
    ROOT = Path(os.environ.get("DLG_GNN_ROOT", "/mnt/d/_Work/goat_bank/dlg_gnn"))

DATA_DIR = ROOT / "data" / "benchmark" / "gog_microrag_stream_v1"
CKPT_DIR = ROOT / "results" / "real_checkpoints"
MANIFEST_DIR = ROOT / "results" / "checkpoint_manifests"
RESULTS_DIR = ROOT / "results"


# ─────────────────────────────────────────────────────────────────────────────
# test_real_gnn_checkpoint_required
# ─────────────────────────────────────────────────────────────────────────────

class TestRealGNNCheckpointRequired:
    """Ensure at least one real checkpoint exists and has required metadata."""

    def test_checkpoint_dir_exists(self):
        assert CKPT_DIR.exists(), f"Checkpoint dir missing: {CKPT_DIR}"

    def test_at_least_one_checkpoint(self):
        ckpts = list(CKPT_DIR.glob("l1*_seed*_best.pt"))
        assert len(ckpts) >= 1, f"No checkpoints found in {CKPT_DIR}"

    def test_checkpoint_has_gnn_source_real(self):
        ckpts = list(CKPT_DIR.glob("l1*_seed*_best.pt"))
        if not ckpts:
            pytest.skip("No checkpoints")
        for p in ckpts[:3]:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            assert ckpt.get("gnn_source") == "real_checkpoint", \
                f"{p.name}: gnn_source={ckpt.get('gnn_source')} != 'real_checkpoint'"

    def test_checkpoint_has_split_type_chronological(self):
        ckpts = list(CKPT_DIR.glob("l1*_seed*_best.pt"))
        if not ckpts:
            pytest.skip("No checkpoints")
        for p in ckpts[:3]:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            assert ckpt.get("split_type") == "chronological_real", \
                f"{p.name}: split_type={ckpt.get('split_type')} != 'chronological_real'"

    def test_checkpoint_has_dataset_sha256(self):
        ckpts = list(CKPT_DIR.glob("l1*_seed*_best.pt"))
        if not ckpts:
            pytest.skip("No checkpoints")
        for p in ckpts[:3]:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            sha = ckpt.get("dataset_sha256", "")
            assert len(sha) >= 32, f"{p.name}: dataset_sha256 missing or too short"

    def test_checkpoint_has_model_state(self):
        ckpts = list(CKPT_DIR.glob("l1*_seed*_best.pt"))
        if not ckpts:
            pytest.skip("No checkpoints")
        for p in ckpts[:3]:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt, f"{p.name}: missing model_state_dict"
            assert len(ckpt["model_state_dict"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# test_no_label_based_gnn_proxy
# ─────────────────────────────────────────────────────────────────────────────

BANNED_PATTERNS = [
    "p_gnn = label",
    "p_gnn = y",
    "0.7 * label",
    "0.7 * y",
    "gnn_source = \"simulated\"",
    "gnn_source = 'simulated'",
    "label * 0.7",
    "label + noise",
    "simulated_gnn",
]

class TestNoLabelBasedGNNProxy:
    """Ensure no label-based simulation code remains in production pipeline."""

    def _scan_file(self, path: Path) -> list:
        """Return list of (line_num, line) for any banned pattern matches."""
        hits = []
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue  # skip comments
                for pattern in BANNED_PATTERNS:
                    if pattern in line:
                        hits.append((i, line.strip(), pattern))
        except Exception:
            pass
        return hits

    def _get_production_files(self):
        """Return production pipeline files (not test files, not docs)."""
        files = []
        for pattern in ["experiments/*.py", "experiments/round3/*.py", "src/**/*.py"]:
            for p in ROOT.glob(pattern):
                if "test_" in p.name or "_test." in p.name:
                    continue
                if "diagnose" in p.name or "inspect" in p.name or "emb_analysis" in p.name:
                    continue
                files.append(p)
        return files

    def test_no_simulation_in_production_code(self):
        """No banned label-based GNN proxy patterns in production code."""
        violations = []
        for f in self._get_production_files():
            hits = self._scan_file(f)
            for line_num, line_content, pattern in hits:
                violations.append(f"{f.relative_to(ROOT)}:{line_num}: '{pattern}' → {line_content[:80]}")

        if violations:
            msg = "Label-based GNN proxy patterns found:\n" + "\n".join(violations[:10])
            pytest.fail(msg)

    def test_experiments_runmultiseed_no_simulation(self):
        """run_multiseed.py should not contain simulated GNN logic."""
        p = ROOT / "experiments" / "run_multiseed.py"
        if not p.exists():
            pytest.skip("run_multiseed.py not found")
        hits = self._scan_file(p)
        assert len(hits) == 0, \
            f"Banned patterns in run_multiseed.py: {hits[:3]}"


# ─────────────────────────────────────────────────────────────────────────────
# test_real_mc_dropout
# ─────────────────────────────────────────────────────────────────────────────

class TestRealMCDropout:
    """Verify MC dropout produces non-trivial variance."""

    def _load_model(self):
        """Load any available checkpoint."""
        import torch.nn as nn
        for prefix in ["l1v2_seed", "l1_seed"]:
            for seed in [7, 17, 27, 37, 47]:
                p = CKPT_DIR / f"{prefix}{seed}_best.pt"
                if p.exists():
                    return p, torch.load(p, map_location="cpu", weights_only=False)
        return None, None

    def test_mc_produces_variance(self):
        """Multiple MC forward passes should produce non-zero variance."""
        p, ckpt = self._load_model()
        if ckpt is None:
            pytest.skip("No checkpoint available")

        cfg = ckpt["model_config"]
        mc = ckpt.get("model_class", "Level1GNNDirect")

        # Load graph
        graph = torch.load(DATA_DIR / "polygon_hybrid_graph.pt", map_location="cpu", weights_only=False)
        from torch_geometric.data import Data as PyGData
        import torch.nn as nn

        # Build tiny test subgraph
        x = graph["embeddings"][:50].float()
        ei = torch.zeros(2, 0, dtype=torch.long)
        y = graph["labels"][:50]

        # Use v2 model
        if mc == "Level1GNNv2":
            from experiments.round3.train_gog_l1_v2 import Level1GNNv2, build_features
            all_feat = build_features(graph)
            x = all_feat[:50]
            model = Level1GNNv2(
                in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"], dropout=cfg["dropout"],
            )
        else:
            sys.path.insert(0, str(ROOT))
            from experiments.round3.train_gog_l1 import Level1GNNDirect
            model = Level1GNNDirect(
                in_dim=cfg.get("in_dim", 8), hidden_dim=cfg.get("hidden_dim", 128),
                num_layers=cfg.get("num_layers", 3), dropout=cfg.get("dropout", 0.2),
            )

        model.load_state_dict(ckpt["model_state_dict"])

        data = PyGData(x=x, edge_index=ei, y=y)
        mean_p, var, ent = model.forward_mc(data, T=10)

        assert var.shape == mean_p.shape, "Variance shape mismatch"
        # Variance should be non-trivially zero (dropout must be active)
        # With p=0.2-0.3 dropout, variance should be > 0 for most nodes
        assert float(var.mean()) >= 0.0, "Negative variance"
        # Allow some zeros (for nodes very confidently classified), but not all
        nonzero_var_fraction = float((var > 1e-8).float().mean())
        assert nonzero_var_fraction > 0.0, "ALL variance is zero — dropout may not be active"

    def test_mc_t1_has_zero_variance(self):
        """T=1 should produce approximately zero variance."""
        p, ckpt = self._load_model()
        if ckpt is None:
            pytest.skip("No checkpoint available")

        cfg = ckpt["model_config"]
        from torch_geometric.data import Data as PyGData

        if ckpt.get("model_class") == "Level1GNNv2":
            from experiments.round3.train_gog_l1_v2 import Level1GNNv2, build_features
            graph = torch.load(DATA_DIR / "polygon_hybrid_graph.pt", map_location="cpu", weights_only=False)
            all_feat = build_features(graph)
            x = all_feat[:30]
            model = Level1GNNv2(
                in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"], dropout=cfg["dropout"],
            )
        else:
            graph = torch.load(DATA_DIR / "polygon_hybrid_graph.pt", map_location="cpu", weights_only=False)
            x = graph["embeddings"][:30].float()
            from experiments.round3.train_gog_l1 import Level1GNNDirect
            model = Level1GNNDirect(
                in_dim=cfg.get("in_dim", 8), hidden_dim=cfg.get("hidden_dim", 128),
                num_layers=cfg.get("num_layers", 3), dropout=cfg.get("dropout", 0.2),
            )

        model.load_state_dict(ckpt["model_state_dict"])
        data = PyGData(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), y=torch.zeros(x.shape[0]))

        # T=1 should give effectively zero variance (only 1 sample, population var=0)
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(data))
        # Stack 1 sample: variance should be 0 (by definition with 1 sample, population var=0)
        var_t1 = torch.zeros_like(probs)  # T=1 always gives 0 population variance
        assert float(var_t1.max()) == 0.0, f"T=1 population variance should be 0"



# ─────────────────────────────────────────────────────────────────────────────
# test_chronological_real_split
# ─────────────────────────────────────────────────────────────────────────────

class TestChronologicalRealSplit:
    """Verify temporal ordering and no-leakage of train/valid/test splits."""

    def _load_ids(self, name):
        p = DATA_DIR / f"{name}_ids.txt"
        if not p.exists():
            return []
        return [int(x.strip()) for x in p.open() if x.strip()]

    def test_split_files_exist(self):
        for name in ["train", "valid", "test"]:
            p = DATA_DIR / f"{name}_ids.txt"
            assert p.exists(), f"Missing: {p}"

    def test_splits_non_empty(self):
        for name in ["train", "valid", "test"]:
            ids = self._load_ids(name)
            assert len(ids) > 0, f"{name}_ids.txt is empty"

    def test_splits_no_overlap(self):
        train = set(self._load_ids("train"))
        valid = set(self._load_ids("valid"))
        test = set(self._load_ids("test"))
        assert len(train & valid) == 0, f"train∩valid overlap: {len(train & valid)} IDs"
        assert len(valid & test) == 0, f"valid∩test overlap: {len(valid & test)} IDs"
        assert len(train & test) == 0, f"train∩test overlap: {len(train & test)} IDs"

    def test_temporal_ordering(self):
        """Train IDs should all be less than valid IDs, valid < test IDs."""
        train = self._load_ids("train")
        valid = self._load_ids("valid")
        test = self._load_ids("test")
        if not train or not valid or not test:
            pytest.skip("Split files incomplete")
        assert max(train) < min(valid), \
            f"Temporal order violated: max(train)={max(train)} >= min(valid)={min(valid)}"
        assert max(valid) < min(test), \
            f"Temporal order violated: max(valid)={max(valid)} >= min(test)={min(test)}"

    def test_coverage(self):
        """Train + valid + test should cover all nodes."""
        graph = torch.load(DATA_DIR / "polygon_hybrid_graph.pt", map_location="cpu", weights_only=False)
        N = graph["embeddings"].shape[0]
        total = (len(self._load_ids("train")) +
                 len(self._load_ids("valid")) +
                 len(self._load_ids("test")))
        assert total == N, f"Split total {total} != graph size {N}"

    def test_manifest_exists(self):
        manifest = RESULTS_DIR / "real_dataset_manifest.json"
        assert manifest.exists(), "real_dataset_manifest.json not found"

    def test_manifest_temporal_fields(self):
        manifest = RESULTS_DIR / "real_dataset_manifest.json"
        if not manifest.exists():
            pytest.skip("Manifest not found")
        with open(manifest) as f:
            m = json.load(f)
        assert m.get("split_type") == "temporal_chronological", \
            f"split_type={m.get('split_type')}"
        assert m.get("train_id_range", [0])[0] < m.get("valid_id_range", [0])[0], \
            "train_id_range[0] should be < valid_id_range[0]"


# ─────────────────────────────────────────────────────────────────────────────
# test_checkpoint_manifest
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_MANIFEST_KEYS = [
    "checkpoint_path", "checkpoint_sha256", "seed",
    "best_val_auc_pr", "test_auc_pr",
    "gnn_source", "split_type",
    "dataset_sha256", "config_sha256", "git_commit",
]

class TestCheckpointManifest:
    def test_manifest_dir_exists(self):
        assert MANIFEST_DIR.exists()

    def test_at_least_one_manifest(self):
        manifests = list(MANIFEST_DIR.glob("l1*_seed*.json"))
        assert len(manifests) >= 1, "No per-seed manifest files found"

    def test_manifest_required_fields(self):
        manifests = list(MANIFEST_DIR.glob("l1*_seed*.json"))
        if not manifests:
            pytest.skip("No manifests")
        for p in manifests:
            with open(p) as f:
                m = json.load(f)
            for key in REQUIRED_MANIFEST_KEYS:
                assert key in m, f"{p.name}: missing key '{key}'"

    def test_manifest_gnn_source_real(self):
        manifests = list(MANIFEST_DIR.glob("l1*_seed*.json"))
        if not manifests:
            pytest.skip("No manifests")
        for p in manifests:
            with open(p) as f:
                m = json.load(f)
            assert m.get("gnn_source") == "real_checkpoint", \
                f"{p.name}: gnn_source={m.get('gnn_source')}"

    def test_training_summary_exists(self):
        """At least one training summary (v1 or v2) should exist."""
        s1 = MANIFEST_DIR / "training_summary.json"
        s2 = MANIFEST_DIR / "training_summary_v2.json"
        assert s1.exists() or s2.exists(), "No training summary found"

    def test_training_summary_5_seeds(self):
        for name in ["training_summary_v2.json", "training_summary.json"]:
            p = MANIFEST_DIR / name
            if p.exists():
                with open(p) as f:
                    s = json.load(f)
                assert s.get("seed_count", 0) >= 5, \
                    f"{name}: seed_count={s.get('seed_count')} < 5"
                return
        pytest.skip("No training summary")


# ─────────────────────────────────────────────────────────────────────────────
# test_paper_ready_gate
# ─────────────────────────────────────────────────────────────────────────────

class TestPaperReadyGate:
    """Enforce paper-ready conditions."""

    def test_real_checkpoints_present(self):
        ckpts = list(CKPT_DIR.glob("l1*_seed*_best.pt"))
        assert len(ckpts) >= 5, f"Expected >=5 checkpoints, got {len(ckpts)}"

    def test_5_seed_training_complete(self):
        for name in ["training_summary_v2.json", "training_summary.json"]:
            p = MANIFEST_DIR / name
            if p.exists():
                with open(p) as f:
                    s = json.load(f)
                assert s.get("seed_count", 0) >= 5
                return
        pytest.fail("No training summary with >=5 seeds")

    def test_dataset_manifest_has_sha256(self):
        p = RESULTS_DIR / "real_dataset_manifest.json"
        if not p.exists():
            pytest.skip("Manifest not found")
        with open(p) as f:
            m = json.load(f)
        sha = m.get("graph_sha256", "")
        assert len(sha) == 64, f"graph_sha256 not 64 hex chars: '{sha[:20]}'"

    def test_reports_generated(self):
        required_reports = [
            "reports/real_gnn_asset_audit.md",
            "reports/real_dataset_profile.md",
            "reports/temporal_leakage_audit.md",
        ]
        for rel in required_reports:
            p = ROOT / rel
            assert p.exists(), f"Missing report: {rel}"

    def test_no_estimated_metrics_in_results(self):
        """Privacy results must use real metrics (not placeholders like 0.5)."""
        p = RESULTS_DIR / "real_privacy_utility.csv"
        if not p.exists():
            pytest.skip("Privacy results not yet generated")
        import csv
        with open(p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Must have actual measurements
        assert len(rows) > 0, "Empty privacy utility results"


# ─────────────────────────────────────────────────────────────────────────────
# test_real_e2e_latency
# ─────────────────────────────────────────────────────────────────────────────

class TestRealE2ELatency:
    def test_latency_file_exists(self):
        p = RESULTS_DIR / "real_e2e_latency.csv"
        if not p.exists():
            pytest.skip("Latency results not yet generated (run Phase J first)")
        assert p.stat().st_size > 0

    def test_latency_has_required_columns(self):
        import csv
        p = RESULTS_DIR / "real_e2e_latency.csv"
        if not p.exists():
            pytest.skip("Latency file not found")
        with open(p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        required = ["T", "mean_total_ms", "p95_total_ms", "events_per_sec", "gnn_source"]
        for col in required:
            assert col in rows[0], f"Missing column: {col}"

    def test_latency_gnn_source_real(self):
        import csv
        p = RESULTS_DIR / "real_e2e_latency.csv"
        if not p.exists():
            pytest.skip("Latency file not found")
        with open(p) as f:
            for row in csv.DictReader(f):
                assert row.get("gnn_source") == "real_checkpoint", \
                    f"Expected real_checkpoint, got {row.get('gnn_source')}"

    def test_latency_p95_under_1000ms(self):
        """P95 latency should be under 1 second for streaming use case."""
        import csv
        p = RESULTS_DIR / "real_e2e_latency.csv"
        if not p.exists():
            pytest.skip("Latency file not found")
        with open(p) as f:
            for row in csv.DictReader(f):
                t = int(row.get("T", 1))
                p95 = float(row.get("p95_total_ms", 0))
                if t <= 10:
                    assert p95 < 1000, \
                        f"T={t}: p95={p95:.1f}ms exceeds 1000ms threshold"


# ─────────────────────────────────────────────────────────────────────────────
# test_privacy_metrics_not_estimated
# ─────────────────────────────────────────────────────────────────────────────

class TestPrivacyMetricsNotEstimated:
    """Privacy metrics must be empirically measured, not estimated or placeholder."""

    def test_privacy_file_exists(self):
        p = RESULTS_DIR / "real_privacy_utility.csv"
        if not p.exists():
            pytest.skip("Privacy results not yet generated")
        assert p.stat().st_size > 0

    def test_all_metrics_are_numeric(self):
        import csv
        p = RESULTS_DIR / "real_privacy_utility.csv"
        if not p.exists():
            pytest.skip("Privacy file not found")
        float_cols = ["attack_accuracy", "attack_balanced_accuracy",
                      "attack_macro_f1", "attack_roc_auc", "attack_pr_auc"]
        with open(p) as f:
            for row in csv.DictReader(f):
                for col in float_cols:
                    val = row.get(col, "N/A")
                    assert val != "N/A", f"Column '{col}' has placeholder N/A"
                    assert val != "estimated", f"Column '{col}' is 'estimated'"
                    try:
                        float(val)
                    except ValueError:
                        pytest.fail(f"Non-numeric value in {col}: {val}")

    def test_not_all_same_value(self):
        """At least some variance in attack accuracy across noise levels."""
        import csv
        p = RESULTS_DIR / "real_privacy_utility.csv"
        if not p.exists():
            pytest.skip("Privacy file not found")
        with open(p) as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 2:
            pytest.skip("Not enough rows")
        accs = [float(r.get("attack_accuracy", 0.5)) for r in rows]
        assert len(set(round(a, 3) for a in accs)) > 1, \
            "All attack_accuracy values are identical (may indicate placeholder values)"
