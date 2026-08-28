"""
experiments/run_leakage.py

Label leakage audit + attribute inference attack experiment.

Steps:
    1. Generate synthetic context dataset
    2. Run structural leakage audit (LeakageDetector)
    3. Run TF-IDF+LR shortcut test (text → fraud_label AUC)
    4. Run attribute inference attack on each privacy representation

Results:
    reports/dataset_leakage_audit.md
    results/leakage/leakage_results.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--output",  default="results/leakage")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_generation.synthetic_context_generator import (
        SyntheticContextGenerator, assign_scenarios_no_leakage
    )
    from graphrag.local_kb import LocalKnowledgeBase
    from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
    from graphrag.risk_extractor import RiskExtractor
    from privacy.leakage_attack import LeakageAttack
    from privacy.vector_codec import VectorCodec
    from validation.leakage_detector import LeakageDetector

    base_cfg = _load_yaml(args.config)
    root = Path(__file__).parent.parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    seed = base_cfg.get("experiment", {}).get("seeds", [42])[0]
    n_samples = 2000
    fraud_rate = 0.10

    rng = np.random.RandomState(seed)
    labels_np = (rng.rand(n_samples) < fraud_rate).astype(int)

    # ── Step 1: Generate synthetic dataset ───────────────────────────────
    log.info("Step 1: Generating synthetic context dataset ...")
    scenarios = assign_scenarios_no_leakage(labels_np, seed=seed)
    gen = SyntheticContextGenerator(seed=seed)
    event_ids = [f"tx_{i:06d}" for i in range(n_samples)]
    context_path = str(output_dir / "leakage_audit_contexts.jsonl")
    records = gen.generate_contexts(
        scenario_types=scenarios,
        event_ids=event_ids,
        output_path=context_path,
    )
    log.info(f"  Generated {len(records)} context records")

    # ── Step 2: Structural leakage audit ─────────────────────────────────
    log.info("Step 2: Running structural leakage audit ...")
    report_path = str(reports_dir / "dataset_leakage_audit.md")
    detector = LeakageDetector(context_path)
    leakage_found = detector.detect_leakage(report_path)
    log.info(f"  Audit result: {'⚠️ LEAKAGE DETECTED' if leakage_found else '✅ CLEAN'}")
    log.info(f"  Report: {report_path}")

    # ── Step 3: TF-IDF shortcut test ─────────────────────────────────────
    log.info("Step 3: Running TF-IDF+LR shortcut test ...")
    attack = LeakageAttack(max_iter=500, cv_folds=5)
    context_texts = [r["context_text"] for r in records]
    shortcut_result = attack.run_shortcut_test(context_texts, labels_np)
    log.info(
        f"  Shortcut test: Acc={shortcut_result.accuracy:.4f}, "
        f"F1={shortcut_result.macro_f1:.4f}, AUC={shortcut_result.auc:.4f}"
    )
    is_sc = (shortcut_result.auc > 0.85) if not np.isnan(shortcut_result.auc) else (shortcut_result.macro_f1 > 0.80)
    if is_sc:
        log.warning("⚠️  SHORTCUT DETECTED — review dataset construction")
    else:
        log.info("  ✅ No trivial shortcut detected (AUC <= 0.85)")

    # ── Step 4: Attribute inference attack per representation ─────────────
    log.info("Step 4: Running attribute inference attacks ...")

    kb = LocalKnowledgeBase()
    retriever = GraphRAGRetriever(kb, RetrieverConfig(top_k=5, graph_hops=1))
    extractor = RiskExtractor()

    base_risk_dicts = []
    for rec in records:
        evidence = retriever.retrieve(rec["context_text"])
        rd = extractor.extract(evidence, event_id=rec["event_id"],
                               pre_transaction_gap_sec=rec.get("pre_transaction_gap_sec", 300))
        base_risk_dicts.append(rd)

    category_labels = np.array([d.get("risk_type_id", 0) for d in base_risk_dicts])
    codec_b = VectorCodec("binary")

    attack_results = []
    for mode_name in ["full_risk_vector", "quantized_risk_vector", "noisy_risk_vector", "minimal_risk_token"]:
        from experiments.run_privacy_utility import _apply_privacy_mode
        transformed = [_apply_privacy_mode(d, mode_name) for d in base_risk_dicts]
        r = attack.run_from_dicts(
            transformed, category_labels,
            representation_name=mode_name,
            target_attribute="scam_category",
            codec=codec_b,
        )
        attack_results.append(r.to_dict())
        log.info(
            f"  [{mode_name}]: Acc={r.accuracy:.4f}, F1={r.macro_f1:.4f}, "
            f"AUC={r.auc:.4f}, bytes={r.bytes_per_sample:.1f}B"
        )

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "structural_audit": {
            "leakage_found": leakage_found,
            "report_path": report_path,
        },
        "shortcut_test": shortcut_result.to_dict(),
        "attribute_inference_attacks": attack_results,
        "seed": seed,
        "n_samples": n_samples,
    }
    out_json = output_dir / "leakage_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nLeakage results saved to {out_json}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("  LEAKAGE AUDIT SUMMARY")
    log.info("=" * 60)
    log.info(f"  Structural audit : {'FAIL' if leakage_found else 'PASS'}")
    is_sc = (shortcut_result.auc > 0.85) if not np.isnan(shortcut_result.auc) else (shortcut_result.macro_f1 > 0.80)
    log.info(f"  Shortcut test    : {'FAIL (AUC>0.85)' if is_sc else 'PASS'} (AUC={shortcut_result.auc:.4f}, Acc={shortcut_result.accuracy:.4f}, F1={shortcut_result.macro_f1:.4f})")
    for ar in attack_results:
        log.info(f"  Attack [{ar['representation']:30s}]: Acc={ar['accuracy']:.4f}, {ar['bytes']:.0f}B")


if __name__ == "__main__":
    main()
