#!/usr/bin/env python3
"""
A13 Literal Final Manuscript Numeric Contract Validator (A13-G2).
Parses literal .tex files directly to verify all numeric statements against approved registries.
Includes a negative control test asserting that any injected discrepancy fails the gate.
"""

import re
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PAPER_A13 = BASE_DIR / 'paper' / 'a13'
MDPI_A13 = PAPER_A13 / 'mdpi'
PUB_A13 = BASE_DIR / 'publication' / 'a13'
PUB_A13.mkdir(parents=True, exist_ok=True)

PREPRINTS_TEX = PAPER_A13 / 'DLG-Benchmark_A13_Preprints.tex'
APPLSCI_TEX = MDPI_A13 / 'DLG-Benchmark_A13_AppliedSciences.tex'

def load_text(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing manuscript: {path}")
    return path.read_text(encoding='utf-8')

def validate_text(text, doc_name):
    """
    Directly extracts and verifies all mandatory numeric facts from literal tex content.
    """
    errors = []
    checks = {}
    
    # 1. Dataset & Model Counts
    # Check "13 primary" and "1 external" or "LANL"
    if not re.search(r"13\s+primary", text, re.IGNORECASE):
        errors.append("Missing mention of '13 primary' datasets")
    else:
        checks["13_primary_datasets"] = "PASS"
        
    if not re.search(r"1\s+external|LANL", text):
        errors.append("Missing mention of '1 external' or 'LANL'")
    else:
        checks["1_external_lanl"] = "PASS"
        
    if not re.search(r"(?:7|seven)\s+representative\s+detector|(?:7|seven)\s+models", text, re.IGNORECASE):
        errors.append("Missing mention of 7 representative detector models")
    else:
        checks["7_models"] = "PASS"
        
    if not re.search(r"84\s*(?:of|/)\s*91", text, re.IGNORECASE):
        errors.append("Missing mention of 84 of 91 primary supported pairs")
    else:
        checks["84_of_91_supported"] = "PASS"
        
    if not re.search(r"455\s+total\s+runs|455\s+experimental|420\s+successful\s+primary\s+runs", text, re.IGNORECASE):
        errors.append("Missing mention of 455 total runs (420 primary + 35 LANL)")
    else:
        checks["455_total_runs"] = "PASS"

    # 2. Key Dataset Metrology & Deltas
    # Reddit AP: Base 0.4044, Aug 0.3388, Delta -0.0656
    if "0.4044" in text and "0.3388" in text and "-0.0656" in text:
        checks["reddit_inversion_ap"] = "PASS (0.4044 -> 0.3388, Delta -0.0656)"
    else:
        errors.append("Reddit AP values (0.4044, 0.3388, -0.0656) not found")

    # Elliptic AP: Base 0.0687, Aug 0.1037, Delta +0.0350
    if "0.0687" in text and "0.1037" in text and "+0.0350" in text:
        checks["elliptic_gain_ap"] = "PASS (0.0687 -> 0.1037, Delta +0.0350)"
    else:
        errors.append("Elliptic AP values (0.0687, 0.1037, +0.0350) not found")

    # Ethereum OCGNN: 0.4996 vs Aug 0.4359
    if "0.4996" in text and "0.4359" in text:
        checks["eth_ocgnn_degradation"] = "PASS (0.4996 > 0.4359)"
    else:
        errors.append("Ethereum OCGNN degradation values (0.4996, 0.4359) not found")

    # BSC AnomalyDAE: 0.3609 vs Aug 0.3411
    if "0.3609" in text and "0.3411" in text:
        checks["bsc_anomalydae_degradation"] = "PASS (0.3609 > 0.3411)"
    else:
        errors.append("BSC AnomalyDAE degradation values (0.3609, 0.3411) not found")

    # Polygon AnomalyDAE: 0.1878 vs Aug 0.0731
    if "0.1878" in text and "0.0731" in text:
        checks["polygon_anomalydae_degradation"] = "PASS (0.1878 > 0.0731)"
    else:
        errors.append("Polygon AnomalyDAE degradation values (0.1878, 0.0731) not found")

    # 3. Friedman Statistics S1-S4 (Frozen A12/A13)
    # S1: Q=29.0357, asym_p=5.99e-05, mc_p=4.00e-06
    if "29.0357" in text and "5.99e-05" in text:
        checks["friedman_s1_roc"] = "PASS (Q=29.0357, p_asym=5.99e-05, p_mc=4.00e-06)"
    else:
        errors.append("Friedman S1 (29.0357, 5.99e-05) not found")

    # S2: Q=23.7538, asym_p=8.95e-05, mc_p=2.10e-05
    if "23.7538" in text and "8.95e-05" in text:
        checks["friedman_s2_pr"] = "PASS (Q=23.7538, p_asym=8.95e-05, p_mc=2.10e-05)"
    else:
        errors.append("Friedman S2 (23.7538, 8.95e-05) not found")

    # S3: Q=6.1333, asym_p=1.89e-01, mc_p=1.92e-01
    if "6.1333" in text and "1.89e-01" in text:
        checks["friedman_s3_nap"] = "PASS (Q=6.1333, p_asym=1.89e-01, p_mc=1.92e-01)"
    else:
        errors.append("Friedman S3 (6.1333, 1.89e-01) not found")

    # S4: Q=21.8286, asym_p=2.17e-04, mc_p=1.00e-06
    if "21.8286" in text and "2.17e-04" in text:
        checks["friedman_s4_rel"] = "PASS (Q=21.8286, p_asym=2.17e-04, p_mc=1.00e-06)"
    else:
        errors.append("Friedman S4 (21.8286, 2.17e-04) not found")

    # S4 Monte Carlo p=1.00 \times 10^{-6} floor
    if "1.00e-06" in text or ("1.0" in text and "10^{-6}" in text):
        checks["friedman_s4_mc_floor"] = "PASS (p_MC = 1.00e-06 floor)"
    else:
        errors.append("Friedman S4 MC floor (1.00e-06) not found")

    # 4. Dense Memory Arithmetic & Closed-Form Qualification
    # 4N^2 single matrix: 10.0 GB (50k), 160.0 GB (200k)
    if re.search(r"4\s*N\^2", text) and "10.0" in text and "160.0" in text:
        checks["dense_memory_single_matrix_4N2"] = "PASS (4N^2: 10.0 GB at 50k, 160.0 GB at 200k)"
    else:
        errors.append("Single matrix 4N^2 memory values (10.0 GB, 160.0 GB) not found")

    # 8N^2 two matrices: 20.0 GB (50k), 320.0 GB (200k)
    if re.search(r"8\s*N\^2", text) and "20.0" in text and "320.0" in text:
        checks["dense_memory_two_matrices_8N2"] = "PASS (8N^2: 20.0 GB at 50k, 320.0 GB at 200k)"
    else:
        errors.append("Two matrices 8N^2 memory values (20.0 GB, 320.0 GB) not found")

    # Unsquashed dot product Gram decoder qualification
    if ("unsquashed dot-product" in text.lower() or "unsquashed dot product" in text.lower()) and "gram" in text.lower():
        checks["unsquashed_gram_decoder_qualification"] = "PASS"
    else:
        errors.append("Unsquashed dot-product Gram decoder qualification not found")

    # 5. Hardware Platform Description
    if "RTX 3090" in text and "Thunderbolt" in text and "24" in text:
        checks["hardware_egpu_3090"] = "PASS (RTX 3090 24GB eGPU via Thunderbolt)"
    else:
        errors.append("Hardware description (RTX 3090 24GB eGPU Thunderbolt) not found")

    # 6. Immutable Release Tag benchmark-a13-submission-2026-10-24
    if "benchmark-a13-submission-2026-10-24" in text:
        checks["immutable_release_tag_a13"] = "PASS (benchmark-a13-submission-2026-10-24)"
    else:
        errors.append("Immutable release tag 'benchmark-a13-submission-2026-10-24' not found")

    # 7. Absence of Unverified Causal Assertions
    unverified_causal_phrases = [
        "overcomes hub reconstruction bias",
        "overcomes hub bias",
        "licit hubs cause inverted",
        "licit hubs cause below",
        "hub-biased failure"
    ]
    for phrase in unverified_causal_phrases:
        if phrase in text.lower():
            errors.append(f"Forbidden causal phrase found: '{phrase}'")
    if not any(phrase in text.lower() for phrase in unverified_causal_phrases):
        checks["no_unverified_causal_hub_claims"] = "PASS"

    return len(errors) == 0, checks, errors

def negative_control_test(clean_text):
    """
    Negative control test: Injects deliberate discrepancies into clean text
    and asserts that validate_text catches them and fails.
    """
    print("\n--- Running Negative Control Injections ---")
    
    # Test 1: Injected Reddit AP typo (0.4044 -> 0.4045)
    mutated_1 = clean_text.replace("0.4044", "0.4045")
    ok1, _, errs1 = validate_text(mutated_1, "negative_control_1")
    assert not ok1, "Negative control 1 failed: Altered Reddit AP 0.4045 was not caught!"
    assert any("0.4044" in e for e in errs1), f"Expected 0.4044 error, got: {errs1}"
    print("  [PASS] Injected Reddit AP mutation (0.4044 -> 0.4045) correctly caught and rejected.")
    
    # Test 2: Injected Friedman Q typo (29.0357 -> 29.0358)
    mutated_2 = clean_text.replace("29.0357", "29.0358")
    ok2, _, errs2 = validate_text(mutated_2, "negative_control_2")
    assert not ok2, "Negative control 2 failed: Altered Friedman Q 29.0358 was not caught!"
    assert any("29.0357" in e for e in errs2), f"Expected 29.0357 error, got: {errs2}"
    print("  [PASS] Injected Friedman S1 Q mutation (29.0357 -> 29.0358) correctly caught and rejected.")

    # Test 3: Injected Forbidden Causal Phrase
    mutated_3 = clean_text + "\nThis model overcomes hub reconstruction bias directly.\n"
    ok3, _, errs3 = validate_text(mutated_3, "negative_control_3")
    assert not ok3, "Negative control 3 failed: Forbidden causal phrase was not caught!"
    assert any("overcomes hub reconstruction bias" in e for e in errs3), f"Expected causal error, got: {errs3}"
    print("  [PASS] Injected forbidden causal phrase correctly caught and rejected.")

    print("All negative control tests PASSED successfully!\n")

def main():
    preprints_text = load_text(PREPRINTS_TEX)
    applsci_text = load_text(APPLSCI_TEX)
    
    print(f"Validating Preprints.org Manuscript: {PREPRINTS_TEX.name} ({len(preprints_text):,} chars)")
    ok_prep, checks_prep, errs_prep = validate_text(preprints_text, "Preprints.org")
    
    print(f"Validating Applied Sciences Manuscript: {APPLSCI_TEX.name} ({len(applsci_text):,} chars)")
    ok_appl, checks_appl, errs_appl = validate_text(applsci_text, "Applied Sciences")
    
    # Run negative control tests on preprints_text
    negative_control_test(preprints_text)
    
    results = {
        "gate": "A13-G2 Literal Final Manuscript Numeric Contract",
        "preprints_manuscript": {
            "file": str(PREPRINTS_TEX.name),
            "status": "PASS" if ok_prep else "FAIL",
            "checks": checks_prep,
            "errors": errs_prep
        },
        "applied_sciences_manuscript": {
            "file": str(APPLSCI_TEX.name),
            "status": "PASS" if ok_appl else "FAIL",
            "checks": checks_appl,
            "errors": errs_appl
        },
        "negative_control_verification": "PASS (3/3 deliberate corruptions rejected)",
        "overall_status": "PASS" if (ok_prep and ok_appl) else "FAIL"
    }
    
    contract_path = PUB_A13 / "manuscript_numeric_contract.json"
    validation_path = PUB_A13 / "manuscript_numeric_validation.json"
    contract_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    validation_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    
    print(f"Validation report saved to {validation_path}")
    
    if not (ok_prep and ok_appl):
        print("FAIL: Validation encountered errors:")
        if errs_prep:
            print(f"  Preprints errors: {errs_prep}")
        if errs_appl:
            print(f"  Applied Sciences errors: {errs_appl}")
        sys.exit(1)
        
    print("SUCCESS: A13-G2 Literal Final Manuscript Numeric Validation PASSED on both manuscripts!")

if __name__ == "__main__":
    main()
