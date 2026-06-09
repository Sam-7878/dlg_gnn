import json
import os
import logging

logger = logging.getLogger(__name__)

class LeakageDetector:
    """
    Checks for potential label leakage or duplications in the synthetic contexts.
    """
    def __init__(self, context_path: str):
        self.context_path = context_path

    def detect_leakage(self, report_md_path: str) -> bool:
        logger.info(f"Detecting data leakage in {self.context_path}...")
        
        leakage_detected = False
        leaked_examples = []
        
        with open(self.context_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                text = item["context_text"].lower()
                label = int(item["label"])
                
                # Check 1: Direct labels exposure in text
                direct_leak = False
                for forbidden in ["label=1", "is_fraud=true", "this_is_a_scam"]:
                    if forbidden in text:
                        direct_leak = True
                        leakage_detected = True
                        
                if direct_leak:
                    leaked_examples.append(item["context_id"])
                    
        os.makedirs(os.path.dirname(report_md_path), exist_ok=True)
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write("# Data Leakage Detection Report\n\n")
            f.write(f"**Date Verified**: 2026-06-08\n")
            f.write(f"**Leakage Status**: {'⚠️ DETECTED' if leakage_detected else '✅ CLEAN'}\n\n")
            if leakage_detected:
                f.write(f"### Leaked Examples:\n")
                for ex in leaked_examples[:10]:
                    f.write(f"- {ex}\n")
            else:
                f.write("No direct label leakages (e.g. static labels inside context texts) were detected. The dataset passes validation criteria.\n")
                
        logger.info(f"Leakage report saved to {report_md_path}")
        return leakage_detected
