import json
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class ContextValidator:
    """
    Validates synthetic context quality, ensuring labels and semantic scenario types match.
    """
    def __init__(self, context_path: str):
        self.context_path = context_path

    def validate(self, report_csv_path: str) -> pd.DataFrame:
        logger.info(f"Validating context dataset at {self.context_path}...")
        
        results = []
        with open(self.context_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                label = int(item["label"])
                scenario = item["scenario_type"]
                cues = item["risk_cues"]
                
                # Validation rules
                label_consistency = 1.0
                if label == 1 and scenario in ["benign", "hard_negative"]:
                    label_consistency = 0.0
                elif label == 0 and scenario not in ["benign", "hard_negative"]:
                    label_consistency = 0.0
                    
                scenario_consistency = 1.0 if scenario != "pending" else 0.0
                
                # Coverage score based on presence of cues in fraud scenarios
                if label == 1:
                    cue_coverage = 1.0 if len(cues) > 0 else 0.5
                else:
                    cue_coverage = 1.0
                    
                overall = (label_consistency + scenario_consistency + cue_coverage) / 3.0
                
                results.append({
                    "context_id": item["context_id"],
                    "label_consistency": label_consistency,
                    "scenario_consistency": scenario_consistency,
                    "risk_cue_coverage": cue_coverage,
                    "overall_validation_score": round(overall, 4),
                    "validation_status": "pass" if overall >= 0.75 else "fail"
                })
                
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(report_csv_path), exist_ok=True)
        df.to_csv(report_csv_path, index=False)
        logger.info(f"Context validation report saved to {report_csv_path}")
        return df
