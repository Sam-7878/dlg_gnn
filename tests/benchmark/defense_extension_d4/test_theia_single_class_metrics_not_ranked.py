import csv
import json

from gog_fraud.extensions.defense.evaluation_policy import undefined_single_class_metrics


def test_theia_single_class_metrics_not_ranked(d4_root):
    scope = json.loads((d4_root / "statistics/statistical_scope.json").read_text(encoding="utf-8"))
    with (d4_root / "statistics/performance_11_dataset_raw.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert scope["theia_placeholder_metrics_included"] is False
    assert not any("THEIA" in row.get("dataset", "") for row in rows)
    undefined = undefined_single_class_metrics()
    assert undefined["metric_status"] == "undefined_single_class"
    assert undefined["roc_auc"] is undefined["pr_auc"] is undefined["f1"] is None
