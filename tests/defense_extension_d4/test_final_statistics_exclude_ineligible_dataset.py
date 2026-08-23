import csv
import json


def test_final_statistics_exclude_ineligible_dataset(d4_root):
    with (d4_root / "statistics/performance_11_dataset_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    datasets = {row["dataset"] for row in rows}
    assert "LANL-RedTeam" in datasets
    assert not any("THEIA" in dataset for dataset in datasets)
    assert len(datasets) == 11
    scope = json.loads((d4_root / "statistics/statistical_scope.json").read_text(encoding="utf-8"))
    assert scope["primary_performance_datasets"] == 10
    assert scope["descriptive_performance_datasets"] == 11
    assert scope["scalability_portfolio_datasets"] == 12
