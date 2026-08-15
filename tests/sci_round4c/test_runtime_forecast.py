import pandas as pd

from gog_fraud.experiments.round4c_policy import runtime_forecast


def test_forecast_expands_each_measured_cell_not_global_mean():
    frame=pd.DataFrame([
        {"dataset":"A","model":"DOMINANT","seed":42,"status":"success","total_wall_sec":10},
        {"dataset":"A","model":"DOMINANT","seed":43,"status":"success","total_wall_sec":14},
        {"dataset":"B","model":"AnomalyDAE","seed":42,"status":"success","total_wall_sec":100},
        {"dataset":"B","model":"AnomalyDAE","seed":43,"status":"success","total_wall_sec":120},
    ])
    result=runtime_forecast(frame,round5_seeds=5)
    assert result["optimistic_sec"] == 550
    assert result["median_sec"] == 610
    assert result["pessimistic_sec"] == 670
    assert result["quadratic_anomalydae"]["median_sec"] == 550

