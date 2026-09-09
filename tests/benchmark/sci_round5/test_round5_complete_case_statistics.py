import pandas as pd

from gog_fraud.experiments.round5_policy import complete_case_views


def test_views_are_derived_from_final_support_not_hardcoded_representative_views():
    rows=[]
    for dataset in ("D1","D2"):
        for model in ("A","B"):
            rows.append({"dataset":dataset,"model":model,"support_status":"supported"})
    rows.append({"dataset":"D1","model":"C","support_status":"supported"})
    rows.append({"dataset":"D2","model":"C","support_status":"unsupported_operational"})
    views=complete_case_views(pd.DataFrame(rows),["D1","D2"])
    fraud=views.loc[views.view_name.eq("fraud_oriented")].iloc[0]
    assert fraud.models == "A;B"
