"""Independent stdlib implementation must honor metric ties and undefined support."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'projects/stream_mc/scripts'))
from r01_public import metrics,exact_p,holm


def test_grouped_tie_average_precision():
    records=[{'label':1,'local_probability':.5,'local_prediction':1},{'label':0,'local_probability':.5,'local_prediction':1}]
    result=metrics(records,'local')
    assert result['ap']==.5 and result['f1']==pytest.approx(2/3) and result['mcc']==0
    assert result==metrics(records[::-1],'local')


def test_negative_support_remains_undefined():
    records=[{'label':0,'local_probability':.2,'local_prediction':0}]
    result=metrics(records,'local')
    assert result['ap'] is result['recall'] is result['f1'] is result['mcc'] is result['precision'] is None


def test_historical_exact_holm_family():
    values=[exact_p(b,c) for b,c in [(14,24),(8,39),(35,18),(41,12),(28,15)]]
    assert values==pytest.approx([.1433066543,.0000055396,.0270083177,.0000817133,.0659940345],abs=1e-10)
    assert holm(values)==pytest.approx([.1433066543,.0000276981,.0810249530,.0003268534,.1319880689],abs=1e-10)
