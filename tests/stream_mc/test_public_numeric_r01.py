"""Independent stdlib implementation must honor metric ties and undefined support."""
from pathlib import Path
import hashlib
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'projects/stream_mc/scripts'))
from r01_public import metrics,exact_p,holm,ensure_archive


def test_grouped_tie_average_precision():
    records=[{'label':1,'local_probability':.5,'local_prediction':1},{'label':0,'local_probability':.5,'local_prediction':1}]
    result=metrics(records,'local')
    assert result['ap']==.5 and result['f1']==pytest.approx(2/3) and result['mcc']==0
    assert result==metrics(records[::-1],'local')


def test_negative_support_remains_undefined():
    records=[{'label':0,'local_probability':.2,'local_prediction':0}]
    result=metrics(records,'local')
    assert result['ap'] is result['recall'] is result['f1'] is result['mcc'] is result['precision'] is None


def test_known_exact_holm_arithmetic_fixture():
    # These numbers are a math fixture, not the actual retained R4 counts.
    values=[exact_p(b,c) for b,c in [(14,24),(8,39),(35,18),(41,12),(28,15)]]
    assert values==pytest.approx([.1433066543,.0000055396,.0270083177,.0000817133,.0659940345],abs=1e-10)
    assert holm(values)==pytest.approx([.1433066543,.0000276981,.0810249530,.0003268534,.1319880689],abs=1e-10)


def parts_fixture(tmp_path):
    pieces=[b'first numeric payload',b'second numeric payload']
    parts=[]
    for i,content in enumerate(pieces,1):
        name=f'r01_numeric_evidence.zip.part{i:03d}';(tmp_path/name).write_bytes(content)
        parts.append({'name':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
    record={'parts':parts,'parts_format':'binary concatenation in listed order',
            'archive_sha256':hashlib.sha256(b''.join(pieces)).hexdigest(),'archive_bytes':sum(map(len,pieces))}
    expected=tmp_path/'r01_release.json';expected.write_text(json.dumps(record))
    return tmp_path/'r01_numeric_evidence.zip',expected,record,b''.join(pieces)


def test_git_parts_exact_assembly(tmp_path):
    target,expected,record,content=parts_fixture(tmp_path)
    assert ensure_archive(target,expected).read_bytes()==content
    assert ensure_archive(target,expected)==target
    assert not list(tmp_path.glob('.r01_assemble_*.tmp'))


def test_git_parts_reject_corruption(tmp_path):
    target,expected,record,content=parts_fixture(tmp_path)
    (tmp_path/record['parts'][0]['name']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='part SHA/size mismatch'):ensure_archive(target,expected)
    assert not target.exists() and not list(tmp_path.glob('.r01_assemble_*.tmp'))


def test_git_parts_reject_unsafe_name(tmp_path):
    target,expected,record,content=parts_fixture(tmp_path)
    record['parts'][0]['name']='../outside';expected.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='unsafe/noncontiguous'):ensure_archive(target,expected)
    assert not target.exists()


def test_git_parts_reject_wrong_whole_hash(tmp_path):
    target,expected,record,content=parts_fixture(tmp_path)
    record['archive_sha256']='0'*64;expected.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='assembled evidence SHA/size mismatch'):ensure_archive(target,expected)
    assert not target.exists()
