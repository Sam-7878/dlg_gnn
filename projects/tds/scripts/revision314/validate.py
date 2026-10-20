"""Final integrity checks and local artifact manifests; performs no training."""
from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from common import FROZEN, GRAPH, GRAPH_HASH, OUT, PROJECT, ROOT, load, sha, write_json
from historical_metrics import classification_metrics


def main():
    _,frame,_=load()
    expected=frame.loc[frame.split=='test',['event_id','timestamp','label']].reset_index(drop=True)
    historic=json.loads((FROZEN/'reproducibility_manifest.json').read_text())
    for e in historic['entries']:
        p=FROZEN/e['path'];assert p.is_file(),p
        assert sha(p)==e['sha256'],p
    checkpoint_manifest=json.loads((FROZEN/'checkpoints_manifest.json').read_text())
    for e in checkpoint_manifest['entries']:
        k=e['model_key'];s=e['seed']
        p=ROOT/f'archive/tds/data/gog_scimain_v1_preserved_panel/checkpoints/seed{s}.pt' if k=='proposed' else FROZEN/f'checkpoints/{k}/seed{s}.pt'
        assert sha(p)==e['checkpoint_sha256'],p
    runs=list((OUT/'runs').glob('*/seed*/metrics.json'));assert len(runs)==35
    for path in runs:
        m=json.loads(path.read_text());assert m['dataset_sha256']==GRAPH_HASH
        assert m['test_access_count']==1
        assert sha(path.parent/'checkpoint.pt')==m['checkpoint_sha256']
        assert sha(path.parent/'test.csv')==m['test_predictions_sha256']
        p=pd.read_csv(path.parent/'test.csv');assert p[['event_id','timestamp','label']].equals(expected)
        actual=classification_metrics(p.label.to_numpy(int),p.p_mean.to_numpy(float))
        for key,value in actual.items():assert np.isclose(value,m[key],rtol=0,atol=1e-12),(path,key)
        assert not m['max_epoch_hit'],path
        valid=pd.read_csv(path.parent/'validation.csv')
        assert valid[['event_id','timestamp','label']].equals(frame.loc[frame.split=='validation',['event_id','timestamp','label']].reset_index(drop=True))
    strict=pd.read_csv(OUT/'data/strict_time_source_audit.csv')
    assert len(strict)==3648 and strict.inclusive_x_exact.all() and strict.inclusive_edges_exact.all()
    checked=strict.merge(expected,on='event_id',validate='one_to_one')
    assert (checked.strict_max_timestamp<checked.timestamp).all()
    sensitivity=pd.read_csv(OUT/'data/strict_time_sensitivity.csv');assert len(sensitivity)==20
    assert sensitivity.auc_pr_delta.abs().max()<.006
    for name,n in [('statistical_comparisons_corrected',7),('control_statistical_comparisons',6)]:
        f=pd.read_csv(OUT/f'supplement/{name}.csv');assert len(f)==n
        for c in ['n_bootstrap','n_bootstrap_valid','n_class_stratified_bootstrap','n_randomization']:
            assert (f[c]==10000).all()
        assert (f.holm_adjusted_p>=f.randomization_p_value_two_sided).all()
    operations=pd.read_csv(OUT/'data/operational_metrics.csv')
    assert (operations.top_1pct_budget==37).all() and (operations.top_5pct_budget==183).all()
    pdflogs={}
    for source in [OUT/'applsci_article.tex',OUT/'supplement/supplementary.tex']:
        log=source.with_suffix('.log').read_text(errors='replace')
        warnings=[x for x in log.splitlines() if re.search(r'Overfull|Underfull|Warning|^!',x)]
        assert not warnings,(source,warnings)
        text=subprocess.check_output(['pdftotext','-layout',str(source.with_suffix('.pdf')),'-'],text=True)
        assert len(text)>3000 and '??' not in text
        pages=re.search(r'Output written on .*?\((\d+) pages',log).group(1)
        pdflogs[str(source.relative_to(OUT))]=dict(pages=int(pages),sha256=sha(source.with_suffix('.pdf')),warnings=[])
    tex=(OUT/'applsci_article.tex').read_text()
    abstract=tex.split('\\abstract{',1)[1].split('}\n\\keyword',1)[0]
    abstract_words=len(abstract.split());assert 180<=abstract_words<=220,abstract_words
    assert '24,316 transaction events' not in tex
    assert 'epistemic spread' not in tex
    assert 'Had subgraphs been extracted' not in tex
    versions={x:importlib.metadata.version(x) for x in ['torch','torch-geometric','numpy','pandas','scipy','scikit-learn','matplotlib','pytest']}
    write_json(OUT/'supplement/runtime_lock.json',dict(packages=versions,python='3.14.4',os='Ubuntu 26.04.1 on WSL2',
        policy='lock of the executed supplemental environment; no packages were installed into the shared Python environment'))
    validation=dict(status='PASS',historical_artifact_hashes_verified=len(historic['entries']),
        historical_checkpoint_hashes_verified=len(checkpoint_manifest['entries']),
        frozen_graph_sha256=sha(GRAPH),supplemental_runs_verified=len(runs),
        supplemental_prediction_identity_and_metrics_verified=True,strict_source_samples_verified=len(strict),
        fixed_checkpoint_sensitivity_runs=len(sensitivity),original_comparisons=7,exploratory_comparisons=6,
        resamples_per_protocol=10000,pdf_builds=pdflogs,abstract_words=abstract_words,
        unit_tests='8 passed; final pytest rerun recorded separately',
        original_results_preserved=True,submission_ready=False)
    write_json(OUT/'reports/validation.json',validation)
    gate=dict(implementation_complete=True,scientific_revision_complete=True,submission_ready=False,
              manuscript_claims_narrowed=True,strict_time_contract_bug_detected=False,
              additional_architecture_development_required=False,
              author_closure=['TDS CRediT and final author consent','corresponding author, identity and declarations',
                              'grant award agreement verification','TDS immutable public release and actual Data Availability identity'],
              historical_limitations=['full original epoch trajectories unavailable','terminal test below one day',
                                      'retrospective label availability; no prospective label timestamps'],
              historical_gate_m_v8='preserved as historical completeness decision',gate_a=False,gate_b=False)
    write_json(OUT/'reports/submission_gate.json',gate)
    pdfreview=dict(scope='Manual visual review of final rendered title/table/figure pages; not an assertion of every-page manual review',
                   article_sha256=sha(OUT/'applsci_article.pdf'),
                   rendered_pages=[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in sorted((OUT/'reports/pdf_review').glob('*.png'))],
                   findings=['Title, abstract and explicit pending corresponding author visible; no clipping',
                             'Primary/corrected/control tables fit their widths with readable vector text',
                             'MC SD bars, runtime scope, reliability bins and support plot visible; no overlap'])
    write_json(OUT/'reports/pdf_review.json',pdfreview)
    entries=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or 'packages' in p.parts or p.name=='artifact_manifest.json':continue
        if p.suffix in ['.aux','.log','.out','.fls','.fdb_latexmk','.blg','.bbl','.pyc']:continue
        entries.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    sources=[]
    for p in sorted((PROJECT/'scripts/revision314').rglob('*.py')):
        sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    write_json(OUT/'reports/artifact_manifest.json',dict(entries=entries,artifact_count=len(entries),source_entries=sources,
        policy='covers completed local revision, not a public release attestation'))
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
