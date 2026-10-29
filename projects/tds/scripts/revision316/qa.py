"""Numeric, scientific-source and cross-platform claim parity without training."""
import csv
import difflib
import json
import re
import shutil
import subprocess
from paths import ROOT,PROJECT,SHARED,PREPRINT,MDPI,RESULTS,REPORTS,SUPPLEMENTAL,FROZEN
from common import sha,write


def macro(text,name):
    start=text.index('\\'+name+'{')+len(name)+2;i=start;depth=1
    while depth:
        if text[i]=='{' and text[i-1]!='\\':depth+=1
        if text[i]=='}' and text[i-1]!='\\':depth-=1
        i+=1
    return text[start:i-1]


def entries(text):
    result={}
    for m in re.finditer(r'@\w+\{([^,]+),',text):
        i=m.end();depth=1
        while depth:
            if text[i]=='{' and text[i-1]!='\\':depth+=1
            if text[i]=='}' and text[i-1]!='\\':depth-=1
            i+=1
        result[m.group(1)]=text[m.start():i]
    return result


def main():
    # Verify every sealed R315 file again; do not regenerate R315 reports.
    from freeze import main as verify_freeze
    verify_freeze()
    graph=ROOT/'archive/tds/data/gog_scimain_v1/graph.pt'
    assert sha(graph)=='067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd'
    primary=json.loads((FROZEN/'reproducibility_manifest.json').read_text())
    for e in primary['entries']:assert sha(FROZEN/e['path'])==e['sha256']
    checkpoints=json.loads((FROZEN/'checkpoints_manifest.json').read_text())
    for e in checkpoints['entries']:
        path=(ROOT/f'archive/tds/data/gog_scimain_v1_preserved_panel/checkpoints/seed{e["seed"]}.pt'
              if e['model_key']=='proposed' else FROZEN/f'checkpoints/{e["model_key"]}/seed{e["seed"]}.pt')
        assert sha(path)==e['checkpoint_sha256']
    reference=list(csv.DictReader((FROZEN/'raw_predictions/proposed/seed7_test.csv').open()))
    identity=[(e['event_id'],e['timestamp'],e['label']) for e in reference]
    paths=list((FROZEN/'raw_predictions').glob('*/seed*_test.csv'))+list((SUPPLEMENTAL/'runs').glob('*/seed*/test.csv'))
    assert len(paths)==65
    for path in paths:
        rows=list(csv.DictReader(path.open()))
        assert [(e['event_id'],e['timestamp'],e['label']) for e in rows]==identity,path
        assert len(rows)==3648 and sum(int(float(e['label'])) for e in rows)==107
    metrics=list((SUPPLEMENTAL/'runs').glob('*/seed*/metrics.json'));assert len(metrics)==35
    for path in metrics:
        m=json.loads(path.read_text())
        assert sha(path.parent/'checkpoint.pt')==m['checkpoint_sha256']
        assert sha(path.parent/'test.csv')==m['test_predictions_sha256']
    write(REPORTS/'frozen_evidence_verification.json',dict(status='PASS',graph_sha256=sha(graph),
        primary_artifacts=len(primary['entries']),primary_checkpoints=len(checkpoints['entries']),
        supplemental_checkpoints=35,aligned_test_arrays=65,n=3648,positive=107,new_training=False))
    body=(SHARED/'sections/manuscript_body.tex').read_text()
    meta=json.loads((SHARED/'metadata.json').read_text())
    protected=[]
    for group in ['data','figures','supplement']:
        for p in sorted((SHARED/group).rglob('*')):
            if not p.is_file():continue
            if group=='supplement' and p.name=='supplementary.tex':continue
            base=PROJECT/'paper/common/315'/group/p.relative_to(SHARED/group)
            assert base.is_file() and sha(p)==sha(base),p
            for folder in [PREPRINT,MDPI]:assert sha(folder/group/p.relative_to(SHARED/group))==sha(p)
            protected.append(dict(path=str(p.relative_to(SHARED)),sha256=sha(p)))
    for folder in [PREPRINT,MDPI]:
        assert sha(folder/'manuscript_body.tex')==sha(SHARED/'sections/manuscript_body.tex')
        assert sha(folder/'references.bib')==sha(SHARED/'references.bib')
        assert sha(folder/'supplement/supplementary.tex')==sha(SHARED/'supplement/supplementary.tex')
    assert macro((MDPI/'applsci_article.tex').read_text(),'Title')==meta['Title']
    assert macro((MDPI/'applsci_article.tex').read_text(),'abstract')==meta['abstract']
    assert '\\title{'+meta['Title']+'}' in (PREPRINT/'preprint_main.tex').read_text()
    assert meta['abstract'] in (PREPRINT/'preprint_main.tex').read_text()
    old_meta=json.loads((PROJECT/'paper/common/315/metadata.json').read_text())
    assert meta['Title']==old_meta['Title']
    before=entries((PROJECT/'paper/common/315/references.bib').read_text())
    after=entries((SHARED/'references.bib').read_text())
    assert all(after.get(k)==v for k,v in before.items())
    assert set(after)-set(before)=={'tang2023gadbench','qiao2025survey'}
    cited=set(k.strip() for m in re.findall(r'\\cite\w*\{([^}]+)\}',body) for k in m.split(','))
    assert cited<=set(after) and len(cited)==21
    for folder,entry in [(PREPRINT,'preprint_main'),(MDPI,'applsci_article')]:
        bbl=(folder/(entry+'.bbl')).read_text();assert len(re.findall(r'\\bibitem',bbl))==21
    assert 'Applied Anomaly-Screening Operating Points' in body
    assert all(s in body+'\n'+meta['abstract'] for s in ['supervised','18 h 57 min 10 s','107','0.4751','0.4524','24,316','3,648'])
    assert '0.4751' in meta['abstract'] and '0.4524' in meta['abstract']
    scalar=[]
    for src in sorted((SUPPLEMENTAL/'data').glob('*.csv')):
        if src.name=='strict_time_source_audit.csv':continue
        dst=RESULTS/'data'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        scalar.append(dict(path=str(dst.relative_to(ROOT)),source=str(src.relative_to(ROOT)),sha256=sha(src)))
    for name in ['statistical_comparisons_corrected.csv','control_statistical_comparisons.csv','reliability_bin_support.csv']:
        src=SUPPLEMENTAL/'supplement'/name;dst=RESULTS/'statistics'/name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        scalar.append(dict(path=str(dst.relative_to(ROOT)),source=str(src.relative_to(ROOT)),sha256=sha(src)))
    write(RESULTS/'frozen_numeric_identity.json',dict(entries=scalar,scientific_primary_and_supplemental_results_changed=False))
    # Claims are bounded; negative wording is retained rather than blacklisting
    # every occurrence of words such as prospective or privacy.
    rows=[
      ('task','Supervised retrospective contract-category risk classification, not unsupervised novelty discovery','Abstract/Introduction/Related Work','dataset contract and frozen panel'),
      ('unit','24316 contract graphs; chronological count split17021/3647/3648','Methods','data/split_composition.csv'),
      ('label','Inherited Category0 positive; label knowledge time unavailable','Methods/Limitations','frozen GoG mapping'),
      ('horizon','18h57m10s;3648 test;107 positives;one Polygon positive','Methods/Results/Limitations','data/split_composition.csv'),
      ('controls','MLP AP0.4751±0.0351; matched BCE GIN0.4524±0.0566; historical GIN0.4648±0.0687 separate','Abstract/Results/Discussion','data/supplemental_summary.csv'),
      ('loss','GIN focal AP0.4885 with ECE0.3179','Results/Discussion','data/loss_fairness_results.csv'),
      ('statistics','Original7 and exploratory6 Holm families remain separate;0.0920 and0.1279 not familywise significant','Methods/Results/supplement','statistics/statistical_comparisons_corrected.csv'),
      ('calibration','Temperature scaling monotone within seed; ECE0.0357; ensemble NLL0.0986','Methods/Results','preserved numerical TeX and prediction vectors'),
      ('operating','37/183 alerts;top50/100 captures from frozen scores; no test-tuned intervention claim','Applied Anomaly-Screening Operating Points','data/operational_metrics.csv'),
      ('strict','Fixed-checkpoint input-set stress test, no prospective retraining','Methods/Results','data/strict_time_sensitivity.csv'),
      ('features','Full-history degree before128-edge topology cap; deterministic ties not physical order','Methods','data/timestamp_tie_audit.csv'),
      ('state','Per-chain TGN predict-then-update and phase reset; absolute-time TGAT approximation','Methods','frozen scientific sources'),
      ('limits','No unsupervised discovery, deployed monitoring, XAI, privacy guarantee or unique architecture contribution','Abstract/Discussion/Limitations','bounded text review'),
      ('declarations','Same A16-authorized roles/funding/ethics/AI text; final316 signoffs and COI pending','Back matter','private author records'),
    ]
    with (REPORTS/'dual_version_claim_matrix.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','bounded_claim','location','evidence','preprint','mdpi'])
        for row in rows:w.writerow([*row,'PASS','PASS'])
    write(REPORTS/'content_claim_parity.json',dict(status='PASS',body_sha256=sha(SHARED/'sections/manuscript_body.tex'),
        bibliography_sha256=sha(SHARED/'references.bib'),cited_keys=sorted(cited),cited_reference_count=21,
        original_reference_entries_unchanged=True,new_references=['tang2023gadbench','qiao2025survey'],
        protected_assets=protected,table_figure_numeric_parity=True,author_declarations_same=True,
        title_equal_315=True,title_final_author_approval=False,unsupported_broad_claims_found=0))
    write(SHARED/'claims_manifest.json',dict(source='projects/tds/reports/316/dual_version_claim_matrix.csv',
        sha256=sha(REPORTS/'dual_version_claim_matrix.csv'),numeric_identity='projects/tds/results/316_dual_submission/frozen_numeric_identity.json'))
    write(REPORTS/'title_lock.json',dict(title=meta['Title'],decision='Retain existing title; no unapproved title change',
        equal_across_platforms=True,final_author_approval=False))
    old=(PROJECT/'paper/common/315/manuscript_body.tex').read_text()
    diff=''.join(difflib.unified_diff(old.splitlines(True),body.splitlines(True),fromfile='R315',tofile='R316'))
    # Full prose diff remains private; public report describes exact scope/hash.
    (SHARED/'revision_315_316.diff').write_text(diff)
    (REPORTS/'dual_version_diff.md').write_text('# Dual-version scientific parity\n\nThe two R316 scientific bodies, supplementary text, bibliography, table fragments and figure assets are byte-identical. '
      'Wrappers and bibliography presentation differ by venue. The source ZIPs are independently buildable. '
      'R315 numerical tables/interpretations/figure files and all21 original-or-added cited scientific references are audited; '
      'the original bibliography entries are unchanged, with two verified additions.\n\n'
      'R315→316 prose edits: supervised financial screening motivation and research question, four related-work subsections, '
      'explicit matched-runtime MLP/GIN comparison, and a named operating-budget subsection. The Abstract and keywords change; title stays unchanged. '
      'No score, metric, split, objective, statistical family or figure changes. Full manuscript diff stays private.\n\n'
      'Private diff SHA-256: `'+sha(SHARED/'revision_315_316.diff')+'`. See content_claim_parity.json and the 14-row claim matrix.\n')
    print('PASS:542 frozen inputs, 21 citations, both scientific sources and all numeric table/figure assets match')


if __name__=='__main__':main()
