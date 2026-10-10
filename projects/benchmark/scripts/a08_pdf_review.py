#!/usr/bin/env python3
"""Render actual private PDFs and bind recorded manual observations to bytes.

Rendering and number matching never imply a manual semantic/layout PASS.
The --finalize action requires an already completed author-local observation
file for every page and explicit checked claim categories.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
E=ROOT/'projects/benchmark/evidence/a08_data_repair'
L=ROOT/'local_storage/benchmark/a08_data_repair'
R=ROOT/'projects/benchmark/reports/a08_data_repair'
REQUIRED_CLAIMS={'input_provenance','threshold_and_metric_policy','support_and_run_counts',
                 'statistical_scope','resource_claims','old_new_population_comparison',
                 'public_private_availability','historical_diagnostics','citations_and_equations'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    build=json.loads((L/'paper_build_manifest.json').read_text())
    if len(build['artifacts'])!=2:raise ValueError('actual neutral/MDPI build required')
    artifacts=[]
    for item in build['artifacts']:
        path=ROOT/item['path']
        if sha(path)!=item['sha256']:raise ValueError('actual PDF differs from build manifest')
        folder=L/'pdf_render'/item['sha256'];folder.mkdir(parents=True,exist_ok=True)
        if not args.finalize:
            subprocess.run(['pdftoppm','-png','-r','115',str(path),str(folder/'page')],check=True)
        images=sorted(folder.glob('page-*.png'))
        if len(images)!=item['pages']:raise ValueError('every actual page must be rendered')
        pages=[{'page':int(p.stem.split('-')[-1]),'render_path':p.relative_to(ROOT).as_posix(),
                'render_sha256':sha(p),'status':'NOT_REVIEWED','observations':[]} for p in images]
        artifacts.append({**item,'page_reviews':pages})
    render_manifest={'status':'RENDERED_NOT_REVIEWED','artifacts':artifacts,
                     'paper_build_manifest_sha256':sha(L/'paper_build_manifest.json')}
    if not args.finalize:
        dump(L/'pdf_render_manifest.json',render_manifest)
        print('Actual pages rendered; manual visual/semantic review has not passed')
        return
    manual=json.loads((L/'pdf_manual_observations.json').read_text())
    if manual.get('semantic_status')!='PASS' or manual.get('layout_status')!='PASS':raise ValueError('manual semantic/layout review incomplete')
    claims=manual.get('claim_reviews',[])
    if {row['category'] for row in claims}!=REQUIRED_CLAIMS or any(row.get('status')!='PASS' or not row.get('observations') for row in claims):
        raise ValueError('manual required claim categories incomplete')
    for artifact in artifacts:
        reviewed=next(a for a in manual['artifacts'] if a['path']==artifact['path'])
        if reviewed['sha256']!=artifact['sha256'] or len(reviewed['page_reviews'])!=artifact['pages']:raise ValueError('manual review bound to old/missing pages')
        for page in artifact['page_reviews']:
            actual=next(row for row in reviewed['page_reviews'] if row['page']==page['page'])
            if actual.get('render_sha256')!=page['render_sha256'] or actual.get('status')!='PASS' or not actual.get('observations'):
                raise ValueError('manual page/render identity or observation absent')
            page.update(status='PASS',observations=actual['observations'])
    # Every registered current display is independently resolved by the final
    # checker, then matched to both actual PDFs. No private TeX is exported.
    import importlib.util
    spec=importlib.util.spec_from_file_location('a08_checker',ROOT/'projects/benchmark/scripts/a08_final_check.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    displays=list(csv.DictReader((E/'paper_number_registry.csv').open()))
    if len(displays)<=20:raise ValueError('numeric expected list incomplete')
    rows=[]
    for artifact in artifacts:
        text=subprocess.check_output(['pdftotext',str(ROOT/artifact['path']),'-'],text=True)
        compact=''.join(text.split())
        for display in displays:
            checker.resolve_crosswalk_number(ROOT,display)
            if ''.join(display['rendered'].split()) not in compact:
                raise ValueError('registered number absent from actual PDF: '+display['claim_id'])
            rows.append({**display,'artifact_path':artifact['path'],'pdf_hash':artifact['sha256'],'check_status':'PASS'})
    with (E/'paper_number_crosswalk.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    review={'semantic_status':'PASS','layout_status':'PASS','artifacts':artifacts,'claim_reviews':claims,
            'manual_observation_sha256':sha(L/'pdf_manual_observations.json'),
            'scope':'actual every-page visual/semantic review and numeric source/lineage crosswalk; author submission approval remains separate'}
    checker.check_claim_map(ROOT,{a['path']:a['sha256'] for a in artifacts})
    dump(E/'audit/pdf_review.json',review)
    dump(E/'audit/G9.json',{'status':'PASS','pdf_review_sha256':sha(E/'audit/pdf_review.json'),
         'paper_number_crosswalk_sha256':sha(E/'paper_number_crosswalk.csv'),
         'paper_number_registry_sha256':sha(E/'paper_number_registry.csv'),
         'claim_evidence_map_sha256':sha(E/'claim_evidence_map.csv'),
         'scope':'actual both-PDF every-page review plus all registered numeric displays'})
    lines=['# A08 final PDF review','', 'Both artifacts below were rendered and reviewed page by page. Numeric displays were resolved to approved evidence and matched to each actual PDF. Final scientific acceptance and author approval remain separate.','']
    for artifact in artifacts:
        lines += ['## '+Path(artifact['path']).name,'','SHA256: `'+artifact['sha256']+'`','',str(artifact['pages'])+' pages.','', '| Page | Result | Actual review observations |','|---:|---|---|']
        for page in artifact['page_reviews']:
            lines.append(f"| {page['page']} | PASS | {'; '.join(page['observations']).replace('|','/')} |")
        lines.append('')
    lines += ['## Claim review','']
    lines += ['- '+row['category']+': '+'; '.join(row['observations']) for row in claims]
    (R/'Final_PDF_Review_A08.md').write_text('\n'.join(lines)+'\n')
    print('G9 actual page/claim/numeric review recorded; G10 and complete final checker remain required')


if __name__=='__main__':main()
