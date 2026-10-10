#!/usr/bin/env python3
"""A09 every-page renders and source-resolved numeric audit; never writes A08."""
import argparse,csv,hashlib,importlib.util,json,re,subprocess
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];L=ROOT/'local_storage/benchmark/a09_submission_closure';O=ROOT/'projects/benchmark/evidence/submission_closure_a09';R=ROOT/'projects/benchmark/reports/submission_closure_a09';E=ROOT/'projects/benchmark/evidence/a08_data_repair'
@lru_cache(maxsize=2000)
def sha(p):
 if p.is_dir():return hashlib.sha256(json.dumps({str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*')) if x.is_file()},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return hashlib.sha256(p.read_bytes()).hexdigest()
def compact(t):return ''.join(t.split()).replace('−','-')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--finalize',action='store_true');args=ap.parse_args();build=json.loads((L/'paper_build_manifest.json').read_text())
 spec=importlib.util.spec_from_file_location('preserved_a08_checker',ROOT/'projects/benchmark/scripts/a08_final_check.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
 displays=list(csv.DictReader((E/'paper_number_registry.csv').open()));cross=[];artifacts=[]
 for d in displays:checker.resolve_crosswalk_number(ROOT,d)
 added=[]
 def register(identifier,value,source,selector,formatting='exact'):
  added.append({'claim_id':identifier,'rendered':value,'source':source,'source_key':selector,'run_ids':'','allowed_rounding':formatting,'source_sha256':sha(O/source)})
 tests=list(csv.DictReader((O/'test_support_by_seed.csv').open()))
 for r in tests:
  for key in ['seed','test_N']:
   register('test/'+r['dataset']+'/'+r['seed']+'/'+key,r[key],'test_support_by_seed.csv',r['dataset']+'/'+r['seed']+'/'+key)
  register('test/'+r['dataset']+'/'+r['seed']+'/posneg',r['test_positive']+'/'+r['test_negative'],'test_support_by_seed.csv',r['dataset']+'/'+r['seed']+'/positive_negative')
  for key in ['pi_test_AP_reference','always_positive_F1']:register('test/'+r['dataset']+'/'+r['seed']+'/'+key,f"{float(r[key]):.4f}",'test_support_by_seed.csv',r['dataset']+'/'+r['seed']+'/'+key,'4 decimals')
 qual=list(csv.DictReader((O/'numerical_qualification_summary.csv').open()))
 for family,case in list(dict.fromkeys((r['family'],r['case']) for r in qual if not r['family'].startswith('historical'))):
  value=max(float(r['max_abs']) for r in qual if r['family']==family and r['case']==case);register('original_qualification/'+family+'/'+case,f'{value:.3e}','numerical_qualification_summary.csv',family+'/'+case+'/maximum_absolute_over_quantities','3 digits scientific')
 recovered=list(csv.DictReader((O/'qualification_recovery_summary.csv').open()))
 groups=list(dict.fromkeys((r['family'],r['case']) for r in recovered if r['family']!='GADNR scalar implemented term'))+[('GADNR scalar implemented term','all twelve finite cases')]
 for family,case in groups:
  rows=[r for r in recovered if r['family']==family and (r['case']==case or family=='GADNR scalar implemented term')]
  for key in ['max_abs','max_relative_with_floor','max_tolerance_fraction']:register('recovered/'+family+'/'+case+'/'+key,f"{max(float(r[key]) for r in rows):.3e}",'qualification_recovery_summary.csv',family+'/'+case+'/'+key,'3 digits scientific')
 num=json.loads((E/'number_registry.json').read_text());cells={(r['dataset'],r['model'],r['metric']):r for r in num['cells']};interpretation=[]
 for chain in ['Ethereum','BSC','Polygon']:
  rs=[r for r in tests if r['dataset']==chain];best=max(['DOMINANT','AnomalyDAE','CoLA','GADNR','OCGNN','DLG-Base','DLG-Aug'],key=lambda m:cells[chain,m,'pr_auc']['mean'])
  vals={'Base_AP':cells[chain,'DLG-Base','pr_auc']['mean'],'Aug_AP':cells[chain,'DLG-Aug','pr_auc']['mean'],'AP_difference':cells[chain,'DLG-Aug','pr_auc']['mean']-cells[chain,'DLG-Base','pr_auc']['mean'],'leader_AP':cells[chain,best,'pr_auc']['mean'],'mean_test_prevalence':sum(float(r['pi_test_AP_reference']) for r in rs)/5,'mean_always_positive_F1':sum(float(r['always_positive_F1']) for r in rs)/5,'Aug_test_F1':cells[chain,'DLG-Aug','validation_f1']['mean']}
  interpretation.append({'dataset':chain,'leader':best,'values':vals,'approved_run_ids':cells[chain,'DLG-Aug','pr_auc']['run_ids']})
 (O/'result_interpretation_numbers.json').write_text(json.dumps({'number_registry_sha256':sha(E/'number_registry.json'),'test_support_sha256':sha(O/'test_support_by_seed.csv'),'records':interpretation},indent=2)+'\n')
 for r in interpretation:
  for key,value in r['values'].items():register('interpretation/'+r['dataset']+'/'+key,f"{value:+.4f}" if key=='AP_difference' else f"{value:.4f}",'result_interpretation_numbers.json',r['dataset']+'/'+key,'4 decimals; difference signed')
 for item in build['artifacts']:
  pdf=ROOT/item['path'];assert sha(pdf)==item['sha256'];folder=L/'pdf_render'/item['sha256'];folder.mkdir(parents=True,exist_ok=True)
  if not args.finalize and len(list(folder.glob('page-*.png')))!=item['pages']:subprocess.run(['pdftoppm','-png','-r','105',str(pdf),str(folder/'page')],check=True)
  images=sorted(folder.glob('page-*.png'));assert len(images)==item['pages'];text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True);(folder/'extracted.txt').write_text(text);page_text=text.split('\f');whole=compact(text)
  for d in displays+added:
   needle=compact(d['rendered']);matches=[i+1 for i,t in enumerate(page_text) if needle in compact(t)]
   assert needle in whole,(item['path'],d['claim_id'],d['rendered'])
   cross.append({**d,'source_sha256':d['source_sha256'] if 'source_sha256' in d else sha(E/d['source']),'artifact_path':item['path'],'pdf_sha256':item['sha256'],'matching_pages':';'.join(map(str,matches)),'check_status':'PASS_SOURCE_RESOLVED_DISPLAY'})
  # LaTeX-generated final logs, not warnings from earlier compilation passes.
  log=pdf.with_suffix('.log').read_text(errors='replace');assert 'undefined references' not in log and 'undefined on input' not in log and '\n! ' not in log
  artifacts.append({**item,'page_reviews':[{'page':int(p.stem.split('-')[-1]),'render_path':str(p.relative_to(ROOT)),'render_sha256':sha(p),'status':'NOT_REVIEWED','observations':[]} for p in images]})
 with (O/'paper_number_crosswalk.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(cross[0]));w.writeheader();w.writerows(cross)
 report={'status':'RENDERED_NUMBERS_CHECKED_NOT_MANUALLY_REVIEWED','artifacts':artifacts,'source_resolved_crosswalk_rows':len(cross),'primary_registry_displays_unchanged':len(displays),'additional_support_qualification_displays':len(added),'crosswalk_sha256':sha(O/'paper_number_crosswalk.csv')}
 if args.finalize:
  manual=json.loads((L/'pdf_manual_observations.json').read_text());assert manual['semantic_status']=='PASS' and manual['layout_status']=='PASS'
  for a in artifacts:
   b=next(x for x in manual['artifacts'] if x['path']==a['path']);assert b['sha256']==a['sha256'] and len(b['page_reviews'])==a['pages']
   for row in a['page_reviews']:
    r=next(x for x in b['page_reviews'] if x['page']==row['page']);assert r['render_sha256']==row['render_sha256'] and r['status']=='PASS' and r['observations'];row.update(r)
  assert all(r['status']=='PASS' and r['observations'] for r in manual['claim_reviews']);report.update(status='PASS_LOCAL_PDF_REVIEW',claim_reviews=manual['claim_reviews'],manual_observation_sha256=sha(L/'pdf_manual_observations.json'),approval='PENDING_AUTHOR',public_release='NOT_PUBLISHED')
  (O/'pdf_review.json').write_text(json.dumps(report,indent=2)+'\n')
 else:(L/'pdf_render_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(report['status'],len(cross),'crosswalk rows;',sum(a['pages'] for a in artifacts),'actual rendered pages')
if __name__=='__main__':main()
