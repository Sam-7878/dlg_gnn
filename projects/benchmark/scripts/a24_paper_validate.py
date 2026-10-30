"""Author-local position-bound table/section and native PDF geometry checks."""
from a24_common import *
import re,xml.etree.ElementTree as ET,argparse

def norm(x):return re.sub(r'\s+','',x)
def table_region(t,label):
 a=t.index('\\label{'+label+'}');end=t.index(r'\end{table}',a);return t[a:end]
def check_source(t,expected):
 for label,spec in expected.items():
  region=table_region(t,label);body=region.split(r'\toprule',1)[1].split(r'\bottomrule',1)[0];body=body.replace(r'\midrule','')
  actual=[[c.strip() for c in line.split('&')] for line in body.split(r'\\') if line.strip()]
  wanted=[spec['headers']]+spec['rows'];assert len(actual)==len(wanted),(label,'rows',len(actual),len(wanted))
  for ri,(a,b) in enumerate(zip(actual,wanted)):
   assert len(a)==len(b),(label,ri,'columns')
   for ci,(x,y) in enumerate(zip(a,b)):assert norm(x)==norm(str(y)),(label,ri,ci,x,y)
 section=t.split(r'\subsection{Friedman Hypothesis Testing and Permutation Design}',1)[1].split(r'\section{Results and Empirical Analysis}',1)[0]
 for r in rows(PUB/'stats_master_a24.csv'):
  required=f"View {r['view_id']} uses $n={r['n_datasets']}$ datasets and $k={r['k_models']}$ models, giving $Q={float(r['Q']):.4f}$ and $W={float(r['Kendall_W']):.4f}$."
  assert required in section,('section stats',r['view_id'])
 assert 'five complete-case' not in section and '23.40' not in section
 a3=table_region(t,'tab:app_ablation_elliptic');assert 'F1' not in a3
 assert r'\ref{tab:a24_ap_sd}' in t.split(r'\label{tab:pr_auc_main}')[0].rsplit(r'\begin{table}',1)[-1],'AP SD link'
 assert 'competitive across all 3' not in t and 'hybrid transfer graphs' not in t
 for n,N,E in [('Ethereum','14,464','95,098'),('BSC','7,499','48,462'),('Polygon','2,353','15,368')]:
  row=next(l for l in table_region(t,'tab:dataset_properties').splitlines() if l.startswith(n+' &'));assert N in row and E in row
 return True

def main():
 expected=json.loads((LOCAL/'paper_cell_expectations.json').read_text());results=[];mutations=[]
 for v,stem in [('applsci','DLG-Benchmark_A24_AppliedSciences'),('preprints','DLG-Benchmark_A24_Preprints')]:
  p=PAPER/v/(stem+'.tex');t=p.read_text();check_source(t,expected)
  changes=[('section Q',t.replace('giving $Q=29.0357$','giving $Q=29.0358$',1)),('section n',t.replace('View S1 uses $n=8$','View S1 uses $n=9$',1)),('section W',t.replace('$W=0.6049$','$W=0.6048$',1))]
  for label,key in [('tab:friedman_stats','29.0357'),('tab:app_roc_auc_full_part2','0.5705'),('tab:a24_ap_sd_part2','0.1037'),('tab:app_ablation_elliptic','0.0254')]:
   r=table_region(t,label);assert key in r,(label,key);changes.append((label+' cell',t.replace(r,r.replace(key,str(float(key)+.0001),1),1)))
  r=table_region(t,'tab:app_ablation_elliptic');changes += [('A3 header',t.replace(r,r.replace('ROC-AUC','',1),1)),('A3 last value',t.replace(r,r.rsplit('&',1)[0]+r.rsplit('&',1)[1].replace('0.', '9.',1),1))]
  for name,mutant in changes:
   assert mutant!=t,name
   try:check_source(mutant,expected)
   except (AssertionError,ValueError,IndexError,KeyError):mutations.append({'variant':v,'mutation':name,'rejected':True})
   else:raise AssertionError('MUTATION NOT REJECTED '+name)
  pdf=p.with_suffix('.pdf');textpath=LOCAL/(v+'_native.txt');bbox=LOCAL/(v+'_bbox.html')
  subprocess.run(['pdftotext','-layout',str(pdf),str(textpath)],check=True);subprocess.run(['pdftotext','-bbox',str(pdf),str(bbox)],check=True)
  xml=ET.ElementTree(ET.fromstring(re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f]','',bbox.read_text())));pages=xml.findall('.//{http://www.w3.org/1999/xhtml}page');bounds=[]
  for pi,page in enumerate(pages,1):
   bad=[]
   for w in page.findall('.//{http://www.w3.org/1999/xhtml}word'):
    if float(w.attrib['xMin'])<-.5 or float(w.attrib['xMax'])>float(page.attrib['width'])+.5 or float(w.attrib['yMin'])<-.5 or float(w.attrib['yMax'])>float(page.attrib['height'])+.5:bad.append(w.text)
   bounds.append({'page':pi,'outside_page_words':bad})
  assert all(not x['outside_page_words'] for x in bounds),(v,bounds)
  native=textpath.read_text();a3page=next((i+1,s) for i,s in enumerate(native.split('\f')) if 'Historical Elliptic controls and primary references' in re.sub(r'\s+',' ',s))
  assert 'ROC-AUC' in a3page[1] and '0.5705' in a3page[1] and '0.0254' in a3page[1],(v,'A3 last-column PDF extraction')
  # Native table header and each last-column numeric component must survive on the table's page.
  for row in expected['tab:app_ablation_elliptic']['rows']:
   for number in re.findall(r'\d+\.\d+',row[-1]):assert number in a3page[1],(v,'A3 last value',number)
  results.append({'variant':v,'status':'PASS_POSITIONAL_SOURCE_NATIVE_PDF_AND_BOUNDS','pdf_sha256':sha(pdf),'pages':len(pages),'historical_control_table_page':a3page[0],'glyph_bounds':bounds,'manual_visual_review':'REQUIRED_SEPARATELY'})
 dump(PUB/'stat_text_consistency_a24.json',{'status':'PASS','tables_per_variant':len(expected),'mutation_tests':mutations,'documents':results})
 print('Paper position and native PDF checks PASS;',len(mutations),'negative mutations rejected')
if __name__=='__main__':main()
