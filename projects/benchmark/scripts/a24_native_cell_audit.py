"""A24 outer closure check: actual last-column PDF words, scoped by x/y coordinates."""
from a24_common import *
import re,xml.etree.ElementTree as ET

def check_last_column(words,expected):
 start=next(float(w['yMin']) for w in words if w['text']=='Historical')
 header=next(w for w in words if w['text']=='ROC-AUC' and float(w['yMin'])>start)
 x=float(header['xMin'])-3;y=float(header['yMin']);found=[]
 for w in words:
  if float(w['xMin'])>=x and float(w['yMin'])>y+4 and re.fullmatch(r'\d\.\d{4}',w['text']):found.append(w)
 # Seven rows, mean and SD in each actual last-column cell.
 assert len(found)==2*len(expected),('last-column token count',len(found),len(expected))
 for i,pair in enumerate(expected):
  a,b=found[2*i:2*i+2];assert (a['text'],b['text'])==tuple(pair),(i,a['text'],b['text'],pair)
  assert abs(float(a['yMin'])-float(b['yMin']))<2,('different table row',i)
 return {'header':header,'cells':[{'row_index':i+1,'mean':found[2*i],'sample_SD':found[2*i+1]} for i in range(len(expected))]}

r=json.loads((PUB/'stat_text_consistency_a24.json').read_text());e=json.loads((LOCAL/'paper_cell_expectations.json').read_text());expected=[re.findall(r'\d\.\d{4}',row[-1]) for row in e['tab:app_ablation_elliptic']['rows']];out=[]
for doc in r['documents']:
 v=doc['variant'];xml=ET.fromstring(re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f]','',(LOCAL/(v+'_bbox.html')).read_text()));pages=xml.findall('.//{http://www.w3.org/1999/xhtml}page');page=pages[doc['historical_control_table_page']-1]
 words=[dict(w.attrib,text=w.text or '') for w in page.findall('.//{http://www.w3.org/1999/xhtml}word')];proof=check_last_column(words,expected)
 remove_id=proof['cells'][-1]['sample_SD'];mutant=[w for w in words if w is not remove_id]
 try:check_last_column(mutant,expected)
 except AssertionError:rejected=True
 else:raise AssertionError('PDF last-cell deletion was not rejected')
 out.append({'variant':v,'PDF_sha256':doc['pdf_sha256'],'page':doc['historical_control_table_page'],'status':'PASS_NATIVE_LAST_COLUMN_CELL_COORDINATES','last_cell_deletion_mutation_rejected':rejected,**proof})
dump(PUB/'native_last_column_cells_a24.json',{'status':'PASS','documents':out});print('Exact native last-column cell coordinates and deletion mutations PASS')
