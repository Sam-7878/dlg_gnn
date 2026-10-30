from a24_common import *
import importlib.metadata as md,platform,shutil
rs=rows(REG);assert len(rs)==455;assert sum(r['dataset']!='LANL-RedTeam' for r in rs)==420
assert len({(r['dataset'],r['model']) for r in rs if r['dataset']!='LANL-RedTeam'})==84
assert sha(REG)=='45d4ca7b4a9b010aa779e7cb60a1816e0f1c08e98039a119d008a1d1cd5db907'
tag='benchmark-a23-preprint-2026-10-30';commit=subprocess.check_output(['git','rev-parse',tag+'^{}'],cwd=ROOT,text=True).strip();assert commit=='94b0abbb4c89405da3f3c4fab7121f62bbe0b827'
paths=[REG]+[p for b in [ROOT/'projects/benchmark/publication/a23',ROOT/'projects/benchmark/submission'] for p in b.rglob('*') if p.is_file() and p.suffix not in ['.aux','.log','.blg','.fls','.out','.bbl','.fdb_latexmk']]
files={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
for variant,stem in [('preprints','DLG-Benchmark_A16_Preprints'),('applsci','DLG-Benchmark_A16_AppliedSciences')]:
 src=ROOT/'projects/benchmark/submission'/variant;dst=LOCAL/'a23_snapshot'/variant;shutil.copytree(src,dst,dirs_exist_ok=True)
 dst2=PAPER/variant;shutil.copytree(src,dst2,dirs_exist_ok=True)
manifest={'gate':'A24-G0','status':'PASS','baseline_tag':tag,'baseline_commit':commit,'tag_object':subprocess.check_output(['git','rev-parse',tag],cwd=ROOT,text=True).strip(),'current_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'successful_runs':455,'primary_runs':420,'external_runs':35,'primary_supported_pairs':84,'primary_possible_pairs':91,'campaign_for_new_results':'A24_SUPPLEMENTARY','python':platform.python_version(),'dependencies':{p:md.version(p) for p in ['torch','torch-geometric','pygod','numpy','scipy','scikit-learn','psutil']},'files':files}
dump(PUB/'a24_frozen_input_manifest.json',manifest)
writecsv(PUB/'a24_issue_traceability.csv',[{'issue_id':'A24-'+str(i),'gate':g,'issue':s,'status':'OPEN'} for i,(g,s) in enumerate([('G1','stale README and crypto graph identity'),('G2','S1 n/Q/W conflict and inconsistent SD'),('G3','A3 clipped and F1 meaning uncertain'),('G4','overstated interpretation and related work'),('G5','feature-only comparison'),('G6','matched Elliptic controls'),('G7','measured resources and Cora equivalence'),('G8','real-domain alert scores'),('G9','optional GADNR diagnostic'),('G10','final paper and release verification')],1)])
print('A24-G0 PASS;',len(files),'frozen files;455 runs preserved')
