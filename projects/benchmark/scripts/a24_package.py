"""Author-local source ZIP, clean rebuild and render. No publication or training."""
from a24_common import *
import zipfile,os,tempfile,shutil,re
out=[]
for v,stem in [('applsci','DLG-Benchmark_A24_AppliedSciences'),('preprints','DLG-Benchmark_A24_Preprints')]:
 folder=PAPER/v;wanted=[folder/(stem+'.tex'),folder/'references.bib']
 wanted += [p for p in (folder/'Definitions').rglob('*') if p.is_file()]
 wanted += [folder/p for p in re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',(folder/(stem+'.tex')).read_text())]
 wanted += [p for p in [folder/'soul.sty',folder/'soul-ori.sty'] if p.is_file()]
 zip_path=folder/'source_bundle_A24.zip'
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sorted(wanted):
   info=zipfile.ZipInfo(str(p.relative_to(folder)),(2026,10,30,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
 dest=Path(tempfile.mkdtemp(prefix='a24-clean-'+v+'-'));zipfile.ZipFile(zip_path).extractall(dest)
 with (LOCAL/(v+'_clean_build.log')).open('w') as log:
  subprocess.run(['latexmk','-g','-pdf','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],cwd=dest,env=dict(os.environ,SOURCE_DATE_EPOCH='1793318400',FORCE_SOURCE_DATE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 assert sha(dest/(stem+'.pdf'))==sha(folder/(stem+'.pdf')),v+' clean PDF not byte-identical'
 rendered=LOCAL/'rendered'/v;rendered.mkdir(parents=True,exist_ok=True)
 subprocess.run(['pdftoppm','-r','95','-png',str(folder/(stem+'.pdf')),str(rendered/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 out.append({'variant':v,'source_zip_sha256':sha(zip_path),'PDF_sha256':sha(folder/(stem+'.pdf')),'clean_pdf_sha256':sha(dest/(stem+'.pdf')),'clean_compile_exit':0,'byte_identical':True,'source_inventory':{str(p.relative_to(folder)):sha(p) for p in wanted},'rendered_pages':len(list(rendered.glob('page-*.png'))),'clean_directory':str(dest)})
 dump(PUB/'clean_source_recompile_a24.json',{'status':'PASS','documents':out})
 print(v,'clean ZIP build and all-page rendering PASS',flush=True)
