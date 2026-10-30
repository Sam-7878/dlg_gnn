"""Freeze current public scientific evidence. Final outer reports are separate."""
from a24_common import *
paths=set()
for folder in [PUB,SUP]:
 for p in folder.rglob('*'):
  if p.is_file() and p.name not in ['public_payload_manifest.json','release_retrieval.json','stat_text_consistency_a24.json','clean_source_recompile_a24.json','fresh_clone_verification.json','FINAL_REPORT_A24.md','pdf_page_review_a24.json','a24_issue_traceability.csv','a24_gate_summary.json']:paths.add(p)
paths.add(REG)
for p in (ROOT/'projects/benchmark/scripts').glob('a24_*.py'):
 if 'build_manuscript' not in p.name:paths.add(p)
for folder in [ROOT/'src/gog_fraud/models',ROOT/'src/gog_fraud/data']:
 for p in folder.rglob('*.py'):paths.add(p)
for name in ['configs/benchmark/a08_crypto_clean_v1.yaml','environment/locks/benchmark-a03-cuda.lock.txt']:paths.add(ROOT/name)
dump(PUB/'public_payload_manifest.json',{'scope':'A24 public scientific inputs/scalars/source;raw-score and private PDF review are separate','files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}})
print('Public manifest:',len(paths),'files')
