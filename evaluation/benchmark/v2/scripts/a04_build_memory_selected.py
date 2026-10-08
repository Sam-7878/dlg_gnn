#!/usr/bin/env python3
"""Publish only selected measured memory rows with original JSON and SHA-256."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'evaluation/benchmark/v2/paper_ready_final'
RAW=OUT/'publication_evidence_a04/memory_raw'


def main():
    rows=[]
    for p in sorted(RAW.glob('*.json')):
        d=json.loads(p.read_text())
        if not d.get('input_hashes') or not d.get('environment_lock_sha256'):continue
        rows.append({'dataset':d['dataset'],'model':d['model'],'seed':d['seed'],'envelope':d['envelope'],
            'device':d['device'],'physical_vram_bytes':d['physical_vram_bytes'],
            'allocator_cap_bytes':d['allocator_cap_bytes'],'support_status':d['status'],
            'peak_allocated_bytes':d['peak_allocated_bytes'],'peak_reserved_bytes':d['peak_reserved_bytes'],
            'train_time_sec':d.get('train_time_sec'),'total_wall_sec':d['total_wall_sec'],
            'run_id':d['run_id'],'raw_log_path':str(p.relative_to(ROOT)),
            'raw_log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'raw_score_sha256':d.get('raw_score_sha256')})
    path=OUT/'table_memory_selected_a04.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['dataset','model','seed','envelope','raw_log_path'])
        w.writeheader();w.writerows(rows)
    print(json.dumps({'measured_rows':len(rows),'pairs':sorted(set((r['dataset'],r['model']) for r in rows))},indent=2))
if __name__=='__main__':main()
