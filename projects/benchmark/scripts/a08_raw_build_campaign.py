#!/usr/bin/env python3
"""Sequential raw A/B reconstruction, with durable logs; no GPU training."""
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
facade=ROOT/'projects/benchmark/scripts/a08_data_repair.py'
for workspace in ['build_a','build_b']:
    for chain in ['polygon','bsc','ethereum']:
        subprocess.run([sys.executable,'-u',str(facade),'--stage','build','--chain',chain,'--workspace',workspace],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-u',str(facade),'--stage','audit'],cwd=ROOT,check=True)
print('RAW_REBUILD_AND_AUDIT_COMPLETE',flush=True)
