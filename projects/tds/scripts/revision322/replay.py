"""Verify a numerical package and replay tie-aware ranking and fixed budgets."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def measure(label, score):
    order = np.argsort(-score, kind='stable')
    ap = average_precision_score(label, score)
    return dict(n=len(label), positive=int(label.sum()), prevalence=float(label.mean()),
                ap=float(ap), roc_auc=float(roc_auc_score(label, score)), ap_lift=float(ap-label.mean()),
                **{f'capture_{k}':int(label[order[:k]].sum()) for k in (37,50,100)})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('bundle',type=Path)
    args=parser.parse_args()
    root=args.bundle
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Checksum mismatch: {name}')
    groups=defaultdict(list)
    for row in manifest['runs']:
        with np.load(root/row['array']) as data:
            metrics=measure(data['label'],data['score'])
        groups[(row['cohort'],row['arm'])].append(metrics)
        for key,value in metrics.items():
            if not np.isclose(value,row[key],atol=1e-12,rtol=0):
                raise ValueError(f"{row['array']} {key} mismatch")
    with (root/'score_summary.csv').open(newline='') as stream:
        summaries=list(csv.DictReader(stream))
    assert len(summaries)==len(groups)
    for row in summaries:
        runs=groups[(row['cohort'],row['arm'])]
        assert len(runs)==int(row['runs'])
        for metric in ('ap','roc_auc','ap_lift','capture_37','capture_50','capture_100'):
            values=[m[metric] for m in runs]
            assert np.isclose(np.mean(values),float(row[f'{metric}_mean']),atol=1e-12,rtol=0)
            if len(values)>1:
                assert np.isclose(np.std(values,ddof=1),float(row[f'{metric}_std']),atol=1e-12,rtol=0)
    print(f"PASS: {len(manifest['runs'])} held-out score arrays")


if __name__=='__main__':
    main()
