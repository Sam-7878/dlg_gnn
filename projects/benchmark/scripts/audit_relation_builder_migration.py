#!/usr/bin/env python3
"""Read-only audit of the known GoG Level2 migration; no pickle execution.

Parses the recorded contiguous tensor layout from torch ZIP storages and checks
the CPU production relation API on small counterfactual label fixtures.
It does not regenerate graphs, train detectors, or replace frozen evidence.
"""
import argparse
import hashlib
import json
import pickletools
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from gog_fraud.data.level2.relation_builder import RelationBuilderConfig, build_level2_graph

INT_OPS = {'BININT', 'BININT1', 'BININT2'}
FIELDS = {'x', 'edge_index', 'edge_attr', 'y', 'graph_id', 'level1_embedding',
          'level1_score', 'level1_logits', 'level1_label', 'labels', 'embeddings'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_field(archive, ops, field, dtype):
    start = next(i for i, (op, a, _) in enumerate(ops)
                 if op.name == 'BINUNICODE' and a == field)
    end = next((i for i in range(start + 1, len(ops))
                if ops[i][0].name == 'BINUNICODE' and ops[i][1] in FIELDS), len(ops))
    fragment = ops[start:end]
    pivot = next(i for i, (op, _, _) in enumerate(fragment) if op.name == 'BINPERSID')
    ids = [a for op, a, _ in fragment[:pivot]
           if op.name == 'BINUNICODE' and a.isdigit()]
    if len(ids) != 1:
        raise ValueError('unsupported storage reference: ' + field)
    integers = [a for op, a, _ in fragment[pivot + 1:pivot + 12]
                if op.name in INT_OPS]
    if not integers or integers[0] != 0 or len(integers) not in (3, 5):
        raise ValueError('unsupported tensor layout: ' + field)
    shape = tuple(integers[1:2] if len(integers) == 3 else integers[1:3])
    stride = integers[2:] if len(integers) == 3 else integers[3:]
    if stride != ([1] if len(shape) == 1 else [shape[1], 1]):
        raise ValueError('noncontiguous tensor layout: ' + field)
    member = next(p for p in archive.namelist() if p.endswith('/data/' + ids[0]))
    raw = archive.read(member)
    if len(raw) != int(np.prod(shape)) * np.dtype(dtype).itemsize:
        raise ValueError('unexpected storage bytes: ' + field)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'projects/benchmark/evidence/astra_revision/relation_builder_migration_audit.json')
    args = parser.parse_args()
    previous = json.loads((ROOT / 'projects/benchmark/evidence/astra_revision/gog_graph_family_audit.json').read_text())
    prefix = 'evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/'
    with zipfile.ZipFile(ROOT / 'projects/benchmark/evidence/public_numeric_evidence.zip') as archive:
        canonical = json.loads(archive.read(prefix + 'dataset_manifest_canonical.json'))
    results = []
    for old in previous['results']:
        chain = old['dataset'].lower()
        path = args.data_root / chain / f'{chain}_level2_graph.pt'
        with zipfile.ZipFile(path) as archive:
            ops = list(pickletools.genops(archive.read(next(
                p for p in archive.namelist() if p.endswith('/data.pkl')))))
            x = numeric_field(archive, ops, 'x', '<f4')
            edges = numeric_field(archive, ops, 'edge_index', '<i8')
            y = numeric_field(archive, ops, 'y', '<f4')
            embedding = numeric_field(archive, ops, 'level1_embedding', '<f4')
            labels = numeric_field(archive, ops, 'labels', '<i8')
        n = x.shape[0]
        if embedding.shape != (n, 8) or edges.shape[0] != 2:
            raise ValueError('unexpected current artifact layout')
        if not np.all((edges >= 0) & (edges < n)):
            raise ValueError('edge bounds')
        normalized = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        all_same_direction = bool(np.isfinite(normalized).all()
                                  and np.array_equal(normalized, np.broadcast_to(normalized[0], normalized.shape)))
        # Verified legacy values were float64; migrated integral values survive
        # float32 casting exactly. Compare the actual earlier feature-byte hash.
        inherited = hashlib.sha256(embedding.astype('<f8').tobytes()).hexdigest() == old['artifacts']['hybrid']['feature_bytes_sha256']
        labels_inherited = hashlib.sha256(labels.tobytes()).hexdigest() == old['artifacts']['hybrid']['label_bytes_sha256']
        count = min(128, n)
        emb = torch.from_numpy(embedding[:count])
        bundle = dict(embedding=emb, score=torch.zeros(count, 1),
                      logits=torch.zeros(count, 1), graph_id=torch.arange(count))
        cfg = RelationBuilderConfig(relation_modes=['embedding_knn'], knn_k=5, knn_similarity='cosine')
        a = build_level2_graph(dict(bundle, label=torch.zeros(count, 1)), cfg)
        b = build_level2_graph(dict(bundle, label=torch.ones(count, 1)), cfg)
        independent = bool(torch.equal(a.edge_index, b.edge_index)
                           and torch.equal(a.edge_attr, b.edge_attr) and torch.equal(a.x, b.x))
        row = next(r for r in canonical if r['dataset_id'] == old['dataset'])
        results.append(dict(dataset=old['dataset'], new_file=str(path), new_sha256=digest(path),
                            nodes=n, edges=int(edges.shape[1]), x_shape=list(x.shape),
                            y_shape=list(y.shape), y_values=y.tolist(), labels_shape=list(labels.shape),
                            node_label_contract_pass=bool(y.size == n),
                            labels_class_counts=np.bincount(labels, minlength=2).tolist(),
                            legacy_feature_values_unchanged=inherited, legacy_label_values_unchanged=labels_inherited,
                            constant_zero_embedding_columns=np.flatnonzero((embedding == 0).all(0)).tolist(),
                            cosine_all_node_directions_identical=all_same_direction,
                            normalized_reference_direction=normalized[0].tolist(),
                            old_canonical_raw_source=row['raw_source_path'], old_canonical_raw_sha256=row['raw_source_sha256'],
                            current_matches_paper_canonical_input=digest(path) == row['raw_source_sha256'],
                            old_canonical_edges=row['graph_edges'], old_canonical_feature_dimension=row['feature_dimension'],
                            removed_legacy_files={kind: not (args.data_root / chain / f'{chain}_{kind}_graph.pt').exists()
                                                  for kind in ('hybrid', 'knn', 'label')},
                            counterfactual_label_probe=dict(nodes=count, labels_all_zero_vs_all_one=True,
                                 topology_weights_features_unchanged=independent,
                                 graph_y_changed=bool(not torch.equal(a.y, b.y))),
                            repeat_builder_extraction_would_take_graph_y_label_count=int(y.size)))
    paths = ['scripts/build_clean_level2_graphs.py', 'src/gog_fraud/data/level2/relation_builder.py',
             'evaluation/benchmark/v2/scripts/a03_run_crypto_production.py',
             'evaluation/benchmark/v2/scripts/a03_rerun_crypto_dlg_aug.py']
    report = dict(schema_version=1, date='2026-10-09', status='FINAL_SCIENTIFIC_AUDIT_NOT_PASSED',
                  method='Non-executing ZIP storage/pickle opcode parsing; trusted repository API CPU fixtures only.',
                  source_sha256=digest(Path(__file__)), source_hashes={p: digest(ROOT / p) for p in paths},
                  results=results,
                  component_status=dict(direct_label_use_in_edge_builder='PASS for fixed embeddings in inspected mode and counterfactual fixtures',
                     node_evaluation_label_contract='FAIL: graph-level y has one entry',
                     feature_provenance='UNRESOLVED: same legacy values inherited',
                     cosine_relation_information='DEGENERATE: every normalized embedding equals the same vector',
                     rebuild_from_provider_data='FAIL: generator needs an existing graph artifact',
                     repeat_generation_label_contract='FAIL: existing Data.y preferred over Data.labels',
                     manuscript_result_input_identity='FAIL: canonical results still refer to old hybrid hashes'),
                  scope='No graph rewrite, original-file restoration, neural training, paper rebuild or push performed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], component_status=report['component_status'],
                         datasets=[{k: r[k] for k in ('dataset', 'nodes', 'edges', 'y_shape', 'legacy_feature_values_unchanged', 'cosine_all_node_directions_identical')}
                                   for r in results]), indent=2))


if __name__ == '__main__':
    main()
