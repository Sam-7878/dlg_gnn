#!/usr/bin/env python3
"""Verify sibling GoG artifacts without unpickling or loading large tensors.

Only the known little-endian int64, interleaved edge layout is supported.
Label-graph uniqueness is checked with a bounded N*N bitset and 4 MiB reads.
The result distinguishes observed construction constraints from producer history.
"""
import argparse
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path

import numpy as np

from audit_hybrid_tensors import decode, digest

ROOT = Path(__file__).resolve().parents[3]
INTEGER_OPS = {'BININT', 'BININT1', 'BININT2'}
MEMO_OPS = {'BINPUT', 'LONG_BINPUT', 'MEMOIZE'}


def scalar(ops, key):
    index = next(i for i, (_, value, _) in enumerate(ops) if value == key)
    for op, value, _ in ops[index + 1:]:
        if op.name in MEMO_OPS:
            continue
        if op.name in INTEGER_OPS or op.name == 'BINUNICODE':
            return value
        raise ValueError(f'unsupported scalar {key}: {op.name}')
    raise ValueError(f'missing scalar {key}')


def inspect_metadata(path, n):
    with zipfile.ZipFile(path) as archive:
        ops = list(pickletools.genops(archive.read(next(
            p for p in archive.namelist() if p.endswith('/data.pkl')))))
        start = next(i for i, (op, _, _) in enumerate(ops)
                     if op.name == 'BINPERSID')
        dims = [a for op, a, _ in ops[start + 1:start + 11]
                if op.name in INTEGER_OPS]
        if len(dims) != 5 or dims[:2] != [0, 2] or dims[3:] != [1, 2]:
            raise ValueError('unsupported edge storage offset/shape/stride')
        edge_member = next(p for p in archive.namelist()
                           if p.endswith('/data/0'))
        if archive.getinfo(edge_member).file_size != dims[2] * 16:
            raise ValueError('edge storage length mismatch')
        endian = next((p for p in archive.namelist() if p.endswith('/byteorder')), None)
        if endian and archive.read(endian) != b'little':
            raise ValueError('unsupported byte order')
        features = [a for op, a, _ in ops if op.name == 'BINUNICODE'
                    and isinstance(a, str) and len(a) == n * 8 * 8]
        if len(features) != 1:
            raise ValueError('unsupported NumPy feature layout')
        x_bytes = features[0].encode('latin1')
        y_bytes = archive.read(next(p for p in archive.namelist()
                                    if p.endswith('/data/1')))
        y = np.frombuffer(y_bytes, dtype='<i8')
        if y.shape != (n,) or not np.isin(y, [0, 1]).all():
            raise ValueError('unsupported label encoding')
        begin = next(i for i, (_, a, _) in enumerate(ops) if a == 'contract_to_idx')
        end = next(i for i, (_, a, _) in enumerate(ops) if a == 'idx_to_contract')
        names = [Path(a).name for op, a, _ in ops[begin:end]
                 if op.name == 'BINUNICODE' and isinstance(a, str) and '/graphs/' in a]
        if len(names) != n or scalar(ops, 'num_nodes') != n:
            raise ValueError('node mapping mismatch')
        meta = dict(method=scalar(ops, 'method'), recorded_k=scalar(ops, 'k'),
                    edge_count=dims[2], file_bytes=path.stat().st_size,
                    feature_bytes_sha256=hashlib.sha256(x_bytes).hexdigest(),
                    label_bytes_sha256=hashlib.sha256(y_bytes).hexdigest())
        return meta, y, x_bytes, names, edge_member


def stream_label_edges(path, member, n, y):
    # Bits encode directed edges, so duplicates across blocks cannot be hidden.
    seen = np.zeros((n * n + 7) // 8, dtype=np.uint8)
    outgoing = np.zeros(n, dtype=np.int64)
    count = same = loops = duplicates = 0
    storage_hash = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
        while block := stream.read(4 * 1024 * 1024):
            if len(block) % 16:
                raise ValueError('truncated edge pair')
            storage_hash.update(block)
            edges = np.frombuffer(block, dtype='<i8').reshape(-1, 2)
            if not np.all((edges >= 0) & (edges < n)):
                raise ValueError('invalid edge bounds')
            a, b = edges[:, 0], edges[:, 1]
            count += len(a)
            same += int(np.count_nonzero(y[a] == y[b]))
            loops += int(np.count_nonzero(a == b))
            outgoing += np.bincount(a, minlength=n)
            codes = np.unique(a * n + b)
            duplicates += len(a) - len(codes)
            slots = codes // 8
            bits = np.left_shift(np.uint8(1), (codes % 8).astype(np.uint8))
            duplicates += int(np.count_nonzero(seen[slots] & bits))
            np.bitwise_or.at(seen, slots, bits)
    class_counts = np.bincount(y, minlength=2)
    expected = int(np.sum(class_counts * (class_counts - 1)))
    complete = (count == expected and same == count and loops == 0
                and duplicates == 0
                and np.array_equal(outgoing, class_counts[y] - 1))
    return dict(edge_count=count, expected_same_label_complete_edges=expected,
                same_label_edges=same, self_loops=loops, duplicate_edges=duplicates,
                complete_same_label_directed_graph=bool(complete),
                edge_storage_sha256=storage_hash.hexdigest(),
                bitset_bytes=int(seen.nbytes), read_block_bytes=4 * 1024 * 1024)


def histogram(values):
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(keys, counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'projects/benchmark/evidence/astra_revision/gog_graph_family_audit.json')
    args = parser.parse_args()
    frozen = json.loads((ROOT / 'projects/benchmark/evidence/astra_revision/hybrid_tensor_audit.json').read_text())
    results = []
    for old in frozen['results']:
        chain = old['dataset'].lower()
        n = old['n_nodes']
        base = args.data_root / chain
        paths = {kind: base / f'{chain}_{kind}_graph.pt'
                 for kind in ('knn', 'label', 'hybrid')}
        metadata = {kind: inspect_metadata(p, n) for kind, p in paths.items()}
        ref = metadata['hybrid']
        for kind, values in metadata.items():
            if not (np.array_equal(values[1], ref[1])
                    and values[2] == ref[2] and values[3] == ref[3]):
                raise ValueError(f'{chain}: sibling feature/label/node map differs')
            values[0]['file_sha256'] = digest(paths[kind])
        if ref[0]['file_sha256'] != old['hybrid_file_sha256']:
            raise ValueError('not the canonical frozen hybrid input')
        print(f'{chain}: checking complete label graph in bounded chunks', flush=True)
        label = stream_label_edges(paths['label'], metadata['label'][4], n, ref[1])
        if not label['complete_same_label_directed_graph']:
            raise ValueError('label graph is not complete as reported')
        knn = decode(paths['knn'], n, feature=False)
        hybrid = decode(paths['hybrid'], n, feature=False)
        extra = np.setdiff1d(hybrid['codes'], knn['codes'])
        out_extra = np.bincount(extra // n, minlength=n)
        out_knn = np.bincount(knn['edges'][0], minlength=n)
        extra_loops = int(np.count_nonzero(extra // n == extra % n))
        same_knn = knn['codes'][(ref[1][knn['codes'] // n] == ref[1][knn['codes'] % n])
                                & (knn['codes'] // n != knn['codes'] % n)]
        overlap_candidates = np.bincount(same_knn // n, minlength=n)
        three_feasible = (extra_loops == 0 and np.all(out_extra <= 3)
                          and np.all(overlap_candidates >= 3 - out_extra))
        removed = np.setdiff1d(knn['codes'], hybrid['codes'])
        sample = []
        x = np.frombuffer(ref[2], dtype='<f8').reshape(n, 8)
        for i in (0, n // 2, n - 1):
            p = base / 'graphs' / ref[3][i]
            data = json.loads(p.read_text())
            sample.append(dict(node=i, json_basename=p.name, json_sha256=digest(p),
                               feature_column_3=float(x[i, 3]),
                               json_edge_count=len(data.get('edges', [])),
                               json_feature_rows=len(data.get('features', [])),
                               tx_count_key_present='tx_count' in data,
                               contract_feature_length=len(data.get('contract_feature', []))))
        row = dict(dataset=old['dataset'], nodes=n,
                   class_counts=np.bincount(ref[1], minlength=2).tolist(),
                   artifacts={k: v[0] for k, v in metadata.items()},
                   all_sibling_feature_bytes_labels_node_names_equal=True,
                   label_graph=label, knn_out_degree_histogram=histogram(out_knn),
                   added_out_degree_histogram=histogram(out_extra),
                   n_added_edges=len(extra), removed_knn_edges=len(removed),
                   added_self_loops=extra_loops,
                   added_same_label_edges=int(np.count_nonzero(ref[1][extra // n] == ref[1][extra % n])),
                   three_per_node_hypothesis=dict(expected_slots=3*n,
                        overlap_slots_if_three_were_sampled=int(3*n-len(extra)),
                        feasible=bool(three_feasible),
                        nodes_without_enough_same_label_knn_overlap_candidates=int(np.count_nonzero(overlap_candidates < 3 - out_extra)),
                        identified_original_sampling_seed_or_rule=False),
                   constant_zero_feature_columns=np.flatnonzero((x == 0).all(0)).tolist(),
                   feature_column_3_range=[float(x[:, 3].min()), float(x[:, 3].max())],
                   feature_semantics='UNVERIFIED: tx_count is asserted in the supplied report but no producer/source-to-feature mapping is provided; sampled current JSON counts differ.',
                   json_feature_counterexamples=sample)
        results.append(row)
        print(f"{chain}: PASS complete-label graph; added degree {row['added_out_degree_histogram']}", flush=True)
    report = dict(schema_version=1, status='ARTIFACT_CONSTRAINTS_VERIFIED_UNSUPERVISED_USE_NOT_CLEARED',
                  source_sha256=digest(Path(__file__)), method='Non-executing pickle opcode and ZIP storage inspection; exhaustive bounded label-edge validation; canonical hybrid hash match.',
                  results=results,
                  interpretation='The label artifact is exactly a same-class complete graph, and hybrid contains knn plus same-class additions compatible with up to three extra neighbors per node. This strongly supports label-informed construction and warrants excluding these inputs from claims of independently verified unsupervised financial-fraud evaluation. Artifact consistency does not uniquely establish original sampling code/seed, feature definition, or historical execution.',
                  scope='No frozen tensors/results overwritten, no training launched.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'WROTE {args.output}', flush=True)


if __name__ == '__main__':
    main()
