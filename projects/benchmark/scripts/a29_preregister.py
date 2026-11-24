"""A29 pre-registration. Run once, BEFORE any new A29 test metric exists.

Writes results/a29/preregistration/A29_PROTOCOL.yaml and a binding file with
its SHA-256, the git HEAD and the hashes of the scripts that will execute it.
"""
import subprocess
import time

import yaml

from a29_common import RESULTS, ROOT, dump, sha

OUT = RESULTS / 'preregistration'
OUT.mkdir(parents=True, exist_ok=True)

common = {
    'frozen': 'The 455 primary runs, the 91-pair support matrix, primary means and primary ranks are not changed. '
              'Every A29 result is an extension reported in separate tables.',
    'labels': 'Labels never enter fit(). Validation labels may select a threshold or a configuration '
              '(declared as semi-supervised model selection). Test labels are read once after scores are fixed.',
    'score_direction': 'Higher is more anomalous for every detector; no sign flips after looking at labels.',
    'device': 'RTX 3090 pinned by UUID GPU-ab53069a-3217-ee09-80f8-748b5fdbd156; missing measurements stay missing.',
    'metrics': 'AP, ROC-AUC, P@1%, P@5%, R@1%, R@5%, lift = P@K / test prevalence (A24 evaluate_scores: ceil budgets, stable ties).',
    'seeds': [42, 43, 44, 45, 46],
    'primary_model_config': {
        'GNN common': 'epoch=50, batch_size=0 (full graph), verbose=0',
        'DOMINANT/DLG-Base/DLG-Aug': 'Shared* exact_sparse Gram backend, sparse_fused messages, hid_dim=64, num_layers=4, lr=0.004; DLG-Aug l1_epochs=20',
        'AnomalyDAE': 'SharedAnomalyDAE chunked_exact, score_chunk_size=256, PyGOD defaults otherwise',
        'CoLA/OCGNN/GADNR': 'PyGOD 1.1.0 classes (GADNR: project copy without worker pool), num_neigh=-1, backbone=AutoSparseFusedGCN, PyGOD defaults otherwise',
    },
}

plans = {
    'A29-06_gadnr_retry': {
        'cells': ['Elliptic', 'DGraphFin', 'Yelp-Syn', 'Reddit-Syn', 'Flickr-Syn'],
        'seed': 42, 'config': 'primary GADNR configuration, unchanged',
        'cap_hours': 12,
        'telemetry': 'phase wall times (load, process_graph, each epoch, scoring), sampled host RSS, CUDA allocated/reserved peaks, heartbeat JSON every epoch',
        'stop_rule': 'After each completed epoch, if elapsed + remaining_epochs * mean_epoch_seconds > cap, stop and classify '
                     'EXCEEDS_CAP_PROJECTED with the measured epoch times. If a single phase passes the cap, classify TIMEOUT_OBSERVED.',
        'classes': ['COMPLETED', 'GPU_OOM', 'HOST_OOM', 'TIMEOUT_OBSERVED', 'EXCEEDS_CAP_PROJECTED', 'OTHER_ERROR'],
        'if_completed': 'run seeds 43-46 with the same configuration and report as an extension (primary 84/91 unchanged).',
    },
    'A29-05a_dense_vs_gram': {
        'graphs': ['Cora-Syn', 'CiteSeer-Syn', 'PubMed-Syn', 'Amazon-Syn', 'BitcoinOTC', 'Ethereum', 'BSC', 'Polygon', 'Elliptic (attempt; OOM recorded)'],
        'pairs': {'DOMINANT': 'PyGOD dense DOMINANT vs SharedDOMINANT exact_sparse',
                  'DLG-Base': 'dense DLG vs SharedDLGBase exact_sparse'},
        'init': 'Gram model initial state_dict copied into the dense model before training (identical start).',
        'seeds': [42], 'epochs': 50,
        'noise_floor': 'dense model trained twice with the same seed and initial state',
        'report': ['max relative loss difference over epochs', 'Spearman rho of final scores', 'top-1% and top-5% overlap', 'delta AP', 'delta ROC-AUC', 'peak memory of both paths'],
    },
    'A29-05b_scaling_sweep': {
        'graph': 'synthetic undirected random graph, mean degree 10 (each undirected edge stored in both directions), 64 standard-normal features, generator seed 0',
        'N': [5000, 10000, 20000, 50000, 100000, 200000],
        'models': ['DOMINANT-dense (PyGOD)', 'DOMINANT-Gram', 'DLG-Base-Gram', 'DLG-Aug-Gram', 'AnomalyDAE-rowblock'],
        'epochs': 'DLG-Aug 3 local + 3 global; others 3',
        'report': 'peak allocated/reserved CUDA bytes, mean epoch seconds; OOM marked and larger N skipped; log-log slope of peak allocated vs N fitted per model on completed points',
    },
    'A29-05c_resource_matrix': {
        'graphs': ['Cora-Syn', 'Elliptic', 'Reddit-Syn', 'DGraphFin'], 'models': '7 primary models', 'seed': 42,
        'config': 'primary configuration, full epoch budget',
        'report': 'peak allocated/reserved, fit seconds, score seconds, host RSS peak; GADNR cells reuse A29-06 telemetry; failures recorded with cause and cap',
        'cap_hours_per_cell': 3,
    },
    'A29-04_baseline_sanity': {
        'step1_cola': {
            'graphs': ['Cora-Syn', 'CiteSeer-Syn', 'PubMed-Syn'],
            'variants': {
                'PyGOD-CoLA-primary': 'primary configuration (epoch 50, single scoring round)',
                'PyGOD-CoLA-paperlike': 'epoch 100, lr 0.001, hid_dim 64, scores averaged over 256 decision_function rounds',
                'CoLA-RWR-reference': 'compact re-implementation of the original design: random-walk-with-restart subgraph of size 4, '
                                      'anonymised target node, 1-layer GCN, average readout, bilinear discriminator, batch 300, '
                                      'epoch 100, lr 0.001, score averaged over 256 rounds',
            },
            'degeneracy_checks': ['loss trajectory (collapse to ~0 or constant)', 'score SD and fraction of tied scores'],
        },
        'step1_ocgnn': {'graphs': ['Cora-Syn', 'CiteSeer-Syn', 'PubMed-Syn'],
                        'checks': ['score SD', 'fraction of nodes inside radius', 'distance-to-centre spread']},
        'step2_equal_budget_tuning': {
            'graphs': ['Cora-Syn', 'CiteSeer-Syn', 'PubMed-Syn', 'Amazon-Syn', 'BSC', 'Polygon'],
            'models': '7 primary models', 'configs_per_model': 12,
            'grid': {'DOMINANT, DLG-Base, DLG-Aug, AnomalyDAE, CoLA, GADNR': 'hid_dim {32,64,128} x lr {0.004,0.001} x epoch {50,100}',
                     'OCGNN': 'hid_dim {32,64,128} x lr {0.004,0.001} x beta {0.1,0.5}, epoch 50'},
            'selection': 'seed 42 validation AP; ties -> smaller hid_dim, then larger lr, then fewer epochs',
            'evaluation': 'selected configuration, seeds 42-46, test metrics read once',
            'label_use': 'semi-supervised model selection (validation labels); reported apart from the unsupervised primary table',
            'friedman_sensitivity': 'Friedman on the tuned table, and on the 3 Gram models only (linked to A29-07)',
        },
    },
    'A29-07_rank_statistics': {
        'input': 'frozen primary registry means (no new training)',
        'analyses': ['S2 without Elliptic and DGraphFin', 'epsilon-equivalence ties at 0.005 and 0.01 AP (values within epsilon of the row best... '
                     'implemented as rounding AP to the epsilon grid before ranking)', 'Nemenyi critical difference (alpha 0.05) for S1, S2, S4',
                     'Bayesian signed-rank with ROPE 0.01 AP for DLG-Aug vs DLG-Base, DLG-Aug vs DOMINANT, DLG-Base vs DOMINANT on S2 (50,000 Dirichlet draws, seed 20261123)'],
    },
    'A29-08_real_label_extension': {
        'candidates': {'Weibo': 'PyGOD/BOND weibo', 'Reddit-Real': 'PyGOD/BOND reddit (real banned-user labels; not Reddit-Syn)',
                       'Tolokers': 'PyG HeterophilousGraphDataset tolokers', 'YelpChi': 'DGL FraudYelp .mat homo relation',
                       'Amazon-Fraud': 'DGL FraudAmazon .mat homo relation; first 3,305 unlabeled nodes kept in the graph but excluded from evaluation'},
        'minimum': 3,
        'models': '7 primary models + IForest + ECOD (A24 feature-only settings)',
        'features': 'as distributed by the source, float32, no extra transform',
        'split': 'full observable label-free training; 50/50 stratified validation/test over evaluable nodes, sklearn train_test_split(random_state=seed)',
        'report': 'AP, test prevalence, ROC-AUC, P@1%, P@5%, lift; graph-model gain over the best feature-only model on the same splits; per-node scores stored internally',
        'statistics': 'descriptive only; no new omnibus view',
    },
    'A29-12_supervised_reference': {
        'graphs': ['Elliptic', 'DGraphFin', 'Ethereum', 'BSC', 'Polygon', 'LANL-RedTeam', 'A29-08 graphs'],
        'model': 'sklearn HistGradientBoostingClassifier defaults, random_state=seed',
        'inputs': ['X', '[X, mean of 1-hop neighbour X]'],
        'fit_population': 'labeled nodes disjoint from test: the train split when it exists and is disjoint, otherwise the validation half',
        'role': 'reference line showing whether labels are predictable from the inputs; never mixed into unsupervised ranks',
    },
}

text = yaml.safe_dump({'common': common, 'plans': plans}, sort_keys=False, allow_unicode=True, width=140)
(OUT / 'A29_PROTOCOL.yaml').write_text(text)
head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
dump(OUT / 'A29_PREREGISTRATION_BINDING.json', {
    'status': 'FROZEN_BEFORE_NEW_A29_TEST_METRICS',
    'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_head': head,
    'protocol_sha256': sha(OUT / 'A29_PROTOCOL.yaml'),
    'primary_registry_sha256': sha(ROOT / 'projects/benchmark/publication/a16/canonical_selected_run_registry.csv'),
})
print('A29 protocol frozen', sha(OUT / 'A29_PROTOCOL.yaml'))
