"""Read-only historical audit; write only the new R3 evidence namespace."""
from pathlib import Path
import json
import pandas as pd
import torch
from gog_fraud.production.submission_r3 import read_config,save_json
from evidence.experiment_identity import file_digest


def main():
    cfg=read_config(); root=Path(cfg['output_root']); dest=root/'streaming'; dest.mkdir(parents=True,exist_ok=True)
    reports=Path('docs/work_reports/sci_v3_submission_r3'); reports.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(cfg['raw_events']); cache=torch.load(cfg['graph_cache'],weights_only=False,map_location='cpu')
    counts=raw.groupby(['chain_id','contract_id']).agg(events=('sample_id','size'),label=('label','first'),label_variants=('label','nunique')).reset_index()
    assert counts.label_variants.max()==1
    counts.to_csv(dest/'replay_contract_event_counts.csv',index=False)
    train=pd.DataFrame(cache['metadata']['train']); test=pd.DataFrame(cache['metadata']['test'])
    summary={'case':'S-B','prediction_unit':'event_with_inherited_contract_label', 'events':len(raw),
        'unique_contracts':len(counts),'positive_events':int(raw.label.sum()),'positive_contracts':int(counts.label.sum()),
        'event_weighted_prevalence':float(raw.label.mean()),'unique_contract_prevalence':float(counts.label.mean()),
        'max_events_per_contract':int(counts.events.max()),'top10_event_fraction':float(counts.events.nlargest(10).sum()/len(raw)),
        'labels_consistent_within_contract':True,'prediction_frequency':'every event',
        'offline_prediction_unit':'contract','directly_comparable_to_offline_f1':False,
        'raw_event_start':int(raw.event_time.min()),'raw_event_end':int(raw.event_time.max()),
        'training_reference_latest_event_end':int(train.event_end.max()),
        'events_before_training_reference_freeze':int((raw.event_time<train.event_end.max()).sum()),
        'test_contracts_before_global_training_reference_freeze':int((test.event_end<train.event_end.max()).sum()),
        'source_sha256':file_digest(cfg['raw_events']),
        'live_deployment_claim_eligible':bool((raw.event_time>=train.event_end.max()).all()),
        'available_original_dataset':Path(cfg['base']['dataset_root']).exists(),
        'graph_cache_keys':list(cache['graphs']['train'][0].keys()),
        'raw_source_note':'frozen chronological prefix sampled from per-contract transaction files; not complete chain traffic'}
    save_json(dest/'prediction_unit_summary.json',summary)
    (reports/'streaming_prediction_unit_audit.md').write_text('# Streaming prediction-unit audit\n\n'+
        '\n'.join(f'- {k}: {v}' for k,v in summary.items())+'\n\nEach row is a transaction event. The raw-event generator inherits the contract label from its metadata; the same contract is scored repeatedly. '
        'The main offline benchmark has one bounded snapshot per contract. Replay confusion matrices are event-weighted diagnostics, not independent contract detection estimates. '
        'A last-observed-event-per-contract diagnostic may be reported separately, but its prefix and graph history still differ from the offline benchmark. '
        'A frozen model replay before its training-reference cutoff is a retrospective systems workload, not a temporal online accuracy evaluation.\n')
    (reports/'temporal_baseline_exclusion_report.md').write_text('# Temporal baseline: formal exclusion\n\n'
        f'Original dataset available: {summary["available_original_dataset"]}. Graph-cache fields: {summary["graph_cache_keys"]}. '
        'The original dataset symlink targets a missing Ubuntu home directory. The bounded cache preserves degree features, graph edges, contract labels and partition metadata, but not edge timestamps or the complete train/validation transaction histories. '
        'The frozen 100k transaction prefix is a sampled replay workload, not the complete matched temporal training population. '
        'Training TGN/TGAT on that prefix would change temporal coverage, prediction units or test support. No artificial baseline or fabricated timing is reported. '
        'Restoring the original chronological transaction files and split manifests is required for a fair baseline; this is a limitation, not evidence that a temporal model is inferior.\n')
    (reports/'predictive_metric_identity_audit.md').write_text('# Predictive identity and historical runtime audit\n\n'
        'The production F1 near 0.621 comes from five supervised frozen GIN/GATv2 models, contract-level held-out predictions, validation Platt maps, calibrated-margin selection, log-odds fusion and a validation-selected final threshold. '
        'The benchmark row near 0.336 belongs to the separately exported graph anomaly/benchmark interface. It is not a second estimate for the calibrated production configuration and is retained as a legacy benchmark diagnostic only. '
        'Inspect the immutable R2 manuscript/table_graph_anomaly_baselines and cascade predictions for the exact interfaces and per-row source identities. No shared production identity is assigned to these numbers.\n\n'
        'Additional code-level mismatch: R2 profile_submission_r2_policy_repeats passed calibrated thresholds/margins into raw_event_selective_e2e_profiler.replay, '
        'which routed raw scores and used the old fixed fusion weights. R2 runtime values therefore do not validate execution of the calibrated primary policy. '
        'R2 offline relation_data also joined all target graphs, whereas raw replay deep_graph used recent context. R3 uses independent target stars with frozen training neighbors in both paths and recalibrates validation on that interface. '
        'This inference-interface correction invalidates direct transfer of R2 calibrated accuracy and timing claims. R2 files remain unchanged.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
