"""Immutable post-lineage, pre-fitting protocol; no outcome-driven search."""
import json
from common import OUT,SEEDS,sha,write


def register():
    decision=json.loads((OUT/'selection_decision.json').read_text())
    assert decision['full_lineage_reconstructed']
    p=dict(id='TDS317-P1',selection_decision_sha256=sha(OUT/'selection_decision.json'),
        seeds=SEEDS,device='cuda:1 (RTX3090; independent of laptop GPU)',
        archival_time_ablation=['gat_time','gat_no_time'],
        finite_horizon_days=7,finite_horizon_arms=['summary_lr','summary_mlp','gin_bce','gat_time'],
        finite_horizon_condition='SELECTION_EFFECT_SUPPORTED',
        finite_horizon_eligibility='first+7d <= observed archive maximum; NOT last>=first+7d',
        finite_horizon_boundary='empirical support endpoint, exact collector stopping rule unknown',
        finite_horizon_split='stable sort(first+7d, original event key), floor(70%n), floor(85%n), remainder',
        finite_horizon_graph='all degree features recomputed from valid transfers <=cutoff; stable last128 topology',
        label_policy='same retrospective Category0 label; discovery time unknown; not early warning',
        neural=dict(hidden=48,dropout=.3,batch_size=64,optimizer='AdamW',lr=.001,weight_decay=.0001,
                    max_epochs=60,patience=5,selection='validation AP increase>1e-6',loss='BCE weighted negatives/positives from train'),
        time_ablation='same GATv2 encoder; no-time head omits harmonic16; both trained from scratch',
        lr=dict(C=1,max_iter=2000,class_weight='balanced'),
        lightgbm=dict(n_estimators=300,learning_rate=.05,num_leaves=31,max_depth=-1,class_weight='balanced',n_jobs=4,validation_early_stop=30),
        isolation_forest=dict(n_estimators=200,contamination='auto',max_samples='auto',score='negative score_samples; higher=more anomalous; not label-selected',labels_used=False),
        scaler='same24 summary features; StandardScaler fit train only',
        calibration='temperature per model/seed via validation NLL; logT bounded[-4,4]; test once',
        uncertainty=dict(per_chain_bootstrap=2000,time_block_bootstrap=2000,time_block_seconds=3600,seed=317),
        families='old7 confirmatory and old6 exploratory unchanged; new317 descriptive, no new significance claims',
        stopping='fixed arm grid only; no test-based choice or retries for higher AP')
    dest=OUT/'experiment_preregistration.json'
    if dest.exists():assert json.loads(dest.read_text())==json.loads(json.dumps(p))
    else:write(dest,p)
    return p


if __name__=='__main__':print(register()['id'])
