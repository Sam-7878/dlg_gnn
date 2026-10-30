"""Freeze hypotheses and baseline evidence before any new diagnostic evaluation."""
import json
import subprocess
from common import ROOT, PROJECT, OUT, PRIVATE, REPORTS, ORIGINAL, HORIZON, SEEDS, sha, write


def main():
    for p in [OUT, PRIVATE, REPORTS]:
        p.mkdir(parents=True, exist_ok=True)
    seal = OUT / 'baseline_freeze_319.json'
    if not seal.exists():
        roots = ['results/main_final_v2','results/314_applsci','results/315_applsci','results/316_dual_submission',
                 'results/317','results/318','reports/317','reports/318','paper/submission_318',
                 'submission/preprints/318','submission/applsci/318','submission/author_records/317',
                 'submission/author_records/318','scripts/revision314','scripts/revision317','scripts/revision318']
        files=[]
        for name in roots:
            for p in sorted((PROJECT / name).rglob('*')):
                if not p.is_file() or '__pycache__' in p.parts or p.suffix in {'.aux','.log','.out','.bbl','.blg','.fls','.fdb_latexmk','.pyc'}:
                    continue
                files.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size))
        for p in [ORIGINAL,HORIZON,PROJECT / 'publication/317/TDS_317_numeric_release_candidate.zip']:
            files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
        write(seal,dict(files=files,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        modified_prior_artifacts=False,scope='R314–318 outputs, selected preserved source/manuscript trees and graph inputs'))
    p=dict(id='TDS319-P1',date='2026-10-30',freeze_sha256=sha(seal),
        cohorts={c:dict(population_sha256=sha(PROJECT / f'results/317/{n}_population.csv'),
                       graph_sha256=sha(ORIGINAL if c=='original' else HORIZON))
                 for c,n in [('original','original'),('horizon7','horizon7')]},
        seeds=list(SEEDS),builder=dict(scope='all 24316 original and all 24179 seven-day graphs from raw source ZIPs',
            ordering=['timestamp','block_number','transaction_hash','original_row_index'],cap=128,
            equivalence=['x float32','edge_index int64 including ordering/multiplicity','label','chain encoding','cutoff'],
            training_gate='any mismatch stops fitting; preserve mismatch evidence before correction'),
        feature_schema=dict(full24='n,e,density; mean/median/max of retained in/out/total degrees; mean/median/max of 3 full-history x columns; chain one-hot',
            count_only=['log1p(retained_nodes)','log1p(retained_edges)','log1p(full_history_nodes_at_cutoff)','log1p(full_history_edges_at_cutoff)'],
            single_feature_method='train-only balanced logistic regression, standardized on training only; one deterministic fit per configuration',
            degree_only_indices=list(range(3,21)),mlp_no_size_indices=list(range(3,24)),
            mlp_topology_only_indices=list(range(12))+[21,22,23],
            note='Removing explicit counts/density does not eliminate implicit size information from degree summaries'),
        oracle=dict(score='negative log1p(full archive observed span in days); fixed retrospective orientation',
            use='ORACLE/RETROSPECTIVE DIAGNOSTIC ONLY; not a model input or performance upper bound',
            subgroups=['L<1d','1<=L<7d','L>=7d'],continued='last>=first+7d; future-conditioned descriptive subset, never eligibility'),
        arms=['gin0','gin_size','gin_sum','mlp_full','mlp_no_size','mlp_topology_only'],
        neural=dict(hidden=48,dropout=.3,batch_size=64,optimizer='AdamW',lr=.001,weight_decay=.0001,
            epochs=60,patience=5,min_validation_ap_improvement=1e-6,loss='BCEWithLogits; train negatives/positives pos_weight',
            selection='validation AP only',test_access='one prediction pass after selecting checkpoint',
            gin0='preserved 2-layer GIN mean+max readout + chain',
            gin_size='same encoder; head includes train-standardized log1p(retained nodes), log1p(retained edges)',
            gin_sum='same encoder; mean+max+sum readout + chain; larger head recorded',
            histories=['train loss','train AP','train ROC','validation loss','validation AP','validation ROC'],
            capacity='parameter counts differ explicitly; no equal-capacity claim',extra_training_metrics='fork RNG for extra monitoring to avoid consuming shuffle/dropout stream'),
        lightgbm=dict(grid=[dict(num_leaves=n,learning_rate=r) for n in [15,31,63] for r in [.03,.1]],
            n_estimators=300,early_stopping=30,class_weight='balanced',n_jobs=4,
            selection='maximum validation AP; fixed grid-order tie break; test evaluated for selected model only',
            implementation='existing project-local LightGBM4.6.0'),
        oneclass=dict(enabled=True,arm='negative_only_iforest',fit='training negatives only',n_estimators=200,max_samples=256,
            score='negative score_samples, fixed orientation; no calibration/probability claim'),
        chain_prior='training chain positive fractions only; unseen chain uses pooled training prior',
        calibration=dict(models='all seven-day models with saved validation probabilities, including preserved R317 arms',
            fit='validation-only NLL; logT bounds [-4,4]; ranking uses unclipped scores'),
        operating=dict(budgets=['ceil(1%n)','ceil(5%n)',50,100],tie='stable frozen sample order; expectation and attainable bounds'),
        subgroups=dict(min_positive=20,min_negative=20,bootstrap_draws=2000,
            inadequate='both classes required for point AP/ROC; below CI threshold report support and point only',
            unit='contract observations; fixed checkpoints; within-stratum ordinary bootstrap; unknown actor dependence'),
        composition=dict(UTC_periods=['month','quarter'],groups=['all','ethereum','bsc','polygon'],interval='Wilson95%',
                         variables=['n','positive','prevalence','observed span median/IQR','time to endpoint','7/30d eligibility']),
        statistics=dict(family='R319 new four-comparison exploratory family, disjoint from preserved R317 7+6 families',
            comparisons=[['gin_size','gin0'],['gin_sum','gin0'],['mlp_full','mlp_no_size'],['mlp_full','mlp_topology_only']],
            paired_unit='same seven-day test contracts and same five seeds; mean per-seed AP difference',
            ordinary_bootstrap=2000,UTC_day_block_bootstrap=2000,paired_score_swaps=4000,holm_family_size=4,rng_seed=319,
            no_cross_cohort_paired_tests=True),
        preprocessing='StandardScaler training rows only; inherited splits/cutoffs/labels unchanged',
        stopping='fixed diagnostic grid; no test-driven retry, reorientation, threshold or best-seed selection',
        public_release='HOLD: no push/tag/DOI/portal action without exact-file approval and rights evidence')
    dest=OUT / 'protocol_freeze_319.json'
    if dest.exists():
        assert json.loads(dest.read_text())==p,'Frozen protocol cannot be overwritten'
    else:write(dest,p)
    print('R319 protocol frozen',sha(dest),flush=True)


if __name__=='__main__':main()
