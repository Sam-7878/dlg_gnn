from a24_common import *
import numpy as np,ast,importlib.metadata as md,sys,os
sys.path.insert(0,str(LOCAL/'deps'))
# Freeze exact installed supplementary-only dependencies; common .venv_cuda untouched.
packages={p:md.version(p) for p in ['torch','torch-geometric','pygod','numpy','scipy','scikit-learn','psutil','pyod','numba','llvmlite']}
dump(SUP/'environment_a24.json',{'packages':packages,'python':sys.version,'CUDA_VISIBLE_DEVICES':'GPU-ab53069a-3217-ee09-80f8-748b5fdbd156','local_dependencies':'local_storage/benchmark/a24/deps','protocols_sha256':{p.name:sha(p) for p in SUP.glob('*plan.yaml')},'sources':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'projects/benchmark/scripts').glob('a24_*.py') if 'build_manuscript' not in p.name}})
summary=[]
for filename,group in [('feature_only_per_seed.csv','feature_only'),('elliptic_ablation_25_runs.csv','elliptic_controls'),('real_domain_alert_budget_per_seed.csv','real_domain_alert')]:
 if not (SUP/filename).exists():continue
 rr=rows(SUP/filename)
 for dataset,model in dict.fromkeys((r['dataset'],r['model']) for r in rr):
  selected=[r for r in rr if r['dataset']==dataset and r['model']==model]
  for metric in ['pr_auc','roc_auc','test_f1','max_validation_f1','P_at_1pct','P_at_5pct','R_at_1pct','R_at_5pct','lift_1pct','lift_5pct']:
   a=np.array([float(r[metric]) for r in selected]);summary.append({'study':group,'dataset':dataset,'model':model,'metric':metric,'n':len(a),'mean':float(a.mean()),'sample_SD':float(a.std(ddof=1)) if len(a)>1 else 0.,'ddof':1})
writecsv(SUP/'supplementary_summary.csv',summary)
# Current crypto lifts from individual raw seed budgets, never ratio-of-means.
rr=rows(REG);lift=[]
for d in ['Ethereum','BSC','Polygon']:
 for model in sorted({r['model'] for r in rr if r['dataset']==d}):
  for r in [r for r in rr if r['dataset']==d and r['model']==model]:
   v=ast.literal_eval(r['alert_budgets']);p=float(r['test_prevalence'])
   for q,b in v.items():lift.append({'dataset':d,'model':model,'seed':r['seed'],'budget':q,'P':b['precision'],'R':b['recall'],'K':b['k'],'test_N':int(float(r['test_total'])),'test_positive':int(float(r['test_positive'])),'test_prevalence':p,'lift':b['precision']/p,'score_sha256':r['raw_score_hash']})
writecsv(PUB/'table7_lift_and_support_per_seed.csv',lift)
agg=[]
for d,m,q in dict.fromkeys((r['dataset'],r['model'],r['budget']) for r in lift):
 selected=[r for r in lift if (r['dataset'],r['model'],r['budget'])==(d,m,q)];a=np.array([r['lift'] for r in selected]);agg.append({'dataset':d,'model':m,'budget':q,'mean_lift':float(a.mean()),'sample_SD_lift':float(a.std(ddof=1)),'mean_P':float(np.mean([r['P'] for r in selected])),'mean_prevalence':float(np.mean([r['test_prevalence'] for r in selected])),'ratio_of_means_not_used':float(np.mean([r['P'] for r in selected])/np.mean([r['test_prevalence'] for r in selected])),'n_seeds':5,'ddof':1,'interpretation':'descriptive seedwise lift;Polygon1%K5 highly discrete'})
writecsv(PUB/'table7_lift_and_support_a24.csv',agg)
# Split role is explicit: training sees observables transductively, evaluation masks select labels.
old=rows(ROOT/'projects/benchmark/publication/a23/dataset_graph_identity_canonical.csv');split=[]
for r in old:
 n=r['dataset_id'];kind='60/20/20 RandomState unstratified' if n in ['Ethereum','BSC','Polygon'] else 'official masks' if n=='DGraphFin' else '60/20/20 classwise permutation' if n=='LANL-RedTeam' else 'full observable training;50/50 stratified validation/test'
 split.append({'dataset':n,'prediction_unit':('smart contract node' if n in ['Ethereum','BSC','Polygon'] else r['label_source_and_prediction_unit'].split(' / ')[-1].split(' (')[0]),'training_observables':'full model-input X/E;no anomaly labels','evaluation_split':kind,'test_N_by_seed':r['test_population_N_by_seed'],'positive_test_by_seed':r['positives_by_seed'],'threshold_policy':'validation distinct scores in A08/A24;original historical policy retained in primary','raw_score_reproducibility':'A08crypto retained;historical source tier preserved;newA24scores separate'})
writecsv(PUB/'split_policy_by_dataset_a24.csv',split)
(SUP/'gadnr_24gb_jitter_diagnostic.md').write_text('# A24 optional G9\n\nNOT_ATTEMPTED_OPTIONAL. The declared feature-only, canonical controls, resource, Cora equivalence and alert-budget studies were prioritized. No24GB Flickr/GADNR or jitter recovery is claimed, and no historical unsupported cell is changed. A23 mathematical/covariance diagnostics remain historical evidence, not an observed failure trace for missing runs.\n')
(SUP/'feature_only_comparison.md').write_text('# Feature-only supplementary comparison\n\n60evaluations:2models/6datasets/5existing splits. Frozen X hashes and split hashes recorded. All methods fit the full observable feature population transductively without targets, matching the GNN observable scope;no extra preprocessing fitted. Test labels are used only after scores. ECOD is deterministic;five split scores are not five independent fits, and DGraphFin masks do not vary across seeds. LANL ECOD mean AP exceeds the historical GNN references;Polygon IForest exceeds DLG-Aug but not AnomalyDAE. These are separate supplementary comparisons,not new primary ranks or proof that graph structure is always helpful/useless.\n')
(SUP/'elliptic_control_provenance.md').write_text('# Matched canonical Elliptic controls\n\n25fresh runs reuse the exact canonical X/E/y and seed42–46 stratified50/50val/test masks. Both aligned and permuted vectors use the same pretrained features with a fixed row permutation. Zero uses the same20local epochs and then zeros those features,retaining consumed RNG and compute;this differs from the old A03zero control. Base70 controls global epoch budget. Results preserve the higher Permuted than aligned mean;they do not identify a capacity/regularization mechanism. Raw vectors and validation-max F1 vs test F1 are separate and retained locally.\n')
print('A24 scalar summaries,seed-wise lifts and14-dataset split policy generated',flush=True)
