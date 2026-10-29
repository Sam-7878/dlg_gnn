"""Editable numerical tables and an evidence-grounded reusable checklist."""
import json
import numpy as np
import pandas as pd
from common import OUT,SEEDS

NAMES={'gin_bce':'GIN + weighted BCE','sage_bce':'GraphSAGE + weighted BCE','summary_lr':'Summary LR','summary_mlp':'Summary MLP','gat_time':'GATv2 + absolute time','gat_no_time':'GATv2 without time','lightgbm':'LightGBM','isolation_forest':'Isolation Forest (unlabeled)','gin_focal':'GIN + weighted focal','sage_focal':'GraphSAGE + weighted focal','gin_no_chain':'GIN without chain','proposed':'Archival GIN','mc_dropout':'Archival MC dropout','temperature_scaled':'Archival temperature-scaled GIN','tgat':'Archival GATv2 + time','tgn':'Chain-level recurrent state','fraudsage':'Archival focal GraphSAGE'}


def esc(s):
    return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def table(name,caption,label,headers,rows):
    path=OUT/'tables'/name;path.parent.mkdir(exist_ok=True)
    t='\\begin{table}[H]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{adjustbox}{max width=\\textwidth}\n'
    t+='\\begin{tabular}{'+('l'+'r'*(len(headers)-1))+'}\\toprule\n'+' & '.join(headers)+r'\\\midrule'+'\n'
    t+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n\\bottomrule\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n'
    path.write_text(t)


def fmt(row,key):
    if not np.isfinite(row[key+'_mean']):return '---'
    return f"{row[key+'_mean']:.4f} $\\pm$ {row[key+'_std']:.4f}"


def main():
    d=pd.read_csv(OUT/'first_last_ordering_diagnostics.csv');rows=[]
    for order in ['last_valid_tx_time','first_valid_tx_time']:
        for split in ['train','validation','test']:
            r=d[(d.ordering==order)&(d.chain=='all')&(d.split==split)].iloc[0]
            rows.append(['Last transfer' if order.startswith('last') else 'First transfer',split,f'{r.n:,}',f'{r.positive:,}',f'{100*r.prevalence:.2f}'+r'\%'])
    table('table_sorting.tex','Partition composition under alternative sorting of the same 24,316 contracts.','tab:sorting',['Ordering','Partition','$N$','Positive','Prevalence'],rows)
    life=pd.read_csv(OUT/'lifespan_by_class.csv');rows=[]
    for r in life.itertuples():rows.append([r.chain,'Positive' if r.label else 'Other',f'{r.n:,}',f'{r.lifetime_median_days:.3f}',f'[{r.lifetime_q25_days:.3f}, {r.lifetime_q75_days:.3f}]'])
    table('table_lifetime.tex','Observed lifetime in days, by chain and class. IQR refers to descriptive archive spans.','tab:lifetime',['Chain','Category','$N$','Median','IQR'],rows)
    summary=pd.read_csv(OUT/'model_summary.csv');rows=[]
    for panel,arms in [('matched',['gin_bce','sage_bce','summary_lr','summary_mlp']),('additional',['gat_time','gat_no_time','lightgbm','isolation_forest'])]:
        for arm in arms:
            r=summary[(summary.panel==panel)&(summary.arm==arm)].iloc[0]
            rows.append([NAMES[arm],fmt(r,'auc_pr'),fmt(r,'auc_roc'),fmt(r,'ece'),fmt(r,'nll')])
    table('table_primary.tex','Original-cohort controlled reference fits (first four rows) and additional fixed-configuration controls; mean $\\pm$ sample SD over five seeds. All use the same held-out IDs. Probability losses are inapplicable to outlier scores. Repeated deterministic LR/LightGBM fits have zero seed spread, not zero sampling uncertainty.','tab:primary',['Model','AP','ROC-AUC','Raw ECE','Raw NLL'],rows)
    rows=[]
    for r in summary[summary.panel=='horizon7'].itertuples():
        r=pd.Series(r._asdict());rows.append([NAMES[r.arm],fmt(r,'auc_pr'),fmt(r,'auc_roc'),fmt(r,'ece'),fmt(r,'nll')])
    table('table_horizon.tex','Distinct first-plus-seven-day cohort: five-seed summaries. AP depends on this cohort\'s own prevalence; no cross-cohort paired inference is performed.','tab:horizon',['Model','AP','ROC-AUC','Raw ECE','Raw NLL'],rows)
    calibration=pd.read_csv(OUT/'modelwise_calibration.csv');rows=[]
    for panel,arms in [('matched',['gin_bce','sage_bce','summary_lr','summary_mlp']),('additional',['gat_time','gat_no_time','lightgbm'])]:
        for arm in arms:
            r=calibration[(calibration.panel==panel)&(calibration.arm==arm)]
            rows.append([NAMES[arm],f'{r.raw_ece.mean():.4f}',f'{r.ts_ece.mean():.4f}',f'{r.raw_nll.mean():.4f}',f'{r.ts_nll.mean():.4f}',f'{r.raw_brier.mean():.4f}',f'{r.ts_brier.mean():.4f}'])
    table('table_calibration.tex','Validation-only scalar-temperature calibration on the original cohort; five-seed mean losses. Raw and calibrated columns compare like stages across models.','tab:calibration',['Model','ECE raw','ECE TS','NLL raw','NLL TS','Brier raw','Brier TS'],rows)
    rows=[]
    for _,r in summary.iterrows():
        panel={'matched':'Reference','additional':'New controls','archival':'Archival','horizon7':'7-day','repeat_control':'LR repeat'}[r.panel]
        values=[f'{r[k+"_mean"]:.4f}' if np.isfinite(r[k+'_mean']) else '---' for k in ['auc_pr','auc_roc','f1','ece','adaptive_ece','brier','nll']]
        rows.append([panel,NAMES.get(r.arm,esc(r.arm)),*values])
    table('table_full_metrics.tex','Complete five-seed mean raw-score metrics by panel; full seed SDs are in the numerical CSV. F1 uses threshold 0.5. Outlier scores have no probability losses or F1 threshold.','tab:fullmetrics',['Panel','Model','AP','ROC-AUC','F1@0.5','ECE','Adaptive ECE','Brier','NLL'],rows)
    rows=[]
    for r in summary[summary.panel=='archival'].itertuples():
        r=pd.Series(r._asdict());rows.append([NAMES[r.arm],fmt(r,'auc_pr'),fmt(r,'ece'),fmt(r,'nll')])
    table('table_archival.tex','Separate archival schedule: retained results are not substituted for controlled-reference fits.','tab:archival',['Model','AP','ECE','NLL'],rows)
    chain=pd.read_csv(OUT/'chain_stratified_ap.csv');rows=[]
    for r in chain.itertuples():
        if (r.panel,r.arm) not in [('matched','gin_bce'),('matched','summary_mlp'),('additional','lightgbm')]:continue
        ci=f'[{r.ci95_low:.4f}, {r.ci95_high:.4f}]' if r.positive>=2 else 'Insufficient support'
        rows.append([NAMES[r.arm],r.chain,r.n,r.positive,f'{r.ap_mean:.4f}',ci])
    table('table_chain.tex','Per-chain AP on the original test. Intervals use 2,000 event draws conditional on fixed checkpoint pairs; Polygon point estimates do not support useful comparative inference.','tab:chain',['Model','Chain','$N$','Positive','Mean AP',r'Conditional 95\% interval'],rows)
    ops=pd.read_csv(OUT/'operational_metrics.csv');rows=[]
    for (panel,arm,k),g in ops.groupby(['panel','arm','budget']):
        if panel not in ['matched','additional','control'] or arm not in ['gin_bce','summary_mlp','lightgbm','chain_prior']:continue
        rows.append([NAMES.get(arm,'Chain prior'),k,f'{g.captured.mean():.2f}',f'{g.precision.mean():.4f}',f'{g.recall.mean():.4f}',f'{g.tie_expected_captured.mean():.2f}',f'[{g.tie_min_captured.mean():.2f}, {g.tie_max_captured.mean():.2f}]'])
    table('table_operating.tex','Original-test investigation budgets: mean fixed-order captures and tie-sensitive alternatives. Counts averaged across five seeds can be non-integer; the chain prior is one fitted control.','tab:operating',['Model','Budget','Captured','Precision','Recall','Tie expectation','Tie capture bounds'],rows)
    stats=pd.read_csv(OUT/'statistics_and_holm.csv');rows=[]
    for r in stats.itertuples():
        name=r.comparison.replace('TGN','chain recurrent').replace('TGAT','time GATv2')
        rows.append([esc(name),f'{r.mean_auc_pr_difference:.4f}',f'[{r.ordinary_ci95_low:.4f}, {r.ordinary_ci95_high:.4f}]',f'{r.randomization_p_value_two_sided:.6f}',f'{r.holm_adjusted_p:.6f}'])
    table('table_statistics.tex','Preserved comparisons: first seven comprise the archival family, following six the separate exploratory family. Holm adjustment is never pooled across the two.','tab:statistics',['Comparison','Mean AP difference',r'Ordinary 95\% interval','Raw swap $p$','Holm $p$'],rows)
    checklist=[
      ['Sample and label time','Source labels; 24,316 one-to-one joins','Contract unit; discovery time unavailable','No prospective metric measured','State retrospective supervision explicitly'],
      ['Archive boundary and lifetime','268,282,924 rows; first/last times','Month documented; exact stopping rule unknown',r'Last-order test 2.93\%; first-order 56.99\%','Diagnose sorting before attributing drift'],
      ['Cutoff and ties','Builder; strict-time tensor reconstruction','3,910 observations have tied cutoffs','Strict checkpoint AP deltas supplied separately','Declare inclusive cutoff and deterministic tie rule'],
      ['Feature admissibility','Full-history degrees and tail128 audit','All new horizon features through cutoff','0 maximum-input cutoff violations','Recompute features as well as edges'],
      ['Chain/entity dependence','Partition keys and cross-chain strings','0 chain:contract intersections; wallets unknown','Block interval supplied; actor effect unmeasured','Report unresolved actor dependence'],
      ['Simple controls','Identical 24 summary inputs','MLP remains competitive','MLP AP 0.4751; GIN+BCE 0.4524','Give tabular models the same admitted information'],
      ['Loss comparability','Matched BCE/focal matrix','Objective changes ranking and losses','GIN focal AP 0.4885; ECE 0.3179','Name actual weighting and match objectives'],
      ['Time extrapolation','Train-only extrema; 5-seed retraining','All original test normalized times exceed1','With/without-time AP in Results','Test implementation-specific time branches'],
      ['Calibration','Preserved validation outputs','Scalar temperature fitted without test labels','Raw/TS ECE, NLL and Brier provided','Compare like calibration stages'],
      ['AP ties','Threshold-grouped AP; sklearn parity','Bootstrap duplicates require tie grouping','Original and corrected audit retained','Test ties and duplicated draws'],
      ['Multiplicity and dependence','Fixed7 and fixed6 families; hour blocks','Conditional inference; no cross-cohort pairing','Exploratory GIN--MLP Holm p=0.1184','Freeze family and resampling unit'],
      ['Prediction lineage','55 old checkpoints; 65 original arrays','Old hashes sealed; new cohorts separated','Numeric replay of archived and new scores','Bind predictions to input/split/checkpoint hashes']]
    lines=[r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{p{.13\textwidth}p{.18\textwidth}p{.19\textwidth}p{.20\textwidth}p{.20\textwidth}}',r'\caption{Reusable validity-first checklist. Quantified consequences refer only to analyses actually performed.}\label{tab:checklist}\\',r'\toprule Audit requirement & Evidence source & Our finding & Tested consequence & Reuse recommendation\\\midrule\endfirsthead',r'\toprule Audit requirement & Evidence source & Our finding & Tested consequence & Reuse recommendation\\\midrule\endhead']
    for row in checklist:lines.append(' & '.join(row)+r'\\\addlinespace')
    lines += [r'\bottomrule\end{longtable}\endgroup']
    (OUT/'reusable_validity_checklist.tex').write_text('\n'.join(lines)+'\n')
    (OUT/'tables/reusable_validity_checklist.tex').write_text('\n'.join(lines)+'\n')
    pd.DataFrame(checklist,columns=['audit_requirement','evidence_source','finding','tested_consequence','reuse_recommendation']).to_csv(OUT/'reusable_validity_checklist.csv',index=False)


if __name__=='__main__':main()
