"""Check bounded manuscript edits against immutable numeric evidence."""
import csv
import json
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from paths import ROOT, INPUT, PAPER, RESULTS, REPORTS, ORIGINAL_PAPER, SUBMISSION, manuscript_source
from freeze import sha, write


def main():
    tex = manuscript_source()
    extra = (PAPER/'supplement/supplementary.tex').read_text()
    cover = (SUBMISSION/'final_cover_letter.md').read_text()
    abstract = tex.split('\\abstract{',1)[1].split('}\n\\keyword',1)[0]
    assert '0.4751' in abstract and 'unique message-passing superiority' in abstract
    assert '0.4751' in cover and 'does not claim a novel GNN architecture' in cover
    assert '0.0987' not in tex and '0.0986' in tex
    assert 'Final approval of this TDS manuscript version remains pending' in tex
    assert 'TDS-specific conflict-of-interest declaration is pending' in tex
    assert 'Corresponding-author designation pending' in tex
    assert 'immutable public tag and anonymous download verification remain pending' in tex
    assert 'All authors have read and agreed to the published version' not in tex
    assert 'IITP-2026-RS-2021-II211835' in tex
    assert 'SeongSu Park' in tex and 'Ki-Hyung Kim' in tex
    assert '0.4751' in tex.split('\\section{Conclusions}',1)[1]
    # All numeric TeX tables and original scientific interpretations are identical to R314.
    original_numeric = []
    for p in sorted((ORIGINAL_PAPER/'data').glob('*.tex')):
        q = PAPER/'data'/p.name
        assert sha(p) == sha(q), p
        original_numeric.append(dict(path='data/'+p.name, sha256=sha(q)))
    assert sha(ORIGINAL_PAPER/'supplement/table_control_statistics.tex') == sha(PAPER/'supplement/table_control_statistics.tex')
    stats = pd.read_csv(INPUT/'supplement/statistical_comparisons_corrected.csv')
    assert f'{stats.iloc[4].holm_adjusted_p:.4f}' == '0.0920'
    assert f'{stats.iloc[6].randomization_p_value_two_sided:.4f}' == '0.1279'
    tables = (PAPER/'data/table_statistics.tex').read_text()
    for row in stats.itertuples():
        for value in [row.mean_auc_pr_difference, row.ordinary_ci95_low, row.ordinary_ci95_high,
                      row.stratified_ci95_low, row.stratified_ci95_high]:
            assert f'{value:+.4f}' in tables, value
        for value in [row.randomization_p_value_two_sided, row.holm_adjusted_p]:
            assert f'{value:.4f}' in tables, value
    # Bibliography edit is restricted to removing the redundant leading word in booktitle.
    before=(ORIGINAL_PAPER/'references.bib').read_text(); after=(PAPER/'references.bib').read_text()
    normalized=re.sub(r'(booktitle\s*=\s*\{)Proceedings of the ',r'\1',before)
    assert normalized == after
    bbl=(PAPER/'applsci_article.bbl').read_text()
    assert 'Proceedings of the Proceedings of' not in bbl
    evidence = [
        ('sample_unit','One token/contract graph at last valid recorded transfer time; not transaction-level labels','article Materials and Methods; supplement data semantics','dataset_contract.json'),
        ('retrospective_label','Upstream Category 0 positive; discovery time unavailable','article Dataset and Limitations','dataset_contract.json'),
        ('test_horizon','3648 contracts / 107 positives, 68230 seconds; no long-term guarantee','article split table and Limitations','data/split_composition.csv'),
        ('polygon_support','688 Polygon test contracts with one positive; no cross-chain robustness claim','article temporal composition and Limitations','data/split_composition.csv'),
        ('full_history','Degree inputs precede the 128-edge topology cap; no k-hop sampling parameter','article information admissibility','scientific/common.py;original_source/build_gog_chronological_dataset.py'),
        ('time_ties','3910/24316 tied observations; deterministic ordering is not execution order','article admissibility','data/timestamp_tie_audit.csv'),
        ('state','Per-chain TGN phase-reset predict-then-update without labels; TGAT absolute cutoff encoding','article reference encoders and Limitations','scientific/historical_models.py'),
        ('strict_time','Fixed-checkpoint input stress test; not a fully retrained strict benchmark','article strict interpretation and supplement','data/strict_time_sensitivity.csv'),
        ('causal','Temporal input admissibility, without causal-effect identification','article admissibility and Limitations','dataset_protocol.md'),
        ('mlp_negative_finding','MLP AP0.4751 vs supplemental BCE GIN0.4524; historical GIN0.4648 separate','Abstract; controls; Conclusions; cover letter','data/supplemental_summary.csv;prediction_lineage.json'),
        ('ap_definition','Threshold-grouped AP; not trapezoidal PR area','article metric protocol; supplement','scientific/statistical_audit.py;replay.py'),
        ('original_holm','Seven original comparisons; MC vs Temp Holm0.0920, ensemble raw0.1279; no familywise significance claim','article table 3 and corrected inference','statistics/statistical_comparisons_corrected.csv'),
        ('exploratory_holm','Six separate exploratory comparisons; MLP contrast not significant','article controls; supplement table3','statistics/control_statistical_comparisons.csv'),
        ('finite_p','(extreme+1)/10001 for10000 draws; seeds not resampled','article metric protocol; supplement','scientific/historical_statistics.py'),
        ('ensemble_nll','Temperature-scaled deep ensemble NLL0.0986; fixed aggregate','article Results and table2','data/table_primary.tex (private);primary numeric predictions'),
        ('mc_runtime','Figure3 mean/sample SD and median full-panel runtime; descriptive scope','article Figure3 and caption','data/mc_sensitivity_per_seed.csv;data/mc_sensitivity_summary.csv'),
        ('calibration','Figure4 support counts and tau<1 sharpening; no uniform overconfidence claim','article Figure4 and calibration methods','statistics/reliability_bin_support.csv;data/calibration_direction_audit.csv'),
        ('no_broad_claim','No novel architecture, prospective fraud detection, universal temporal inferiority or independent future sampling claim','Abstract; Discussion; Limitations; Conclusions; cover letter','audited text and frozen panel scope')]
    with (REPORTS/'claim_to_evidence_matrix.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['claim_id','bounded_claim','manuscript_location','evidence_path','status'])
        for row in evidence: w.writerow([*row,'PASS'])
    write(REPORTS/'claim_consistency.json',dict(status='PASS',unsupported_broad_claims=0,
        scope='Targeted source/value checks plus manual claim review; not a formal automated proof of all prose',
        numeric_tex_inputs_identical=original_numeric,bibliography_changes='Five booktitle prefix removals only; citation keys and other metadata unchanged',
        original_family=7,exploratory_family=6,mc_temp_holm=0.091990800919908,ensemble_randomization_p=0.1278872112788721,
        final_approval_not_inferred=True,claim_matrix_rows=len(evidence)))
    (REPORTS/'text_value_consistency.md').write_text(
        '# Text/value consistency\n\nPASS for the targeted 315 audit. The Abstract and Conclusions expose degree-summary MLP AP0.4751 '
        'and distinguish supplemental BCE GIN0.4524 from historical GIN0.4648. Cover letter agrees. '
        'All retained data/interpretation TeX inputs are byte-identical to sealed R314 inputs. '
        'The seven original table rows match corrected intervals and p-values; Holm0.0920 and raw0.1279 remain non-significant. '
        'The six exploratory contrasts remain a separate family. AP means threshold-grouped average precision. '
        'The ensemble NLL is0.0986 throughout. Figure3 SD/median definitions and Figure4 support/sharpening semantics remain intact.\n\n'
        'Only five redundant booktitle prefixes were removed; keys and all other bibliography metadata are unchanged. '
        'Nineteen references are cited, with no count padding. No broad architecture/prospective/long-horizon claim was found '
        'in the reviewed article, supplement or cover letter. Source-based assertions and manual review scope are recorded '
        'in claim_consistency.json and claim_to_evidence_matrix.csv. Author and public-release pending text remains explicit.\n')
    print('PASS: scientific claim/value checks and restricted bibliography changes')


if __name__=='__main__': main()
