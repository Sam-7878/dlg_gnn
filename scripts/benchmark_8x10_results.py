import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 및 금융/사기 도메인 필터링
try:
    df = pd.read_csv('./dlg_gnn/docs/work_reports/35-dlg_full_pipeline_benchmark/benchmark_8x10_results.csv')
except FileNotFoundError:
    print("Error: benchmark_8x10_results.csv 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

target_datasets = ['Elliptic', 'DGraphFin', 'Yelp', 'Amazon', 'BitcoinOTC']
df_financial = df[df['Dataset'].isin(target_datasets)].copy()

# 2. 논문용 스타일 및 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10})

# 3. Figure 및 Subplots 설정 (1행 2열 구조)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Graph 1: 데이터셋별 ROC-AUC 성능 비교 ---
sns.barplot(
    data=df_financial, 
    x='Dataset', 
    y='ROC-AUC', 
    hue='Model', 
    ax=axes[0], 
    palette='muted'
)
axes[0].set_title('(a) ROC-AUC Comparison across Financial/Fraud Domains', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Target Dataset')
axes[0].set_ylabel('ROC-AUC Score')
axes[0].set_ylim(0, 1.1)
axes[0].legend().remove()  # 범례는 중간에 하나만 배치

# --- Graph 2: 금융/사기 도메인에서의 종합 복합 순위 (Lower is Better) ---
# 순위 계산 파이프라인
df_financial['ROC_Rank'] = df_financial.groupby('Dataset')['ROC-AUC'].rank(ascending=False)
df_financial['PR_Rank'] = df_financial.groupby('Dataset')['PR-AUC'].rank(ascending=False)
df_financial['F1_Rank'] = df_financial.groupby('Dataset')['F1-Score'].rank(ascending=False)
df_financial['Composite_Rank'] = (df_financial['ROC_Rank'] + df_financial['PR_Rank'] + df_financial['F1_Rank']) / 3

rank_summary = df_financial.groupby('Model')['Composite_Rank'].mean().reset_index().sort_values(by='Composite_Rank')

# 순위 시각화 (가로 바 차트)
colors = ['#1f77b4' if x == 'DLG' else '#aec7e8' if x == 'DLG-Base' else '#bcbd22' for x in rank_summary['Model']]
sns.barplot(
    data=rank_summary, 
    x='Composite_Rank', 
    y='Model', 
    ax=axes[1], 
    palette=colors if 'colors' in locals() else 'Blues_r'
)
axes[1].set_title('(b) Average Composite Rank in Financial Domains', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Average Rank (Lower is Better)')
axes[1].set_ylabel('Evaluation Model')

# 제안 모델(DLG) 강조선 추가
for p in axes[1].patches:
    if p.get_width() == rank_summary[rank_summary['Model'] == 'DLG']['Composite_Rank'].values[0]:
        p.set_alpha(1.0)
        p.set_edgecolor('black')
        p.set_linewidth(1.5)
    else:
        p.set_alpha(0.7)

# 4. 범례 통합 및 레이아웃 조정
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=4, frameon=True)
plt.tight_layout(rect=[0, 0, 1, 0.90])

# 5. 고해상도 이미지 저장 (논문 제출용 EPS 또는 PNG)
plt.savefig('./dlg_gnn/docs/work_reports/35-dlg_full_pipeline_benchmark/financial_domain_benchmark_results.png', dpi=300, bbox_inches='tight')
print("Graph successfully saved as 'financial_domain_benchmark_results.png' (300 DPI).")
plt.show()