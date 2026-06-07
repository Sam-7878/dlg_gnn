import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 및 금융/사기 도메인 필터링
try:
    df = pd.read_csv('./dlg_gnn/docs/work_reports/35-dlg_full_pipeline_benchmark/benchmark_8x10_results.csv')
except FileNotFoundError:
    print("Error: benchmark_8x10_results.csv 파일을 찾을 수 없습니다.")
    exit()

target_datasets = ['Elliptic', 'DGraphFin', 'Yelp', 'Amazon', 'BitcoinOTC']
df_financial = df[df['Dataset'].isin(target_datasets)].copy()

# 2. 모든 그래프에서 모델 색상을 통일하기 위한 고유 고정 팔레트 정의
models_order = ['DOMINANT', 'AnomalyDAE', 'CoLA', 'CONAD', 'GADNR', 'OCGNN', 'DLG-Base', 'DLG']
# 저널용 세련된 팔레트(Deep/Muted 계열)에서 8가지 색상 추출
base_colors = sns.color_palette("deep", len(models_order))
model_colors = dict(zip(models_order, base_colors))

# 3. 논문 표준 스타일 및 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10})

# 4. Figure 생성 (여백 확보를 위해 세로 높이를 6.5로 약간 확장)
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# --- Graph 1: 데이터셋별 ROC-AUC 성능 비교 ---
sns.barplot(
    data=df_financial, 
    x='Dataset', 
    y='ROC-AUC', 
    hue='Model', 
    hue_order=models_order,
    ax=axes[0], 
    palette=model_colors
)
axes[0].set_title('(a) ROC-AUC Comparison across Financial/Fraud Domains', fontsize=13, fontweight='bold', pad=15)
axes[0].set_xlabel('Target Dataset', labelpad=10)
axes[0].set_ylabel('ROC-AUC Score', labelpad=10)
axes[0].set_ylim(0, 1.1)
axes[0].legend().remove()  # 개별 범례 삭제 (상단 통합 범례 사용)

# --- Graph 2: 금융/사기 도메인에서의 종합 복합 순위 계산 및 시각화 ---
df_financial['ROC_Rank'] = df_financial.groupby('Dataset')['ROC-AUC'].rank(ascending=False)
df_financial['PR_Rank'] = df_financial.groupby('Dataset')['PR-AUC'].rank(ascending=False)
df_financial['F1_Rank'] = df_financial.groupby('Dataset')['F1-Score'].rank(ascending=False)

            # avg_ranks["ROC-AUC"] * 0.40
            # + avg_ranks["PR-AUC"] * 0.35
            # + avg_ranks["F1-Score"] * 0.25
# df_financial['Composite_Rank'] = (df_financial['ROC_Rank'] + df_financial['PR_Rank'] + df_financial['F1_Rank']) / 3
df_financial['Composite_Rank'] = df_financial['ROC_Rank'] * 0.40 + df_financial['PR_Rank'] * 0.35 + df_financial['F1_Rank'] * 0.25

rank_summary = df_financial.groupby('Model')['Composite_Rank'].mean().reset_index().sort_values(by='Composite_Rank')

# (b) 그래프 역시 모델 순서에 맞는 고유 색상 리스트를 생성하여 매핑
colors_for_b = [model_colors[m] for m in rank_summary['Model']]
 
sns.barplot(
    data=rank_summary, 
    x='Composite_Rank', 
    y='Model', 
    ax=axes[1], 
    palette=colors_for_b
)
axes[1].set_title('(b) Average Composite Rank in Financial Domains', fontsize=13, fontweight='bold', pad=15)
axes[1].set_xlabel('Average Rank (Lower is Better)', labelpad=10)
axes[1].set_ylabel('Evaluation Model', labelpad=10)

# 제안 모델(DLG)의 가시성을 위해 테두리 강조선 적용 (색상은 유지)
for p, model_name in zip(axes[1].patches, rank_summary['Model']):
    if model_name == 'DLG':
        p.set_alpha(1.0)
        p.set_edgecolor('black')      # 검은색 테두리선
        p.set_linewidth(2.0)          # 선 두께 강조
    else:
        p.set_alpha(0.85)             # 타 모델은 부드럽게 처리

# 5. 상단 통합 범례 생성 및 간격 배치 최적화 (겹침 해결 핵심)
handles, labels = axes[0].get_legend_handles_labels()
# bbox_to_anchor 조정 및 rect 상단 마진을 0.86으로 낮추어 타이틀과 범례 사이 가독성 확보
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True, facecolor='white')
plt.tight_layout(rect=[0, 0, 1, 0.86])

# 6. 고해상도 이미지 저장 (300 DPI)
plt.savefig('./dlg_gnn/docs/work_reports/35-dlg_full_pipeline_benchmark/financial_domain_benchmark_results2.png', dpi=300, bbox_inches='tight')
print("수정된 그래프가 'financial_domain_benchmark_results.png'로 저장되었습니다.")
plt.show()