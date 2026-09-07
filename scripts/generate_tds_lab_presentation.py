"""
scripts/generate_tds_lab_presentation.py

Generates a 10-slide high-quality presentation deck in PPTX format for the lab meeting:
"A Validity-First Evaluation of Streaming Graph Neural Networks for Financial Fraud Detection under Temporal Distribution Shift"
Focuses on the research evolution from GraphRAG to Temporal Distribution Shift (TDS),
backed by Round 7 experimental results and the final paper manuscript (_43_01_TDS/tds.tex).
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Color Palette ─────────────────────────────────────────────────────────────
COLOR_DARK_NAVY   = RGBColor(15, 23, 42)     # #0F172A - Titles, Headers
COLOR_TEXT_MAIN   = RGBColor(51, 65, 85)     # #334155 - Body text
COLOR_TEXT_MUTED  = RGBColor(100, 116, 139)  # #64748B - Subtitles, Captions
COLOR_PRIMARY     = RGBColor(29, 78, 216)    # #1D4ED8 - Core Blue
COLOR_PRIMARY_BG  = RGBColor(239, 246, 255)  # #EFF6FF - Light Blue tint
COLOR_ACCENT_GREEN= RGBColor(16, 185, 129)   # #10B981 - Success, Best metrics
COLOR_ACCENT_AMBER= RGBColor(217, 119, 6)    # #D97706 - Warning, Trade-offs
COLOR_ACCENT_RED  = RGBColor(225, 29, 72)     # #E11D48 - Bottlenecks, Audits
COLOR_CARD_BG     = RGBColor(255, 255, 255)  # #FFFFFF - Clean Card background
COLOR_CARD_BORDER = RGBColor(226, 232, 240)  # #E2E8F0 - Subtle border
COLOR_ALT_ROW     = RGBColor(248, 250, 252)  # #F8FAFC - Table alternate row
COLOR_HEADER_BG   = RGBColor(30, 41, 59)     # #1E293B - Table header dark navy

FONT_HEADING = "Segoe UI"
FONT_BODY    = "Segoe UI"


def create_base_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def add_slide_header(slide, slide_num: int, category: str, title: str, subtitle: str = ""):
    """Standardized top header bar for consistent layout."""
    # Top Category Badge & Slide Number
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.35))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = f"SECTION {slide_num:02d} | {category.upper()}"
    p_cat.font.name = FONT_HEADING
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_PRIMARY

    # Main Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.733), Inches(0.55))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_DARK_NAVY

    # Subtitle if exists
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.25), Inches(11.733), Inches(0.35))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED


def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
    """Creates a rounded card shape for modular content."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_slide_1_title(prs):
    """Slide 1: Title Slide"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Decorative background card
    add_card(slide, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9), COLOR_CARD_BG, COLOR_PRIMARY)

    # Tag Badge
    tag_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.4), Inches(1.3), Inches(3.8), Inches(0.4))
    tag_shape.fill.solid()
    tag_shape.fill.fore_color.rgb = COLOR_PRIMARY_BG
    tag_shape.line.color.rgb = COLOR_PRIMARY
    tag_tf = tag_shape.text_frame
    tag_tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tag_p = tag_tf.paragraphs[0]
    tag_p.text = "연구 논문 초안 및 랩미팅 보고 자료"
    tag_p.font.name = FONT_HEADING
    tag_p.font.size = Pt(11)
    tag_p.font.bold = True
    tag_p.font.color.rgb = COLOR_PRIMARY
    tag_p.alignment = PP_ALIGN.CENTER

    # Main Paper Title
    title_box = slide.shapes.add_textbox(Inches(1.4), Inches(1.9), Inches(10.5), Inches(1.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "A Validity-First Evaluation of Streaming Graph Neural Networks"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_DARK_NAVY

    p2 = tf.add_paragraph()
    p2.text = "for Financial Fraud Detection under Temporal Distribution Shift"
    p2.font.name = FONT_HEADING
    p2.font.size = Pt(28)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_PRIMARY

    # Subtitle / Core Theme
    sub_box = slide.shapes.add_textbox(Inches(1.4), Inches(3.8), Inches(10.5), Inches(0.9))
    tf_sub = sub_box.text_frame
    tf_sub.word_wrap = True
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "핵심 연구 방향 전환: GraphRAG 기반 사기 탐지에서 '시간적 인과성 및 유병률 급변(TDS) 평가'로의 고도화"
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(15)
    p_sub.font.bold = True
    p_sub.font.color.rgb = COLOR_ACCENT_AMBER

    # Presenter Information Card
    info_card = add_card(slide, Inches(1.4), Inches(4.8), Inches(10.5), Inches(1.4), COLOR_PRIMARY_BG, None)
    info_box = slide.shapes.add_textbox(Inches(1.6), Inches(4.9), Inches(10.1), Inches(1.2))
    tf_info = info_box.text_frame
    tf_info.word_wrap = True
    
    p_info1 = tf_info.paragraphs[0]
    p_info1.text = "• 발표자: 박성수 (아주대학교 대학원 컴퓨터공학과)"
    p_info1.font.name = FONT_BODY
    p_info1.font.size = Pt(13)
    p_info1.font.bold = True
    p_info1.font.color.rgb = COLOR_DARK_NAVY

    p_info2 = tf_info.add_paragraph()
    p_info2.text = "• 지도교수: 김기형 교수님 (사이버보안학과 / 컴퓨터공학과)"
    p_info2.font.name = FONT_BODY
    p_info2.font.size = Pt(13)
    p_info2.font.color.rgb = COLOR_TEXT_MAIN

    p_info3 = tf_info.add_paragraph()
    p_info3.text = "• 기준 버전: 310_graphRAG_dataset_round_7 최종 검증 완료본 & _43_01_TDS IEEE-format 논문 초안"
    p_info3.font.name = FONT_BODY
    p_info3.font.size = Pt(12)
    p_info3.font.color.rgb = COLOR_TEXT_MUTED


def build_slide_2_initial_graphrag(prs):
    """Slide 2: Background & Initial Goal (The GraphRAG Phase)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide, 2, "Research Origins",
        "1. 연구의 출발점과 초기 기획 (GraphRAG 사기 캠페인 탐지)",
        "온체인 금융 거래망(DLG-GNN)과 오프체인 소셜 캠페인 지식(GraphRAG)을 융합하려는 시도"
    )

    # 3 Cards Layout
    w_card = Inches(3.64)
    h_card = Inches(5.1)
    y_top = Inches(1.7)

    # Card 1: Initial Problem Statement
    add_card(slide, Inches(0.8), y_top, w_card, h_card)
    tb1 = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(0.2), w_card - Inches(0.4), h_card - Inches(0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "초기 연구 기획 의도"
    p.font.name = FONT_HEADING
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets1 = [
        ("단일 온체인 탐지의 한계", "블록체인 거래 그래프만으로는 신규 생성된 사기 지갑(Cold-start)이나 사전 홍보 활동을 사전에 감지하기 어려움."),
        ("Cross-Layer 시너지 가설", "오프체인 소셜 바운티 캠페인(Telegram/Bitcointalk)의 텍스트 홍보글과 온체인 지갑 주소를 연결하면 조기 경보 가능할 것으로 기대."),
        ("Multi-Modal GraphRAG", "소셜 캠페인 텍스트 의미(Semantic) + 이종 지식 그래프(Topology)를 검색 증강하여 위험 벡터 v2 생성."),
    ]
    for title, desc in bullets1:
        p_t = tf1.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf1.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_MAIN

    # Card 2: 4-Dataset Integration
    add_card(slide, Inches(4.84), y_top, w_card, h_card)
    tb2 = slide.shapes.add_textbox(Inches(5.04), y_top + Inches(0.2), w_card - Inches(0.4), h_card - Inches(0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "4대 이종 데이터셋 연계"
    p.font.name = FONT_HEADING
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_DARK_NAVY

    bullets2 = [
        ("CCC 데이터셋 (소셜)", "15,870개 캠페인 이벤트, 185,709명 참여자 토론 및 바운티 홍보 텍스트."),
        ("CryptoScamTracker (CST)", "10,079건의 사기 도메인 ↔ 암호화폐 지갑 주소 매핑 레지스트리."),
        ("CryptoScamDB (CSDB)", "9,889건의 악성 URL / 주소 신고 데이터베이스."),
        ("GoG (온체인 트랜잭션)", "Ethereum, BSC, Polygon 기반의 대규모 다중 체인 금융 트랜잭션 서브그래프."),
    ]
    for title, desc in bullets2:
        p_t = tf2.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = COLOR_PRIMARY
        p_d = tf2.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_MAIN

    # Card 3: Early Apparent Success
    add_card(slide, Inches(8.88), y_top, w_card, h_card)
    tb3 = slide.shapes.add_textbox(Inches(9.08), y_top + Inches(0.2), w_card - Inches(0.4), h_card - Inches(0.4))
    tf3 = tb3.text_frame
    tf3.word_wrap = True

    p = tf3.paragraphs[0]
    p.text = "초기 실험의 외견상 성과"
    p.font.name = FONT_HEADING
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN

    bullets3 = [
        ("2-hop 검색 확장 이점", "0-hop(단일 텍스트) 대비 2-hop 지식 확장 시 Precision@5가 0.0664 → 0.1536 (+131.3%) 대폭 향상."),
        ("높은 초기 지표 보고", "초기 융합 실험에서 AUC-PR 0.97~0.99 수준의 극도로 우수한 탐지율 기록."),
        ("15일 사전 경보 가설", "소셜 홍보 시점이 온체인 정산보다 평균 15.2일 앞선다는 사전 경보 리드타임 관측."),
    ]
    for title, desc in bullets3:
        p_t = tf3.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf3.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_MAIN

    # Bottom Alert Note
    p_alert = tf3.add_paragraph()
    p_alert.text = "\n⚠️ 그러나 심층 감사(Round 1~4)를 거치며 이러한 외견상 고성능의 이면에 심각한 인과적/통계적 결함이 드러남!"
    p_alert.font.size = Pt(11)
    p_alert.font.bold = True
    p_alert.font.color.rgb = COLOR_ACCENT_RED


def build_slide_3_graphrag_audits(prs):
    """Slide 3: Critical Bottlenecks & Scientific Audits in GraphRAG (Why the Shift Occurred)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide, 3, "Scientific Audit & Fail-Closed Decision",
        "2. 심층 과학적 감사 결과와 방향 전환의 필연성 (Why TDS?)",
        "피상적 성능 유지를 거부하고 과학적 진실성을 위해 사기(Scam) Gate를 닫은 이유"
    )

    y_top = Inches(1.7)
    w_card = Inches(5.66)
    h_card = Inches(2.45)

    # Box 1: Degree Shortcut Discovery
    add_card(slide, Inches(0.8), y_top, w_card, h_card)
    tb1 = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(0.15), w_card - Inches(0.4), h_card - Inches(0.3))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "결함 1: 텍스트 의미론 부재 & 그래프 차수 지름길 (Degree Shortcut)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED

    p_body = tf1.add_paragraph()
    p_body.text = (
        "• 심층 Ablation 실험 결과 (Round 4):\n"
        "  - GraphRAG 2-hop AUC-PR: 0.9737\n"
        "  - GraphRAG 2-hop (텍스트 임베딩 제거 시): 0.9804 (오히려 상승!)\n"
        "  - Degree-only (단순 노드 차수만 사용): 0.9755\n"
        "• 결론: 모델이 소셜 텍스트 의미(Semantics)를 이해한 것이 아니라, 사기 집합과 정상 집합 간 '단순 그래프 연결 차수 차이'라는 데이터셋 구축 아티팩트를 외운 것에 불과함."
    )
    p_body.font.size = Pt(11)
    p_body.font.color.rgb = COLOR_TEXT_MAIN

    # Box 2: Lack of Independent Benign Ground Truth
    add_card(slide, Inches(6.86), y_top, w_card, h_card)
    tb2 = slide.shapes.add_textbox(Inches(7.06), y_top + Inches(0.15), w_card - Inches(0.4), h_card - Inches(0.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "결함 2: 독립적 정상군(Benign Ground Truth)의 원천적 부재"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED

    p_body = tf2.add_paragraph()
    p_body.text = (
        "• CCC 데이터셋은 홍보/바운티 캠페인 집합이며 순수 정상군이 아님.\n"
        "• 신고되지 않았다는 이유로 정상(Benign)으로 간주할 경우 False Negative 위험.\n"
        "• 신뢰할 수 있는 정상군 확정을 위해서는 최소 300+건 이상의 '인간 전문가 독립 이중 검증(Human Double-Annotation)'이 필요하나 외부 작업 미확보.\n"
        "• 정상 라벨 없이 도출된 AUC-PR 0.99는 학술적으로 방어 불가."
    )
    p_body.font.size = Pt(11)
    p_body.font.color.rgb = COLOR_TEXT_MAIN

    # Box 3: Missing Real Wallet Transaction Lineage
    y_top2 = Inches(4.35)
    add_card(slide, Inches(0.8), y_top2, w_card, h_card)
    tb3 = slide.shapes.add_textbox(Inches(1.0), y_top2 + Inches(0.15), w_card - Inches(0.4), h_card - Inches(0.3))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "결함 3: 온체인 지갑 트랜잭션 Lineage 단절 & 리드타임 붕괴"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED

    p_body = tf3.add_paragraph()
    p_body.text = (
        "• 실제 GoG 온체인 컨트랙트(23,967개)와 레지스트리 사기 지갑(6,871개) 간 실제 교집합 = 0건!\n"
        "• 기존 연결은 CCC 포럼 계정 등록 지갑을 통한 프록시 연결이었음.\n"
        "• 실제 트랜잭션 타임스탬프 기반 리드타임 재검증 시, 유효한 social→on-chain 쌍이 전무하여 '15일 사전 경보' 주장을 철회해야 했음."
    )
    p_body.font.size = Pt(11)
    p_body.font.color.rgb = COLOR_TEXT_MAIN

    # Box 4: The Strategic Decision (Fail-Closed Gates)
    add_card(slide, Inches(6.86), y_top2, w_card, h_card, COLOR_PRIMARY_BG, COLOR_PRIMARY)
    tb4 = slide.shapes.add_textbox(Inches(7.06), y_top2 + Inches(0.15), w_card - Inches(0.4), h_card - Inches(0.3))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    p = tf4.paragraphs[0]
    p.text = "핵심 연구 전략적 결단: Gate A/B 봉쇄 및 본질적 난제로 전환"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    p_body = tf4.add_paragraph()
    p_body.text = (
        "• Gate A (Scam GraphRAG) & Gate B (Cross-Layer) = FALSE로 동결!\n"
        "• 학술적 정직성 준수: 결함이 있는 데이터를 억지로 포장하지 않고 fail-closed 원칙에 따라 사기 캠페인 브랜치를 연구적 유보(Hold) 상태로 전환.\n"
        "• 새로운 연구 기회 포착: 금융 거래망에서 실제로 가장 치명적이고 풀리지 않은 근본 문제인 '시간적 분포 이동(Temporal Distribution Shift)' 및 '인과적 평가 타당성(Validity-First)'으로 전면 재정의!"
    )
    p_body.font.size = Pt(11)
    p_body.font.bold = True
    p_body.font.color.rgb = COLOR_DARK_NAVY


def build_slide_4_tds_paradigm(prs):
    """Slide 4: Research Pivot: Validity-First Evaluation under Temporal Distribution Shift (TDS)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide, 4, "Core Problem Redefinition",
        "3. 연구 패러다임 전환: 왜 Temporal Distribution Shift (TDS)인가?",
        "기존 금융 GNN 논문들의 3대 치명적 평가 결함을 타파하는 'Validity-First' 방법론 정립"
    )

    y_top = Inches(1.7)
    w_card = Inches(3.64)
    h_card = Inches(5.1)

    # 3 Pitfalls of Conventional Fraud GNNs
    # Pitfall 1: Future-Edge Leakage
    add_card(slide, Inches(0.8), y_top, w_card, h_card)
    tb1 = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(0.2), w_card - Inches(0.4), h_card - Inches(0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "기존 한계 1: 미래 정보 누출\n(Future-Edge Leakage)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED

    bullets1 = [
        ("사후 지식 오염", "예측 대상 시점 $t_i$ 이후에 발생한 미래 거래 간선, 노드 차수, 전역 임베딩이 전처리 과정에서 유입됨."),
        ("허위 고성능 유발", "실제 배포 환경에서는 미래의 거래망을 알 수 없으나, 기존 정적 GNN 논문들은 사후에 형성된 거대 사기 클러스터를 보고 쉽게 분류함."),
        ("Causal Cutoff 필수", "모든 이벤트 $i$는 반드시 $t \le t_i$인 과거 간선만으로 그래프를 동적 구축해야 함."),
    ]
    for title, desc in bullets1:
        p_t = tf1.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf1.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_MAIN

    # Pitfall 2: Artificial Stationarity
    add_card(slide, Inches(4.84), y_top, w_card, h_card)
    tb2 = slide.shapes.add_textbox(Inches(5.04), y_top + Inches(0.2), w_card - Inches(0.4), h_card - Inches(0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "기존 한계 2: 무작위 분할의 착시\n(Artificial Stationarity)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_AMBER

    bullets2 = [
        ("Random Split의 문제", "과거와 미래 이벤트를 무작위로 섞어 학습/평가함으로써 시계열적 행동 변화를 은폐함."),
        ("공격자 패턴 진화", "실제 블록체인 사기꾼들은 세탁 방식과 거래 토폴로지를 끊임없이 진화시키므로 시간적 단절이 필연적임."),
        ("엄격한 시계열 분할", "Train(과거) → Val(중간) → Test(미래)로 시간 축을 단방향으로 엄격히 동결 분할해야 진정한 일반화 검증 가능."),
    ]
    for title, desc in bullets2:
        p_t = tf2.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf2.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_MAIN

    # Pitfall 3: Prevalence Shift & Miscalibration
    add_card(slide, Inches(8.88), y_top, w_card, h_card)
    tb3 = slide.shapes.add_textbox(Inches(9.08), y_top + Inches(0.2), w_card - Inches(0.4), h_card - Inches(0.4))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "기존 한계 3: 유병률 급변과 보정\n(Prevalence Shift & Calibration)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets3 = [
        ("극단적 양성 유병률 감소", "학습 시점(39.8%) 대비 배포 시점(2.9%)의 사기 비율이 1/13 수준으로 폭락."),
        ("확률 보정(Calibration) 왜곡", "유병률이 급변하면 모델이 출력하는 예측 확률값이 실제 사기 발생 빈도와 심각하게 어긋나 경보 임계치 설정 불가."),
        ("AUC-PR 중심 평가", "ROC-AUC는 불균형 하에서 착시를 주므로 AUC-PR 및 ECE/NLL 보정 지표 동시 평가가 필수적임."),
    ]
    for title, desc in bullets3:
        p_t = tf3.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf3.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_MAIN


def build_slide_5_benchmark_recovery(prs):
    """Slide 5: Benchmark Recovery & Strict Causal Protocol (GoG-SCIMain-v1)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide, 5, "Benchmark Provenance & Leakage Audit",
        "4. GoG-SCIMain-v1 벤치마크 완전 복원 및 인과적 스트리밍 환경 구축",
        "NeurIPS 2024 공식 Multi-Chain GoG 24,316개 원본 파일 100% 해시 일치 복원 및 전수 누출 감사"
    )

    y_top = Inches(1.7)

    # Left Column: Statistics & Audits (Width: 6.2 Inches)
    w_left = Inches(6.0)
    add_card(slide, Inches(0.8), y_top, w_left, Inches(5.1))
    tb = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(0.2), w_left - Inches(0.4), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "공식 원천 데이터 완전 복원 및 무결성 검증"
    p.font.name = FONT_HEADING
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    text_content = (
        "• 공식 출처 복원: NeurIPS 2024 Graphs-of-Graphs 공식 배포본 확보\n"
        "  - Ethereum: 14,464개 파일 (100% SHA-256 일치)\n"
        "  - BSC (Binance Smart Chain): 7,499개 파일 (100% 일치)\n"
        "  - Polygon: 2,353개 파일 (100% 일치)\n"
        "  - 총 24,316 / 24,316개 소스 파일 전수 해시 일치 확인\n\n"
        "• 동결 아티팩트 해시 일치:\n"
        "  - graph.pt, transactions.parquet, split_manifest.json, future_edge_audit.csv\n\n"
        "• 전수 인과 누출 감사 (Exhaustive Causal Audit):\n"
        "  - 24,316개 이벤트별 국소 그래프 전수 검사\n"
        "  - 각 이벤트 $i$의 예측 시점 $t_i$ 이후 간선 포함 여부 확인\n"
        "  - 결과: 미래 간선 위반(Future-Edge Violations) = 0건 (Zero Leakage)"
    )
    p_b = tf.add_paragraph()
    p_b.text = text_content
    p_b.font.size = Pt(11)
    p_b.font.color.rgb = COLOR_TEXT_MAIN

    # Split Table inside left card
    p_tbl = tf.add_paragraph()
    p_tbl.text = "\n[표] GoG-SCIMain-v1 시계열 분할 및 유병률 급변:"
    p_tbl.font.bold = True
    p_tbl.font.size = Pt(11)
    p_tbl.font.color.rgb = COLOR_DARK_NAVY

    split_info = (
        "  - Train (70%): 17,021 events | 6,783 사기 (39.85% 유병률)\n"
        "  - Val   (15%):  3,647 events |   308 사기 ( 8.45% 유병률)\n"
        "  - Test  (15%):  3,648 events |   107 사기 ( 2.93% 유병률) -> 1/13로 격감!"
    )
    p_split = tf.add_paragraph()
    p_split.text = split_info
    p_split.font.size = Pt(11)
    p_split.font.bold = True
    p_split.font.color.rgb = COLOR_ACCENT_AMBER

    # Right Column: Embedded Figure (Width: 5.4 Inches)
    w_right = Inches(5.4)
    add_card(slide, Inches(7.133), y_top, w_right, Inches(5.1))
    
    tb_r = slide.shapes.add_textbox(Inches(7.333), y_top + Inches(0.2), w_right - Inches(0.4), Inches(0.5))
    tf_r = tb_r.text_frame
    p_r = tf_r.paragraphs[0]
    p_r.text = "시간적 인과 스트리밍 그래프 파이프라인"
    p_r.font.name = FONT_HEADING
    p_r.font.size = Pt(14)
    p_r.font.bold = True
    p_r.font.color.rgb = COLOR_DARK_NAVY

    # Add Image: causal_pipeline.png
    img_path = "dlg_gnn/figures/main_final/causal_pipeline.png"
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(7.333), y_top + Inches(0.8), width=Inches(5.0))

    # Caption / Description
    tb_cap = slide.shapes.add_textbox(Inches(7.333), y_top + Inches(3.8), w_right - Inches(0.4), Inches(1.1))
    tf_cap = tb_cap.text_frame
    tf_cap.word_wrap = True
    p_cap = tf_cap.paragraphs[0]
    p_cap.text = (
        "• 각 이벤트 $i$ 발생 시점 $t_i$ 기준으로 과거 최신 128개 간선만을 추출.\n"
        "• 3개 노드 특성 + 체인 공변량(Chain Covariate)을 결합하여 국소 그래프 구성.\n"
        "• 모든 모델(제안 기법 및 3종 베이스라인)이 동일한 인과 정보 경계를 엄격히 공유."
    )
    p_cap.font.size = Pt(10.5)
    p_cap.font.color.rgb = COLOR_TEXT_MAIN


def build_slide_6_methodology(prs):
    """Slide 6: Proposed Methodology: CausalLocalGIN & Evaluated Baselines"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide, 6, "Proposed Architecture & Baselines",
        "5. 제안 기법(CausalLocalGIN) 및 공정한 시계열 비교군 설계",
        "동일한 인과적 정보 경계 하에서 5개 독립 시드, 단 1회 테스트셋 접근(Test-Once) 원칙 준수"
    )

    y_top = Inches(1.7)
    w_col = Inches(5.66)
    h_col = Inches(5.1)

    # Left: CausalLocalGIN Architecture
    add_card(slide, Inches(0.8), y_top, w_col, h_col)
    tb1 = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(0.2), w_col - Inches(0.4), h_col - Inches(0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "제안 모델: CausalLocalGIN (인과적 국소 GIN)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets1 = [
        ("인과적 국소 구조", "전역 그래프 연산 대신, 이벤트 시점 $t_i$에 도달 가능한 $\le 128$개의 최신 과거 간선으로 에고-서브그래프(Ego-subgraph) 구성."),
        ("GIN 메시지 패싱", "그래프 동형 식별력이 우수한 GIN(Graph Isomorphism Network) 2-hop 계층 적용: $h_v^{(k)} = \\text{MLP}^{(k)}((1+\\epsilon)h_v^{(k-1)} + \\sum_{u} h_u^{(k-1)})$."),
        ("체인 공변량 결합", "Ethereum, BSC, Polygon 간 거래 양식 차이를 보정하기 위해 원-핫 체인 공변량(Chain Covariate)을 readout 벡터와 결합."),
        ("경량화 및 스트리밍 적합", "총 파라미터 수 47,873개로 극히 경량화되어 단일 이벤트 추론 시 서브-밀리초(0.83ms) 처리 보장."),
    ]
    for title, desc in bullets1:
        p_t = tf1.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(12.5)
        p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf1.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_MAIN

    # Right: Rigorous Baselines & Uncertainty Controls
    add_card(slide, Inches(6.86), y_top, w_col, h_col)
    tb2 = slide.shapes.add_textbox(Inches(7.06), y_top + Inches(0.2), w_col - Inches(0.4), h_col - Inches(0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "동일 프로토콜 기반 3대 베이스라인 & 불확실성 제어군"
    p.font.name = FONT_HEADING
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_DARK_NAVY

    bullets2 = [
        ("TGAT (시간 어텐션 GNN)", "이벤트 간 시간 차이를 연속 시간 임베딩으로 변환하여 셀프 어텐션 수행 (Xu et al., ICLR 2020). 파라미터 48,001개."),
        ("TGN (이벤트 메모리 GNN)", "이벤트가 발생할 때마다 노드 상태를 업데이트하는 연속 시간 메모리 모듈 탑재 (Rossi et al., ICML 2020). 파라미터 48,129개."),
        ("Fraud-oriented GraphSAGE", "사기 탐지 분야 대표 모델(CARE-GNN 기반 모티브)로 이상 이웃 가중 집계 수행. 파라미터 47,489개."),
        ("불확실성 및 보정 통제군", (
            "① MC Dropout ($T=10$ 추론 반복)\n"
            "② Validation-only Temperature Scaling (검증셋 로짓으로 단일 스칼라 $T$ 적합)\n"
            "③ 5-Model Deep Ensemble (독립 5개 시드 앙상블)"
        )),
    ]
    for title, desc in bullets2:
        p_t = tf2.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(12.5)
        p_t.font.color.rgb = COLOR_PRIMARY
        p_d = tf2.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_MAIN


def build_slide_7_main_results(prs):
    """Slide 7: Main Empirical Findings (Ranking vs Baseline Models)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide, 7, "Experimental Results 1: Detection Performance",
        "6. 핵심 실험 결과: 모델별 사기 탐지 및 랭킹 성능 비교",
        "5개 시드 held-out 평가 및 10,000회 페어드 부트스트랩 / 랜덤화 검정 결과 (Table 5 & 6)"
    )

    y_top = Inches(1.7)
    
    # Left Card: Main Results Table (Width: 7.6 Inches)
    w_tbl = Inches(7.5)
    add_card(slide, Inches(0.8), y_top, w_tbl, Inches(5.1))
    
    tb_title = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(0.15), w_tbl - Inches(0.4), Inches(0.4))
    tf_tt = tb_title.text_frame
    p_tt = tf_tt.paragraphs[0]
    p_tt.text = "GoG-SCIMain-v1 5-시드 Held-Out 성능 평가 (평균 ± 표준편차)"
    p_tt.font.name = FONT_HEADING
    p_tt.font.size = Pt(13)
    p_tt.font.bold = True
    p_tt.font.color.rgb = COLOR_DARK_NAVY

    # Add Table
    rows = 7
    cols = 6
    tbl_shape = slide.shapes.add_table(rows, cols, Inches(1.0), y_top + Inches(0.6), w_tbl - Inches(0.4), Inches(2.9))
    tbl = tbl_shape.table

    # Column widths
    tbl.columns[0].width = Inches(2.3)
    tbl.columns[1].width = Inches(1.1)
    tbl.columns[2].width = Inches(1.0)
    tbl.columns[3].width = Inches(0.9)
    tbl.columns[4].width = Inches(0.9)
    tbl.columns[5].width = Inches(0.9)

    headers = ["Method", "AUC-PR (주지표)", "ROC-AUC", "F1", "ECE (보정)", "NLL"]
    for i, h in enumerate(headers):
        cell = tbl.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_HEADER_BG
        p = cell.text_frame.paragraphs[0]
        p.font.name = FONT_HEADING
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    data = [
        ["CausalLocalGIN (제안)", "0.4648 ± 0.069", "0.9009 ± 0.006", "0.4782", "0.1672", "0.2365"],
        ["MC Dropout GIN (T=10)", "0.4416 ± 0.086", "0.8997 ± 0.018", "0.4230", "0.1285", "0.1935"],
        ["Temp-Scaled GIN (제안+보정)", "0.4648 ± 0.069", "0.9009 ± 0.006", "0.4782", "0.0357", "0.1070"],
        ["TGAT (시간 어텐션)", "0.3412 ± 0.047", "0.8946 ± 0.017", "0.2432", "0.1013", "0.1688"],
        ["TGN (이벤트 메모리)", "0.2324 ± 0.009", "0.8983 ± 0.004", "0.2115", "0.0380", "0.1244"],
        ["Fraud-oriented GraphSAGE", "0.3084 ± 0.033", "0.8914 ± 0.010", "0.2900", "0.2595", "0.3548"],
    ]
    for r_idx, row_data in enumerate(data, start=1):
        for c_idx, val in enumerate(row_data):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            if r_idx in [1, 3]: # Proposed highlights
                cell.fill.fore_color.rgb = COLOR_PRIMARY_BG
            else:
                cell.fill.fore_color.rgb = COLOR_ALT_ROW if r_idx % 2 == 1 else COLOR_CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT_BODY
            p.font.size = Pt(9.5)
            if c_idx == 0:
                p.alignment = PP_ALIGN.LEFT
                p.font.bold = (r_idx in [1, 3])
            else:
                p.alignment = PP_ALIGN.CENTER
                if c_idx == 1 and r_idx in [1, 3]:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_PRIMARY
                if c_idx == 4 and r_idx == 3:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_ACCENT_GREEN

    # Statistical significance summary box under table
    tb_stat = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(3.6), w_tbl - Inches(0.4), Inches(1.3))
    tf_stat = tb_stat.text_frame
    tf_stat.word_wrap = True
    p_s = tf_stat.paragraphs[0]
    p_s.text = "10,000회 페어드 부트스트랩 및 랜덤화 통계 검정 결과 (Table 6):"
    p_s.font.bold = True
    p_s.font.size = Pt(11)
    p_s.font.color.rgb = COLOR_DARK_NAVY
    
    stat_bullets = (
        "• CausalLocalGIN vs. TGN: Δ AUC-PR = +0.2323 (95% CI [0.1621, 0.2969], p = 0.0001)\n"
        "• CausalLocalGIN vs. TGAT: Δ AUC-PR = +0.1236 (95% CI [0.0606, 0.1848], p = 0.0001)\n"
        "• CausalLocalGIN vs. FraudSAGE: Δ AUC-PR = +0.1564 (95% CI [0.0974, 0.2209], p = 0.0001)\n"
        "-> 제안 기법이 모든 시계열/사기 탐지 베이스라인 대비 0을 포함하지 않는 95% 신뢰구간으로 유의하게 우수함."
    )
    p_sb = tf_stat.add_paragraph()
    p_sb.text = stat_bullets
    p_sb.font.size = Pt(10)
    p_sb.font.color.rgb = COLOR_TEXT_MAIN

    # Right Card: Academic Insights (Width: 4.1 Inches)
    w_right = Inches(4.0)
    add_card(slide, Inches(8.5), y_top, w_right, Inches(5.1))
    tb_ins = slide.shapes.add_textbox(Inches(8.7), y_top + Inches(0.2), w_right - Inches(0.4), Inches(4.7))
    tf_ins = tb_ins.text_frame
    tf_ins.word_wrap = True

    p = tf_ins.paragraphs[0]
    p.text = "주요 학술적 분석 및 시사점"
    p.font.name = FONT_HEADING
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    insights = [
        ("ROC-AUC의 착시와 AUC-PR의 필연성", (
            "모든 모델의 ROC-AUC는 약 0.89~0.90으로 겉보기엔 유사함. "
            "하지만 2.93%의 극단적 불균형 하에서는 대다수 정상 샘플로 인해 ROC-AUC가 왜곡됨. "
            "AUC-PR에서 CausalLocalGIN(0.46)은 TGN(0.23) 대비 2배의 실질적 사기 식별력을 증명함."
        )),
        ("복잡한 시계열 메모리의 과적합", (
            "TGN과 같은 장기 연속 메모리 구조는 과거 학습 기간(39.85% 사기)의 패턴을 과도하게 기억하여, "
            "미래(2.93%)의 공격 패턴 진화 환경에서 오히려 성능 저하를 겪음."
        )),
        ("국소 에고-그래프의 유효성", (
            "최근 $\le 128$개 간선만을 인과적으로 집계하는 CausalLocalGIN이 동적 금융망에서 가장 강인한 일반화 능력을 보임."
        )),
    ]
    for title, desc in insights:
        p_t = tf_ins.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf_ins.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = COLOR_TEXT_MAIN


def build_slide_8_tradeoff(prs):
    """Slide 8: Core Insight: The Ranking vs Calibration Trade-Off"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide, 8, "Core Scientific Insight",
        "7. 핵심 통찰: 랭킹(Ranking)과 확률 보정(Calibration)의 트레이드오프",
        "불확실성 추정 기법이 항상 우수한 것은 아니다: MC Dropout의 한계와 Temperature Scaling의 실용성"
    )

    y_top = Inches(1.7)
    w_col = Inches(5.66)
    h_col = Inches(5.1)

    # Left: The Paradox of MC Dropout
    add_card(slide, Inches(0.8), y_top, w_col, h_col)
    tb1 = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(0.15), w_col - Inches(0.4), h_col - Inches(0.3))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "MC Dropout의 양면성: 보정 개선 vs. 랭킹 페널티"
    p.font.name = FONT_HEADING
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_AMBER

    text_mc = (
        "• 기존 문헌의 일반적 통념:\n"
        "  'MC Dropout은 베이지안 근사를 통해 불확실성을 추정하므로 성능이 향상된다.'\n\n"
        "• 본 연구의 실증적 발견 (엄격한 시간적 분포 이동 환경):\n"
        "  1) 보정 지표 개선:\n"
        "     - ECE: 0.1672 -> 0.1285 (23.1% 개선)\n"
        "     - NLL: 0.2365 -> 0.1935 (18.2% 개선)\n"
        "     - Brier Score: 0.0533 -> 0.0410 개선\n"
        "  2) 통계적으로 유의한 랭킹 손실 수반:\n"
        "     - AUC-PR: 0.4648 -> 0.4416 (Δ = -0.0232)\n"
        "     - 95% CI [-0.0385, -0.0081], Randomization p = 0.0001\n\n"
        "• 핵심 결론: MC Dropout은 확률값의 신뢰도를 높여주지만, 정작 가장 의심스러운 사기 거래를 최상위로 올리는 순위(Ranking) 역량을 유의하게 훼손함."
    )
    p_b = tf1.add_paragraph()
    p_b.text = text_mc
    p_b.font.size = Pt(10.8)
    p_b.font.color.rgb = COLOR_TEXT_MAIN

    # Add mc_tradeoff.png if exists
    img_mc = "dlg_gnn/figures/main_final/mc_tradeoff.png"
    if os.path.exists(img_mc):
        slide.shapes.add_picture(img_mc, Inches(1.0), y_top + Inches(3.4), width=Inches(5.2))

    # Right: Temperature Scaling Dominance & Deep Ensemble
    add_card(slide, Inches(6.86), y_top, w_col, h_col)
    tb2 = slide.shapes.add_textbox(Inches(7.06), y_top + Inches(0.15), w_col - Inches(0.4), h_col - Inches(0.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "사후 온도 스케일링(TS)과 딥 앙상블의 최적 결합"
    p.font.name = FONT_HEADING
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN

    text_ts = (
        "• Validation-Only Temperature Scaling의 압도적 우위:\n"
        "  - 단일 스칼라 $T$를 검증셋에서 적합 (테스트셋 비침해)\n"
        "  - 단조 변환(Monotonic)이므로 랭킹 보존: AUC-PR 0.4648 완벽 유지\n"
        "  - ECE를 0.0357로, NLL을 0.1070으로 극적 감소 (단일 모델 중 최강의 보정력)\n\n"
        "• 5-Model Deep Ensemble의 성과:\n"
        "  - 독립 5개 시드 앙상블 시 AUC-PR 0.4902 달성\n"
        "  - 온도 스케일링 적용 앙상블: AUC-PR 0.4941, ECE 0.0355, NLL 0.0987\n"
        "  -> 랭킹과 보정 양면에서 전 실험군 최고 성능 기록"
    )
    p_b2 = tf2.add_paragraph()
    p_b2.text = text_ts
    p_b2.font.size = Pt(10.8)
    p_b2.font.color.rgb = COLOR_TEXT_MAIN

    # Add reliability_comparison.png
    img_rel = "dlg_gnn/figures/main_final_v2/reliability_comparison.png"
    if os.path.exists(img_rel):
        slide.shapes.add_picture(img_rel, Inches(7.5), y_top + Inches(2.95), width=Inches(4.4))


def build_slide_9_temporal_slices(prs):
    """Slide 9: Non-Stationary Temporal Slices & Inference Latency Breakdown"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide, 9, "Deployment Viability & Temporal Robustness",
        "8. 비정상 시계열 슬라이스 분석 및 스트리밍 지연시간(Latency)",
        "테스트 기간 내 6개 연속 시계열 구간 안정성 및 실제 금융망 배포 가능성 검증 (Table 7 & 9)"
    )

    y_top = Inches(1.7)
    
    # Left Card: 6 Slices Table (Width: 7.2 Inches)
    w_left = Inches(7.0)
    add_card(slide, Inches(0.8), y_top, w_left, Inches(5.1))

    tb_t = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(0.15), w_left - Inches(0.4), Inches(0.4))
    tf_t = tb_t.text_frame
    p_t = tf_t.paragraphs[0]
    p_t.text = "테스트셋 6개 연속 시계열 슬라이스(각 608건) 세부 평가"
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(13)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_DARK_NAVY

    # Table
    rows = 7
    cols = 7
    tbl_shape = slide.shapes.add_table(rows, cols, Inches(1.0), y_top + Inches(0.55), w_left - Inches(0.4), Inches(3.0))
    tbl = tbl_shape.table

    col_widths = [Inches(0.9), Inches(0.9), Inches(1.0), Inches(1.0), Inches(0.9), Inches(0.9), Inches(1.0)]
    for i, w in enumerate(col_widths):
        tbl.columns[i].width = w

    headers = ["Slice", "사기(n)", "유병률", "제안 AP", "제안 ECE", "TS ECE", "TGAT AP"]
    for i, h in enumerate(headers):
        cell = tbl.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_HEADER_BG
        p = cell.text_frame.paragraphs[0]
        p.font.name = FONT_HEADING
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    slice_data = [
        ["1", "19", "3.13%", "0.4414", "0.1828", "0.0536", "0.2883"],
        ["2", "39", "6.41%", "0.5948", "0.1613", "0.0387", "0.2946"],
        ["3", "17", "2.80%", "0.4849", "0.1715", "0.0406", "0.4100"],
        ["4", "9",  "1.48%", "0.3925", "0.1856", "0.0501", "0.1650"],
        ["5", "7",  "1.15%", "0.4450", "0.1861", "0.0379", "0.2737"],
        ["6", "16", "2.63%", "0.7395", "0.1826", "0.0471", "0.5599"],
    ]
    for r_idx, row_data in enumerate(slice_data, start=1):
        for c_idx, val in enumerate(row_data):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_ALT_ROW if r_idx % 2 == 1 else COLOR_CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT_BODY
            p.font.size = Pt(9.5)
            p.alignment = PP_ALIGN.CENTER
            if c_idx == 3:
                p.font.bold = True
                p.font.color.rgb = COLOR_PRIMARY
            if c_idx == 5:
                p.font.bold = True
                p.font.color.rgb = COLOR_ACCENT_GREEN

    # Observation text under slice table
    tb_obs = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(3.65), w_left - Inches(0.4), Inches(1.3))
    tf_obs = tb_obs.text_frame
    tf_obs.word_wrap = True
    p_o = tf_obs.paragraphs[0]
    p_o.text = "시계열 슬라이스 분석의 핵심 발견:"
    p_o.font.bold = True
    p_o.font.size = Pt(11)
    p_o.font.color.rgb = COLOR_DARK_NAVY

    obs_bullets = (
        "• 유병률 변동(1.15% ~ 6.41%) 속에서도 제안 CausalLocalGIN은 6개 전 구간에서 TGAT 압도.\n"
        "• 온도 스케일링(TS ECE) 적용 시 전 슬라이스에서 0.038~0.053의 매우 안정적인 보정력 유지.\n"
        "• 양성 샘플 수가 적은 구간(Slice 4, 5는 사기 9건, 7건)의 AP 변동성을 진단적으로 투명하게 공개."
    )
    p_ob = tf_obs.add_paragraph()
    p_ob.text = obs_bullets
    p_ob.font.size = Pt(10)
    p_ob.font.color.rgb = COLOR_TEXT_MAIN

    # Right Card: Latency Breakdown (Width: 4.6 Inches)
    w_right = Inches(4.5)
    add_card(slide, Inches(8.0), y_top, w_right, Inches(5.1))
    tb_lat = slide.shapes.add_textbox(Inches(8.2), y_top + Inches(0.2), w_right - Inches(0.4), Inches(4.7))
    tf_lat = tb_lat.text_frame
    tf_lat.word_wrap = True

    p = tf_lat.paragraphs[0]
    p.text = "스트리밍 지연시간(Latency) 엄밀화"
    p.font.name = FONT_HEADING
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    lat_bullets = [
        ("이전 보고서의 혼선 해소", (
            "Round 6에서 이전 단일 이벤트 지연시간(~0.8ms)과 3,648건 전체 패널 배치 추론 시간(4.89s)이 혼용되었던 문제를 "
            "논문 Section VI-E에서 두 개의 독립된 추정치(Estimand)로 명확히 분리 정의함."
        )),
        ("Latency A (단일 이벤트 E2E)", (
            "• 실제 실시간 이벤트 도착 환경 측정\n"
            "• 100회 합성 시계열 이벤트 E2E 지연시간: 평균 0.83ms (서브-밀리초)\n"
            "• 초당 1,200건 이상의 고속 블록체인 트랜잭션 실시간 차단 가능"
        )),
        ("Latency B (3,648건 전체 패널 추론)", (
            "• 배치 크기 128로 전체 테스트셋 추론 시:\n"
            "  - Deterministic: 4.89초 소요\n"
            "  - MC Dropout (T=10): 48.9초 (10배 연산량)\n"
            "  - Temperature Scaling: 4.89초 (추가 연산 거의 0)"
        )),
    ]
    for title, desc in lat_bullets:
        p_t = tf_lat.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf_lat.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = COLOR_TEXT_MAIN


def build_slide_10_conclusion(prs):
    """Slide 10: Conclusion, Academic Contributions & Future Directions"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide, 10, "Summary & Future Work",
        "9. 결론, 학술적 기여도 및 향후 계획 (랩미팅 보고 요약)",
        "타협 없는 엄밀성(Gate M v8 = True)으로 완성된 IEEE-format 논문 초안 및 피드백 요청 포인트"
    )

    y_top = Inches(1.7)
    w_card = Inches(5.66)
    h_card = Inches(5.1)

    # Left: Summary of Contributions
    add_card(slide, Inches(0.8), y_top, w_card, h_card, COLOR_PRIMARY_BG, COLOR_PRIMARY)
    tb1 = slide.shapes.add_textbox(Inches(1.0), y_top + Inches(0.2), w_card - Inches(0.4), h_card - Inches(0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "논문의 핵심 학술적 기여도 (Contributions)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    contribs = [
        ("1. 재현 가능한 무누출 금융 GNN 벤치마크 확립", (
            "공식 NeurIPS 2024 GoG 24,316건 원본 100% 해시 복원 및 미래 간선 누출 0건 전수 감사로 "
            "금융 그래프 분야의 고질적 평가 타당성 결함을 극복함."
        )),
        ("2. 인과적 스트리밍 GNN의 우수성 실증", (
            "극심한 유병률 급감(39.85% -> 2.93%) 환경에서 경량 인과 GIN이 복잡한 동적 메모리 모델(TGN, TGAT)을 "
            "AUC-PR 0.4648 대 0.23~0.34로 유의하게 압도함을 5-시드 검증."
        )),
        ("3. 랭킹-보정 트레이드오프 및 최적 전략 제시", (
            "MC Dropout의 랭킹 페널티를 최초 규명하고, 사후 온도 스케일링(TS)과 딥 앙상블이 "
            "금융 사기 탐지에서 가장 실용적인 해결책임을 통계적으로 증명함."
        )),
        ("4. 완벽한 연구 아티팩트 보존 (Gate M v8 = True)", (
            "체크포인트, 원시 예측 CSV 50종, 10,000회 부트스트랩 결과 등 완전한 Lineage 구축."
        )),
    ]
    for title, desc in contribs:
        p_t = tf1.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf1.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = COLOR_TEXT_MAIN

    # Right: Future Plan & Advisor Feedback Request
    add_card(slide, Inches(6.86), y_top, w_card, h_card)
    tb2 = slide.shapes.add_textbox(Inches(7.06), y_top + Inches(0.2), w_card - Inches(0.4), h_card - Inches(0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "향후 계획 및 지도교수님 피드백 요청 포인트"
    p.font.name = FONT_HEADING
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_DARK_NAVY

    future_points = [
        ("타겟 저널 및 투고 일정", (
            "• 현재 작성 완료된 IEEEtran 초안(tds.tex)을 기반으로 IEEE Transactions on Information Forensics and Security (TIFS) "
            "또는 IEEE Transactions on Neural Networks and Learning Systems (TNNLS) 고려 중.\n"
            "• 교수님께서 권장하시는 타겟 저널 및 보완 방향 자문 요청."
        )),
        ("사기(Scam/GraphRAG) 브랜치의 후속 연계 로드맵", (
            "• 현재 본 논문에서는 fail-closed로 분리 보존된 GraphRAG 브랜치를,\n"
            "  향후 '인간 전문가 이중 어노테이션(300건)' 및 '실제 지갑 트랜잭션 API 스크래핑' 확보 시 "
            "후속 Part-2 논문(Cross-Layer Social Fraud Detection)으로 확장할 계획임."
        )),
        ("오늘 랩미팅 주요 논의 요청 사항", (
            "① 논문 제목('A Validity-First Evaluation...')의 적절성\n"
            "② 서론에서 'GraphRAG 한계 -> TDS 전환' 배경을 간략히 언급할지 여부\n"
            "③ 투고 전 추가할 실험이 있는지에 대한 검토"
        )),
    ]
    for title, desc in future_points:
        p_t = tf2.add_paragraph()
        p_t.text = f"\n• {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = COLOR_PRIMARY
        p_d = tf2.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = COLOR_TEXT_MAIN


def main():
    print("Initializing 16:9 Presentation Builder...")
    prs = create_base_presentation()

    print("Building Slide 1: Title...")
    build_slide_1_title(prs)

    print("Building Slide 2: Research Origins (GraphRAG)...")
    build_slide_2_initial_graphrag(prs)

    print("Building Slide 3: Scientific Audit & Why TDS...")
    build_slide_3_graphrag_audits(prs)

    print("Building Slide 4: TDS Paradigm & Problem Definition...")
    build_slide_4_tds_paradigm(prs)

    print("Building Slide 5: GoG-SCIMain-v1 Benchmark Recovery & Audit...")
    build_slide_5_benchmark_recovery(prs)

    print("Building Slide 6: Proposed Methodology & Baselines...")
    build_slide_6_methodology(prs)

    print("Building Slide 7: Detection Performance & Statistics...")
    build_slide_7_main_results(prs)

    print("Building Slide 8: Ranking vs Calibration Trade-Off...")
    build_slide_8_tradeoff(prs)

    print("Building Slide 9: Non-Stationary Slices & Latency...")
    build_slide_9_temporal_slices(prs)

    print("Building Slide 10: Conclusion & Discussion...")
    build_slide_10_conclusion(prs)

    output_path = "/mnt/d/_Work/goat_bank/dlg_gnn/docs/papers/_43_01_TDS/43_01_TDS_LabMeeting_Presentation.pptx"
    prs.save(output_path)
    print(f"Successfully generated presentation: {output_path}")

    # Also save a copy in 310_graphRAG_dataset_round_7 for easy access
    r7_path = "/mnt/d/_Work/goat_bank/dlg_gnn/docs/work_reports/310_graphRAG_dataset_round_7/43_01_TDS_LabMeeting_Presentation.pptx"
    prs.save(r7_path)
    print(f"Saved duplicate copy to: {r7_path}")


if __name__ == "__main__":
    main()
