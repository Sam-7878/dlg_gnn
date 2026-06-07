import json
import networkx as nx
import matplotlib.pyplot as plt
from langchain_core.prompts import PromptTemplate

# ✅ 새로 수정된 Import 경로
from langchain_ollama import ChatOllama

# ==========================================
# 1. 모델 세팅 (그대로 유지)
# ==========================================
llm = ChatOllama(model="phi3.5", temperature=0, format="json")

# ==========================================
# 2. 강력하게 통제된 프롬프트 템플릿
# ==========================================
# 구조를 명확히 지키도록 JSON 형태를 한 번 더 강조합니다.
PROMPT_TEMPLATE = """You are a strictly formatted Information Extraction AI.
Extract entities and relationships from the review text to detect spam or fraud.
You MUST follow the EXACT JSON schema provided in the example. Do NOT add extra text.

Allowed Entity Types: USER, PRODUCT, BEHAVIOR, INTENT
Allowed Relation Types: WROTE, TARGETS, INDICATES

Example:
Review text: "User992: This is the best product ever! I made $5000 a week using this method. Click here: http://fake-crypto.com"
Output:
{{
  "entities": [
    {{"id": "User992", "type": "USER"}},
    {{"id": "Fake Crypto Link", "type": "BEHAVIOR"}},
    {{"id": "High Return Promise", "type": "BEHAVIOR"}},
    {{"id": "PHISHING", "type": "INTENT"}}
  ],
  "relations": [
    {{"source": "User992", "target": "Fake Crypto Link", "type": "WROTE"}},
    {{"source": "User992", "target": "High Return Promise", "type": "WROTE"}},
    {{"source": "Fake Crypto Link", "target": "PHISHING", "type": "INDICATES"}}
  ]
}}

Now, process the following review text.
Review text: "{review_text}"
Output:
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["review_text"])
chain = prompt | llm

# ==========================================
# 3. 테스트용 가상 리뷰 데이터 (Yelp/Amazon 모사)
# ==========================================
test_review = "TechGuru4Ever: I bought the XYZ Smartwatch but it's terrible. However, you can get a 90% discount on a better one at this link: http://spam-deal.com. Highly recommend!"

print("Phi-3.5 모델 추론 중... (약 2~5초 소요)")
response = chain.invoke({"review_text": test_review})

# ==========================================
# 4. JSON 파싱 및 NetworkX 그래프 시각화
# ==========================================
try:
    # LLM의 텍스트 응답을 파이썬 딕셔너리로 변환
    graph_data = json.loads(response.content)
    print("\n✅ [성공] 추출된 JSON 데이터:")
    print(json.dumps(graph_data, indent=2, ensure_ascii=False))

    # NetworkX 그래프 생성
    G = nx.DiGraph()

    # 노드(엔티티) 추가
    for entity in graph_data.get("entities", []):
        G.add_node(entity["id"], type=entity["type"])

    # 에지(관계) 추가
    for rel in graph_data.get("relations", []):
        G.add_edge(rel["source"], rel["target"], label=rel["type"])

    # WSL2 환경에서는 GUI 창이 안 뜰 수 있으므로 png 파일로 저장
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # 노드 배치 알고리즘
    
    # 노드와 에지 그리기
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Extracted Micro-GraphRAG (Track A)")
    plt.savefig("micro_graphrag_result.png") # 이미지로 저장
    print("\n✅ [성공] 그래프 시각화가 'micro_graphrag_result.png' 파일로 저장되었습니다!")

except json.JSONDecodeError:
    print("\n❌ [실패] 모델이 올바른 JSON을 생성하지 않았습니다. 응답 원본:")
    print(response.content)
except Exception as e:
    print(f"\n❌ [오류 발생]: {e}")