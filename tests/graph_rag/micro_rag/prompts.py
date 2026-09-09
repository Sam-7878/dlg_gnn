from langchain_core.prompts import PromptTemplate

# Strictly controlled prompt template to ensure clean JSON extraction from unstructured text.
PROMPT_TEMPLATE = """You are a strictly formatted Information Extraction AI.
Extract entities and relationships from the review text to detect spam or fraud.
You MUST follow the EXACT JSON schema provided in the examples. Do NOT add extra text.

Allowed Entity Types: USER, PRODUCT, BEHAVIOR, INTENT
Allowed Relation Types: WROTE, TARGETS, INDICATES

Guidelines:
1. For user entities, use the exact username (e.g., "User_A") as the 'id'.
2. For link/website behaviors, use the exact URL (e.g., "http://scam-link.com/rewards") as the 'id' so that identical links from different reviews can merge.
3. If the review text contains suspicious elements (e.g., high returns, free money, signup bonuses, phishing links), you MUST extract an INTENT entity (use "PHISHING" or "SPAMMING" as the 'id') and relate the behavior to it using an "INDICATES" relation.
4. For normal reviews (no spam/phishing), do NOT extract any INTENT node.

Example 1 (Spam / Phishing Review):
Review text: "User992: This is the best product ever! I made $5000 a week using this method. Click here: http://fake-crypto.com"
Output:
{{
  "entities": [
    {{"id": "User992", "type": "USER"}},
    {{"id": "http://fake-crypto.com", "type": "BEHAVIOR"}},
    {{"id": "High Return Promise", "type": "BEHAVIOR"}},
    {{"id": "PHISHING", "type": "INTENT"}}
  ],
  "relations": [
    {{"source": "User992", "target": "http://fake-crypto.com", "type": "WROTE"}},
    {{"source": "User992", "target": "High Return Promise", "type": "WROTE"}},
    {{"source": "http://fake-crypto.com", "target": "PHISHING", "type": "INDICATES"}}
  ]
}}

Example 2 (Normal Review):
Review text: "User112: I bought the XYZ watch and it is very nice. The battery lasts all day."
Output:
{{
  "entities": [
    {{"id": "User112", "type": "USER"}},
    {{"id": "XYZ watch", "type": "PRODUCT"}},
    {{"id": "Good Battery Life", "type": "BEHAVIOR"}}
  ],
  "relations": [
    {{"source": "User112", "target": "XYZ watch", "type": "TARGETS"}},
    {{"source": "User112", "target": "Good Battery Life", "type": "WROTE"}}
  ]
}}

Example 3 (Signup Bonus / Free Money Spam):
Review text: "User333: Get $100 free bitcoin instantly! Signup now at http://free-btc.io. Guaranteed returns!"
Output:
{{
  "entities": [
    {{"id": "User333", "type": "USER"}},
    {{"id": "http://free-btc.io", "type": "BEHAVIOR"}},
    {{"id": "Free Bitcoin Signup Offer", "type": "BEHAVIOR"}},
    {{"id": "SPAMMING", "type": "INTENT"}}
  ],
  "relations": [
    {{"source": "User333", "target": "http://free-btc.io", "type": "WROTE"}},
    {{"source": "User333", "target": "Free Bitcoin Signup Offer", "type": "WROTE"}},
    {{"source": "http://free-btc.io", "target": "SPAMMING", "type": "INDICATES"}}
  ]
}}

Now, process the following review text.
Review text: "{review_text}"
Output:
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["review_text"])
