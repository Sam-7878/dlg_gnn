import os
import json
import logging
from micro_rag.llm_extractor import GraphExtractor
from micro_rag.graph_builder import GraphManager

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 1. Define Dummy Yelp Reviews (mix of spam and normal)
DUMMY_REVIEWS = [
    {
        "review_id": "rev_001",
        "user_id": "User_A",
        "text": "User_A: This XYZ product did not work at all. But I went to http://scam-link.com/rewards and made $10,000 in two days! Click here to check it out."
    },
    {
        "review_id": "rev_002",
        "user_id": "User_B",
        "text": "User_B: Total waste of money. However, if you want real income, try this amazing tool at http://scam-link.com/rewards. It guarantees passive income."
    },
    {
        "review_id": "rev_003",
        "user_id": "User_C",
        "text": "User_C: Register now to get a free crypto signup bonus of 5000 USDT! Fully verified and guaranteed. Signup here: http://free-bonus-usdt.io. This is a limited time promotion!"
    },
    {
        "review_id": "rev_004",
        "user_id": "User_D",
        "text": "User_D: I purchased the ABC coffee maker. It makes decent espresso but the cleanup is a bit tedious. Overall, 4 stars."
    },
    {
        "review_id": "rev_005",
        "user_id": "User_E",
        "text": "User_E: The ABC coffee maker is excellent! Best espresso I've had at home. Highly recommended to everyone!"
    }
]

def main():
    logger.info("Starting Micro-GraphRAG Phase 1 Pipeline")

    # Initialize modules
    extractor = GraphExtractor(model_name="phi3.5", temperature=0.0)
    manager = GraphManager()

    # Process each review
    for item in DUMMY_REVIEWS:
        review_text = item["text"]
        user_id = item["user_id"]
        logger.info(f"Processing review by {user_id}...")

        # Extract graph structure via LLM
        graph_data = extractor.extract_from_text(review_text)
        
        # Pretty print the extracted JSON
        print(f"\n--- Extracted JSON for {user_id} ---")
        print(json.dumps(graph_data, indent=2, ensure_ascii=False))
        print("------------------------------------\n")

        # Merge into global graph
        manager.add_review_graph(graph_data)

    # Save visualization to the work reports directory
    output_dir = "dlg_gnn/docs/work_reports/40-microRAG"
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "micro_graphrag_result.png")
    
    logger.info("Generating global graph visualization...")
    manager.save_visualization(img_path)

    # Calculate R_local score for each USER in the graph
    users = [node for node in manager.G.nodes if manager.G.nodes[node].get("type") == "USER"]
    
    print("\n" + "="*50)
    print("       MICRO-GRAPHRAG USER RISK SCORE TABLE")
    print("="*50)
    print(f"| {'User ID':<15} | {'R_local Score':<15} | {'Status':<12} |")
    print("-" * 50)
    
    for user in sorted(users):
        r_local = manager.calculate_r_local(user)
        if r_local >= 0.8:
            status = "🔴 SUSPICIOUS"
        elif r_local >= 0.4:
            status = "🟡 WARNING"
        else:
            status = "🟢 NORMAL"
            
        print(f"| {user:<15} | {r_local:<15.2f} | {status:<12} |")
        
    print("="*50)
    print(f"Total nodes in merged graph: {len(manager.G.nodes)}")
    print(f"Total edges in merged graph: {len(manager.G.edges)}\n")

if __name__ == "__main__":
    main()
