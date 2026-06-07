import networkx as nx
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class GraphManager:
    """
    Manages the global NetworkX DiGraph by merging individual review graphs
    and calculating local risk scores (R_local) for USER nodes.
    Supports multi-domain tracking and domain-specific intent weighting.
    """
    def __init__(self):
        self.G = nx.DiGraph()

    def add_review_graph(self, graph_data: dict, domain: str = None):
        """
        Merges new entities and relations into the global directed graph.
        Entities with the same ID are automatically merged, and domains are tracked.
        """
        domain_val = domain.lower() if domain else None

        # Add nodes with their type attributes
        for entity in graph_data.get("entities", []):
            node_id = entity.get("id")
            node_type = entity.get("type")
            if not node_id:
                continue
            
            # If node already exists, merge attributes and track domains
            if self.G.has_node(node_id):
                self.G.nodes[node_id]["type"] = node_type
                if "domains" not in self.G.nodes[node_id]:
                    self.G.nodes[node_id]["domains"] = set()
                if domain_val:
                    self.G.nodes[node_id]["domains"].add(domain_val)
            else:
                domains = {domain_val} if domain_val else set()
                self.G.add_node(node_id, type=node_type, domains=domains)

        # Add edges with relationship types
        for rel in graph_data.get("relations", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            if not source or not target:
                continue
            
            # Add edge. If it exists, update type/label
            self.G.add_edge(source, target, label=rel_type)
            if domain_val:
                self.G.edges[source, target]["domain"] = domain_val
            
            # Fallback type check for source and target if they were not declared in entities
            if "type" not in self.G.nodes[source]:
                self.G.nodes[source]["type"] = "USER" if "USER" in str(source).upper() else "UNKNOWN"
                self.G.nodes[source]["domains"] = {domain_val} if domain_val else set()
            elif "domains" not in self.G.nodes[source]:
                self.G.nodes[source]["domains"] = {domain_val} if domain_val else set()
            elif domain_val:
                self.G.nodes[source]["domains"].add(domain_val)
                
            if "type" not in self.G.nodes[target]:
                target_upper = str(target).upper()
                if any(x in target_upper for x in ["PHISHING", "SPAMMING", "INTENT", "PUMP", "MANIPULATION", "SCAM"]):
                    self.G.nodes[target]["type"] = "INTENT"
                elif any(x in target_upper for x in ["LINK", "URL", "WEBSITE"]):
                    self.G.nodes[target]["type"] = "BEHAVIOR"
                else:
                    self.G.nodes[target]["type"] = "UNKNOWN"
                self.G.nodes[target]["domains"] = {domain_val} if domain_val else set()
            elif "domains" not in self.G.nodes[target]:
                self.G.nodes[target]["domains"] = {domain_val} if domain_val else set()
            elif domain_val:
                self.G.nodes[target]["domains"].add(domain_val)

    def is_intent_node(self, node_id: str) -> bool:
        """
        Determines whether a node is an INTENT node (e.g. PHISHING, SPAMMING, PUMP_AND_DUMP).
        """
        node_attr = self.G.nodes[node_id]
        node_type = str(node_attr.get("type", "")).upper()
        node_name = str(node_id).upper()
        
        intent_keywords = ["PHISHING", "SPAMMING", "INTENT", "PUMP", "MANIPULATION", "SCAM"]
        return node_type == "INTENT" or any(k in node_name for k in intent_keywords)

    def get_intent_weight(self, intent_node: str, domain: str) -> float:
        """
        Returns the domain-specific weight of an INTENT node.
        - Yelp: Reputation manipulation risk priority.
        - Amazon: Phishing URL propagation risk priority.
        - Reddit: Organized market manipulation risk priority.
        """
        domain_key = (domain or "yelp").lower()
        intent = str(intent_node).upper()

        weights = {
            "yelp": {
                "SPAMMING": 1.0, "FAKE_REVIEW": 1.0, "COMPLAINT": 0.8,
                "PHISHING": 0.6, "SCAM_LINK": 0.6, "SCAM": 0.6,
                "PUMP_AND_DUMP": 0.4, "MARKET_MANIPULATION": 0.4, "SHILLING": 0.5
            },
            "amazon": {
                "PHISHING": 1.0, "SCAM_LINK": 1.0, "SCAM": 1.0,
                "SPAMMING": 0.7, "FAKE_REVIEW": 0.8, "COMPLAINT": 0.6,
                "PUMP_AND_DUMP": 0.3, "MARKET_MANIPULATION": 0.3, "SHILLING": 0.4
            },
            "reddit": {
                "PUMP_AND_DUMP": 1.0, "MARKET_MANIPULATION": 1.0, "SHILLING": 0.9,
                "SPAMMING": 0.6, "FAKE_REVIEW": 0.5, "COMPLAINT": 0.4,
                "PHISHING": 0.5, "SCAM_LINK": 0.5, "SCAM": 0.5
            }
        }

        domain_weights = weights.get(domain_key, weights["yelp"])

        # Check for matching substring
        for key, val in domain_weights.items():
            if key in intent:
                return val
        
        return 0.5  # Default weight fallback

    def calculate_r_local(self, user_id: str, domain: str = "yelp") -> float:
        """
        Calculates a domain-specific fraud risk score (R_local) between 0.0 and 1.0.
        
        Scoring Model:
        - 1-hop link: 1.0 * intent_weight
        - 2-hop path: 0.8 * intent_weight
        - Shared behavior path: 0.5 * intent_weight
        """
        if user_id not in self.G:
            return 0.0

        node_type = str(self.G.nodes[user_id].get("type", "")).upper()
        if node_type != "USER":
            logger.warning(f"calculate_r_local called on non-user node: {user_id} (type: {node_type})")

        score = 0.0

        # 1. Check direct link (1-hop directed: USER -> INTENT)
        for successor in self.G.successors(user_id):
            if self.is_intent_node(successor):
                w = self.get_intent_weight(successor, domain)
                score = max(score, 1.0 * w)

        # 2. Check directed 2-hop path (USER -> BEHAVIOR/PRODUCT -> INTENT)
        for successor in self.G.successors(user_id):
            for grand_successor in self.G.successors(successor):
                if self.is_intent_node(grand_successor):
                    w = self.get_intent_weight(grand_successor, domain)
                    score = max(score, 0.8 * w)

        # 3. Check shared behavior node (USER_A -> BEHAVIOR <- USER_B -> INTENT)
        for behavior_node in self.G.successors(user_id):
            behavior_type = str(self.G.nodes[behavior_node].get("type", "")).upper()
            if behavior_type == "BEHAVIOR":
                for other_user in self.G.predecessors(behavior_node):
                    if other_user == user_id:
                        continue
                    
                    other_has_intent = False
                    max_other_w = 0.0
                    for out in self.G.successors(other_user):
                        if self.is_intent_node(out):
                            other_has_intent = True
                            max_other_w = max(max_other_w, self.get_intent_weight(out, domain))
                        for out_2 in self.G.successors(out):
                            if self.is_intent_node(out_2):
                                other_has_intent = True
                                max_other_w = max(max_other_w, self.get_intent_weight(out_2, domain))
                    
                    if other_has_intent:
                        score = max(score, 0.5 * max_other_w)

        return score

    def save_visualization(self, output_path: str):
        """
        Saves the global graph visualization as a PNG file.
        Color-codes nodes according to their types for a premium visual appearance.
        """
        if len(self.G.nodes) == 0:
            logger.warning("Graph is empty, skipping visualization.")
            return

        plt.figure(figsize=(12, 10))
        
        color_map = []
        for node in self.G.nodes:
            node_type = str(self.G.nodes[node].get("type", "")).upper()
            if node_type == "USER":
                color_map.append("#5dade2")  # Premium Blue
            elif node_type == "BEHAVIOR":
                color_map.append("#f39c12")  # Orange
            elif node_type == "PRODUCT":
                color_map.append("#2ecc71")  # Green
            elif node_type == "INTENT":
                color_map.append("#ec7063")  # Salmon/Red
            else:
                color_map.append("#bdc3c7")  # Silver/Gray

        pos = nx.spring_layout(self.G, k=1.2, seed=42)
        
        # Draw nodes and labels
        nx.draw_networkx_nodes(self.G, pos, node_color=color_map, node_size=1800, alpha=0.9)
        nx.draw_networkx_labels(self.G, pos, font_size=8, font_weight="bold", font_family="sans-serif")
        
        # Draw edges with nice curved styling
        nx.draw_networkx_edges(self.G, pos, edge_color="#7f8c8d", width=1.5, arrowsize=15, min_source_margin=15, min_target_margin=15)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.G, "label")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=7, font_color="#c0392b")

        plt.title("Micro-GraphRAG Global Semantic Knowledge Graph (Track A)", fontsize=14, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Graph visualization successfully saved to {output_path}")
