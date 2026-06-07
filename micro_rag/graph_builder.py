import networkx as nx
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class GraphManager:
    """
    Manages the global NetworkX DiGraph by merging individual review graphs
    and calculating local risk scores (R_local) for USER nodes.
    """
    def __init__(self):
        self.G = nx.DiGraph()

    def add_review_graph(self, graph_data: dict):
        """
        Merges new entities and relations into the global directed graph.
        Entities with the same ID are automatically merged.
        """
        # Add nodes with their type attributes
        for entity in graph_data.get("entities", []):
            node_id = entity.get("id")
            node_type = entity.get("type")
            if not node_id:
                continue
            
            # If node already exists, merge attributes
            if self.G.has_node(node_id):
                # Update attributes if needed
                self.G.nodes[node_id]["type"] = node_type
            else:
                self.G.add_node(node_id, type=node_type)

        # Add edges with relationship types
        for rel in graph_data.get("relations", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            if not source or not target:
                continue
            
            # Add edge. If it exists, update type/label
            self.G.add_edge(source, target, label=rel_type)
            
            # Fallback type check for source and target if they were not declared in entities
            if "type" not in self.G.nodes[source]:
                self.G.nodes[source]["type"] = "USER" if "USER" in str(source).upper() else "UNKNOWN"
            if "type" not in self.G.nodes[target]:
                if "PHISHING" in str(target).upper() or "SPAMMING" in str(target).upper():
                    self.G.nodes[target]["type"] = "INTENT"
                elif "LINK" in str(target).upper() or "URL" in str(target).upper():
                    self.G.nodes[target]["type"] = "BEHAVIOR"
                else:
                    self.G.nodes[target]["type"] = "UNKNOWN"

    def is_intent_node(self, node_id: str) -> bool:
        """
        Determines whether a node is an INTENT node (e.g. PHISHING, SPAMMING).
        """
        node_attr = self.G.nodes[node_id]
        node_type = str(node_attr.get("type", "")).upper()
        node_name = str(node_id).upper()
        return node_type == "INTENT" or node_name in ("PHISHING", "SPAMMING", "INTENT")

    def calculate_r_local(self, user_id: str) -> float:
        """
        Calculates a fraud risk score (R_local) between 0.0 and 1.0 for a user node.
        
        Scoring Model:
        - 1.0: Direct link to an INTENT node (1-hop directed).
        - 0.8: Connected to an INTENT node via a 2-hop directed path (e.g., USER -> BEHAVIOR -> INTENT).
        - 0.5: Shared behavior node with another user who has a directed path to an INTENT node (fraud ring / sybil indicator).
        - 0.0: No connection to INTENT nodes within 2 hops or shared behavior paths.
        """
        if user_id not in self.G:
            return 0.0

        # Check if the node type is indeed USER
        node_type = str(self.G.nodes[user_id].get("type", "")).upper()
        if node_type != "USER":
            logger.warning(f"calculate_r_local called on non-user node: {user_id} (type: {node_type})")

        score = 0.0

        # 1. Check direct link (1-hop directed)
        for successor in self.G.successors(user_id):
            if self.is_intent_node(successor):
                score = max(score, 1.0)

        # 2. Check directed 2-hop path (USER -> BEHAVIOR/PRODUCT -> INTENT)
        for successor in self.G.successors(user_id):
            for grand_successor in self.G.successors(successor):
                if self.is_intent_node(grand_successor):
                    score = max(score, 0.8)

        # 3. Check shared behavior node (USER_A -> BEHAVIOR <- USER_B -> INTENT)
        # Find behaviors written by this user
        for behavior_node in self.G.successors(user_id):
            behavior_type = str(self.G.nodes[behavior_node].get("type", "")).upper()
            if behavior_type == "BEHAVIOR":
                # Find other users writing the same behavior
                for other_user in self.G.predecessors(behavior_node):
                    if other_user == user_id:
                        continue
                    
                    # Check if the other user has a directed link or 2-hop path to an INTENT node
                    other_has_intent = False
                    for out in self.G.successors(other_user):
                        if self.is_intent_node(out):
                            other_has_intent = True
                            break
                        for out_2 in self.G.successors(out):
                            if self.is_intent_node(out_2):
                                other_has_intent = True
                                break
                    
                    if other_has_intent:
                        score = max(score, 0.5)

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
        
        # Determine node colors based on type
        # USER: lightblue, BEHAVIOR: orange, PRODUCT: lightgreen, INTENT: salmon, OTHER: gray
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
