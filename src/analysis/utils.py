# -*- coding: utf-8 -*-
"""
DLG-GNN Benchmark Analysis Utilities
"""

DOMAIN_MAP = {
    "Elliptic": "Financial/Blockchain",
    "DGraphFin": "Financial/Blockchain",
    "Yelp": "Financial/Review Fraud",
    "Amazon": "Financial/Review Fraud",
    "BitcoinOTC": "Blockchain/Trust",
    "Flickr": "Social Network",
    "Reddit": "Social Network",
    "Cora": "Citation",
    "CiteSeer": "Citation",
    "PubMed": "Citation"
}

DOMAIN_GROUP_MAP = {
    "Elliptic": "Financial & Blockchain Fraud",
    "DGraphFin": "Financial & Blockchain Fraud",
    "Yelp": "Financial & Blockchain Fraud",
    "Amazon": "Financial & Blockchain Fraud",
    "BitcoinOTC": "Financial & Blockchain Fraud",
    "Flickr": "Social/Sybil",
    "Reddit": "Social/Sybil",
    "Cora": "Citation/Homophilous",
    "CiteSeer": "Citation/Homophilous",
    "PubMed": "Citation/Homophilous"
}

MODEL_FAMILY_MAP = {
    "DOMINANT": "Reconstruction",
    "AnomalyDAE": "Reconstruction",
    "CoLA": "Contrastive",
    "CONAD": "Contrastive/Augmented",
    "GADNR": "Neighborhood Reconstruction",
    "OCGNN": "One-Class",
    "DLG-Base": "Decoupled",
    "DLG": "Decoupled"
}

def normalize_columns(df):
    """Normalize input dataframe column names to uniform strings."""
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['dataset', 'data']:
            rename_dict[col] = 'Dataset'
        elif col_lower in ['model']:
            rename_dict[col] = 'Model'
        elif col_lower in ['roc-auc', 'roc_auc', 'auc']:
            rename_dict[col] = 'ROC-AUC'
        elif col_lower in ['pr-auc', 'pr_auc', 'ap']:
            rename_dict[col] = 'PR-AUC'
        elif col_lower in ['f1-score', 'f1_score', 'f1', 'best-f1', 'best_f1']:
            rename_dict[col] = 'F1-Score'
        elif col_lower in ['time', 'time (s)', 'time_s', 'runtime']:
            rename_dict[col] = 'Time (s)'
        elif col_lower in ['peak ram (mb)', 'peak_ram_mb', 'ram', 'peak ram']:
            rename_dict[col] = 'Peak RAM (MB)'
        elif col_lower in ['peak vram (mb)', 'peak_vram_mb', 'vram', 'peak vram']:
            rename_dict[col] = 'Peak VRAM (MB)'
        elif col_lower in ['nodes', 'num_nodes']:
            rename_dict[col] = 'Nodes'
    return df.rename(columns=rename_dict)
