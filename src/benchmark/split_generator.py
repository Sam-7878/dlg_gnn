import numpy as np
import logging

logger = logging.getLogger(__name__)

class SplitGenerator:
    """
    Generates data split indexes (train, validation, test) for evaluating the models.
    Supports random, temporal (causal), and inductive splits.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_random_split(self, num_nodes: int, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """Randomly splits node indices."""
        indices = np.random.permutation(num_nodes)
        n_train = int(num_nodes * train_ratio)
        n_val = int(num_nodes * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    def generate_temporal_split(self, timestamps: list, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """
        Splits indices chronologically (temporal split).
        Crucial for causality in streaming evaluation.
        """
        # Sort indices by timestamp
        sorted_indices = np.argsort(timestamps)
        num_nodes = len(timestamps)
        
        n_train = int(num_nodes * train_ratio)
        n_val = int(num_nodes * val_ratio)
        
        train_idx = sorted_indices[:n_train]
        val_idx = sorted_indices[n_train:n_train+n_val]
        test_idx = sorted_indices[n_train+n_val:]
        
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    def generate_inductive_split(self, num_nodes: int, unseen_ratio=0.2):
        """
        Splits nodes into seen (train/val) and unseen (test) components
        to evaluate inductive GNN properties.
        """
        indices = np.random.permutation(num_nodes)
        n_test = int(num_nodes * unseen_ratio)
        n_train = int((num_nodes - n_test) * 0.8)
        
        test_idx = indices[:n_test]
        train_idx = indices[n_test:n_test+n_train]
        val_idx = indices[n_test+n_train:]
        
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
