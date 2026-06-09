import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MCStreamingInference:
    """
    Handles Monte Carlo (MC) Dropout stochastic inference for streaming DLG-GNN models.
    """
    def __init__(self, model, num_samples: int = 10, dropout_rate: float = 0.2):
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

    def enable_dropout(self, m):
        """Forces dropout layers to remain active during evaluation."""
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            m.train()

    def predict_with_uncertainty(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs T forward passes with active dropout to compute:
        - Mean Probability (P_fraud)
        - Variance / Uncertainty (U_mc)
        - Predictive Entropy (H)
        
        Returns:
            mean_prob: torch.Tensor [num_nodes]
            variance: torch.Tensor [num_nodes]
            entropy: torch.Tensor [num_nodes]
        """
        # 1. Put model in evaluation mode, then force enable dropout
        torch_model = self.model
        if hasattr(self.model, 'model'):
            torch_model = self.model.model
            
        torch_model.eval()
        if self.dropout_rate > 0.0:
            torch_model.apply(self.enable_dropout)
            
        num_nodes = data.x.size(0)
        all_probs = []

        logger.info(f"Running {self.num_samples} MC Dropout passes for streaming event...")
        
        with torch.no_grad():
            for i in range(self.num_samples):
                # Check model type to resolve appropriate forward method
                if hasattr(self.model, 'decision_function'):
                    # PyGOD detector structure
                    scores = self.model.decision_function(data)
                elif hasattr(self.model, 'predict_with_uncertainty') and hasattr(self.model, 'model'):
                    # Custom DLGFull model
                    scores = self.model.model(data.x.to(self.model.device), data.edge_index.to(self.model.device))
                    # If scores returns multiple, resolve to reconstruction loss
                    if isinstance(scores, tuple):
                        x_rec, s_rec = scores
                        # Reconstruction error per node
                        scores = torch.sum((data.x.to(self.model.device) - x_rec)**2, dim=1)
                else:
                    # Generic PyTorch model fallback
                    scores = self.model(data.x, data.edge_index)
                    if isinstance(scores, tuple):
                        scores = scores[0]
                    if scores.dim() > 1:
                        scores = torch.softmax(scores, dim=-1)[:, 1]

                # Ensure tensor is CPU float
                scores = scores.cpu().float()
                # Min-max scale or sigmoid to map to [0, 1] probability range
                # Use sigmoid response function for standard anomaly scores
                if scores.max() > 1.0 or scores.min() < 0.0:
                    probs = torch.sigmoid((scores - scores.mean()) / (scores.std() + 1e-6))
                else:
                    probs = scores
                    
                all_probs.append(probs)

        # Stack predictions: [T, num_nodes]
        stacked_probs = torch.stack(all_probs, dim=0)
        
        # Calculate mean, variance, and entropy
        mean_prob = torch.mean(stacked_probs, dim=0)
        variance = torch.var(stacked_probs, dim=0)
        
        # Avoid log(0) in entropy
        eps = 1e-8
        entropy = -mean_prob * torch.log(mean_prob + eps) - (1.0 - mean_prob) * torch.log(1.0 - mean_prob + eps)
        
        # Restore evaluation mode
        torch_model.eval()
        
        return mean_prob, variance, entropy
