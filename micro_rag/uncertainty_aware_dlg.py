import torch
import torch.nn as nn
import logging
from gog_fraud.models.pygod.dlg_full import DLGFull

logger = logging.getLogger(__name__)

class UncertaintyAwareDLG(DLGFull):
    """
    Uncertainty-Aware Decoupled Local-to-Global (DLG-Full) Detector.
    Extends DLGFull to support Monte Carlo (MC) Dropout uncertainty estimation
    and fusion with external R_local risk scores.
    """
    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 l1_hops=2,
                 l1_epochs=20,
                 l1_hid_dim=64,
                 dropout=0.2,   # Non-zero dropout is required for MC Dropout variation
                 weight_decay=0.,
                 gpu=-1,
                 batch_size=0,
                 weight=0.5,
                 verbose=0,
                 **kwargs):
        super().__init__(
            hid_dim=hid_dim,
            num_layers=num_layers,
            l1_hops=l1_hops,
            l1_epochs=l1_epochs,
            l1_hid_dim=l1_hid_dim,
            dropout=dropout,
            weight_decay=weight_decay,
            gpu=gpu,
            batch_size=batch_size,
            weight=weight,
            verbose=verbose,
            **kwargs
        )
        logger.info("Initialized UncertaintyAwareDLG with active dropout layer.")

    def predict_with_uncertainty(self, data, num_samples: int = 10):
        """
        Runs multiple forward passes with active dropout (MC Dropout) to calculate
        the mean reconstruction score (P_fraud) and its variance (U_mc) for each node.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Input PyG graph data.
        num_samples : int, optional
            Number of MC Dropout forward passes. Default: ``10``.
            
        Returns
        -------
        p_fraud : torch.Tensor
            Mean anomaly prediction score [num_nodes].
        u_mc : torch.Tensor
            Prediction uncertainty (variance) [num_nodes].
        """
        # Ensure the graph has been augmented with L1 embeddings
        self.process_graph(data)
        
        # Set the model to training mode to activate dropout during inference
        self.model.train()
        
        # Resolve batching targets for full-batch fallback
        batch_size = getattr(data, 'batch_size', None)
        if batch_size is None or batch_size == 0:
            batch_size = data.num_nodes
            
        node_idx = getattr(data, 'n_id', None)
        if node_idx is None:
            node_idx = torch.arange(data.num_nodes, device=self.device)
            
        x = data.x.to(self.device)
        x_orig = x[:, :self._orig_dim]
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        logger.info(f"Running {num_samples} MC Dropout passes for uncertainty estimation...")
        all_scores = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Forward pass under model.train() so dropout is random
                x_rec, s_rec = self.model(x, edge_index)
                
                # Compute per-node anomaly score
                score = self.model.loss_func(
                    x_orig[:batch_size],
                    x_rec[:batch_size],
                    s[:batch_size, node_idx],
                    s_rec[:batch_size],
                    self.weight
                )
                all_scores.append(score.cpu())
                
        # Stack results: shape [num_samples, batch_size]
        stacked_scores = torch.stack(all_scores, dim=0)
        
        # Calculate mean prediction (P_fraud) and variance (U_mc)
        p_fraud = torch.mean(stacked_scores, dim=0)
        u_mc = torch.var(stacked_scores, dim=0)
        
        # Restore evaluation mode
        self.model.eval()
        
        return p_fraud, u_mc

    def fusion_layer(self, p_fraud, u_mc, r_local_tensor, gamma: float = 1.0, u_max: float = None):
        """
        Applies Uncertainty-Aware Dynamic Weighting Fusion to combine GNN prediction (P_fraud)
        and local textual fraud score (R_local) based on model uncertainty (U_mc).
        
        Formula:
        S_final = (1 - alpha) * P_fraud + alpha * R_local
        where alpha = min(1.0, gamma * (U_mc / U_max))
        
        Parameters
        ----------
        p_fraud : torch.Tensor
            GNN anomaly prediction score [num_nodes].
        u_mc : torch.Tensor
            Prediction uncertainty [num_nodes].
        r_local_tensor : torch.Tensor
            Injected textual risk score [num_nodes].
        gamma : float, optional
            Sensitivity tuning parameter. Default: ``1.0``.
        u_max : float, optional
            Maximum uncertainty normalization factor. Defaults to max(u_mc).
            
        Returns
        -------
        s_final : torch.Tensor
            Final fused risk score [num_nodes].
        alpha : torch.Tensor
            Dynamic weighting factor alpha [num_nodes].
        """
        # Resolve U_max
        if u_max is None or u_max == 0.0:
            u_max = u_mc.max().item()
            if u_max == 0.0:
                u_max = 1.0
                
        # Compute alpha: min(1.0, gamma * (U_mc / U_max))
        alpha = torch.clamp(gamma * (u_mc / u_max), max=1.0)
        
        # Calculate S_final
        s_final = (1 - alpha) * p_fraud + alpha * r_local_tensor
        
        return s_final, alpha
