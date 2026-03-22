"""
Prometheus Graph Neural Network Module.

This module provides a robust, highly extensible, and production-ready
implementation of a Graph Convolutional Network (GNN).
"""
import os
import sys
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import Tensor

# ROOT PATH INJECTION (Flawless Method)
_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

from utils.logger import get_logger

logger = get_logger()


def seed_everything(seed: int = 42) -> None:
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    
    Args:
        seed (int): The arbitrary seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class PrometheusConfig:
    """
    Configuration dataclass for the PrometheusGNN.
    Centralizing hyperparameters improves code maintainability and tracking.
    """
    input_dim: int = 4
    hidden_dim: int = 32
    output_dim: int = 16
    dropout_rate: float = 0.5
    use_batch_norm: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PrometheusGNN(nn.Module):
    """
    Prometheus Core Graph Neural Network.
    
    A sophisticated 2-layer GCN utilizing optional Batch Normalization
    and Dropout for enhanced training stability and generalization.

    Args:
        config (PrometheusConfig, optional): Model hyperparameters.
    """

    def __init__(self, config: Optional[PrometheusConfig] = None) -> None:
        super().__init__()
        self.config = config or PrometheusConfig()

        # Input to Hidden Layer Convolution
        self.conv1 = GCNConv(self.config.input_dim, self.config.hidden_dim)
        
        # Batch Normalization layer mitigates internal covariate shift
        self.bn1 = (
            nn.BatchNorm1d(self.config.hidden_dim) 
            if self.config.use_batch_norm 
            else nn.Identity()
        )
        
        # Hidden to Output Layer Convolution
        self.conv2 = GCNConv(self.config.hidden_dim, self.config.output_dim)

        # Send model to configured device internally
        self.to(self.config.device)
        logger.info(f"PrometheusGNN initialized efficiently. Hardware: {self.config.device.upper()}")

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Executes the forward pass of the model.

        Args:
            x (Tensor): Node feature matrix of shape (num_nodes, input_dim).
            edge_index (Tensor): Graph connectivity matrix of shape (2, num_edges).

        Returns:
            Tensor: Output node embeddings of shape (num_nodes, output_dim).
            
        Raises:
            ValueError: If the feature dimension of `x` is incompatible with `input_dim`.
        """
        if x.size(1) != self.config.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: Expected {self.config.input_dim}, "
                f"but received {x.size(1)}."
            )

        # 1st Layer: Graph Convolution -> Batch Norm -> ReLU -> Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.config.dropout_rate, training=self.training)
        
        # 2nd Layer: Graph Convolution
        # No activation/dropout is applied here to maintain raw logits or embeddings
        x = self.conv2(x, edge_index)
        
        return x


def run_diagnostics() -> None:
    """
    Validates the structural integrity and operational status of the model
    via a rigorous forward-pass inference sanity check.
    """
    try:
        seed_everything(seed=42)
        logger.info("Initializing diagnostic environment for PrometheusGNN...")
        
        config = PrometheusConfig()
        model = PrometheusGNN(config)
        
        # Instantiate synthetic test data simulating 'num_nodes' interconnected identities
        num_nodes = 10
        test_x = torch.randn(num_nodes, config.input_dim, device=config.device)
        
        # Create a simple connected graph logic (source -> target edges)
        test_edge_index = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], 
            dtype=torch.long, 
            device=config.device
        )
        
        # Transitioning to evaluation phase (disables dropout, solidifies batch metrics)
        model.eval()
        
        # Scoped diagnostic forward pass omitting computational graph tracking memory overhead
        with torch.no_grad():
            output = model(test_x, test_edge_index)
            
        # Assert rigorous type-checking bounds
        assert output.shape == (num_nodes, config.output_dim), \
            f"Shape validation failed. Extracted: {output.shape}"
            
        logger.success(f"GNN Validation Complete! Final Output Shape: {tuple(output.shape)}")
        
    except Exception as e:
        logger.error(f"GNN Validation Phase Terminated With Errors: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_diagnostics()
