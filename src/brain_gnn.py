import os
import sys

# ROOT PATH INJECTION (Flawless Method)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils.logger import get_logger

logger = get_logger()

class PrometheusGNN(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=16):
        super(PrometheusGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        logger.info("GNN Neural Architecture Loaded.")
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

if __name__ == "__main__":
    try:
        test_x = torch.randn(10, 4) 
        test_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        model = PrometheusGNN()
        output = model(test_x, test_edge_index)
        logger.success(f"GNN Verified! Output Shape: {output.shape}")
    except Exception as e:
        logger.error(f"GNN Test Failed: {e}")
