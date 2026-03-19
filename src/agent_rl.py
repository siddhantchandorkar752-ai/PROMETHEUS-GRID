import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Root Path Injection
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.logger import get_logger

logger = get_logger()

class PrometheusAgent(nn.Module):
    def __init__(self, state_dim=16, action_dim=3):
        """
        state_dim: GNN se aane wale embeddings (16)
        action_dim: [0: Store, 1: Release, 2: Do Nothing]
        """
        super(PrometheusAgent, self).__init__()
        logger.info(f"Building RL Agent: State({state_dim}) -> Actions({action_dim})")
        
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output_layer(x)

    def select_action(self, state, epsilon=0.1):
        # Epsilon-greedy: Exploration vs Exploitation
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, 3, (1,)).item()
        
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values).item()

if __name__ == "__main__":
    try:
        # Test: GNN ke output (10 nodes, 16 features) ko process karna
        test_state = torch.randn(1, 16) 
        agent = PrometheusAgent()
        action = agent.select_action(test_state)
        
        action_map = {0: "STORE", 1: "RELEASE", 2: "IDLE"}
        logger.success(f"RL Agent Decided Action: {action_map[action]}")
    except Exception as e:
        logger.error(f"RL Agent Test Failed: {e}")
