import os
import sys
import torch
from loguru import logger

# Root Path Injection
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.environment import PrometheusEnv
from src.brain_gnn import PrometheusGNN
from src.agent_rl import PrometheusAgent
from utils.visualization import plot_grid_state

def run_production_cycle():
    logger.info("🚀 PROMETHEUS-GRID: LIVE DASHBOARD CYCLE STARTING")
    
    env = PrometheusEnv(nodes=10)
    gnn = PrometheusGNN(input_dim=4, output_dim=16)
    agent = PrometheusAgent(state_dim=16, action_dim=3)
    
    edge_index = torch.tensor([[i, (i+1)%10] for i in range(10)] + [[(i+1)%10, i] for i in range(10)], dtype=torch.long).t()

    for hour in range(24):
        state = env.step()
        
        # 1. Visualization Frame Save Karo
        plot_grid_state(state['hour'], state['solar'], state['demand'], nodes=10)
        
        # 2. AI Decision Logic
        node_features = torch.tensor([[state['hour'], state['solar'], state['demand'], state['net_load']]] * 10, dtype=torch.float)
        with torch.no_grad():
            node_embeddings = gnn(node_features, edge_index)
            city_embedding = node_embeddings.mean(dim=0).unsqueeze(0)
            action_idx = agent.select_action(city_embedding)
            
        action_map = {0: "CHARGE BATTERIES", 1: "RELEASE ENERGY", 2: "STABILIZE GRID"}
        logger.success(f"HOUR {state['hour']:02d}:00 | AI ACTION: {action_map[action_idx]}")

    logger.success("✅ Full 24-Hour Dashboard Frames Generated in 'data/' folder.")

if __name__ == "__main__":
    run_production_cycle()
