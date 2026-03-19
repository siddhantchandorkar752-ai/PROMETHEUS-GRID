import networkx as nx
import matplotlib.pyplot as plt
from utils.logger import get_logger

logger = get_logger()

def plot_grid_state(hour, solar, demand, nodes=10):
    G = nx.cycle_graph(nodes)
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(8, 6))
    
    # Logic: Agar solar > demand toh node Green, nahi toh Red
    color_map = []
    for i in range(nodes):
        if solar > demand:
            color_map.append('green')
        else:
            color_map.append('red')
            
    nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=800, edge_color='gray')
    
    plt.title(f"PROMETHEUS-GRID | Hour: {hour:02d}:00 | Solar: {solar}MW | Demand: {demand}MW")
    
    # Save frame for 2030 Portfolio
    plt.savefig(f"data/frame_{hour:02d}.png")
    plt.close()
    logger.info(f"Frame saved for Hour {hour:02d}")

if __name__ == "__main__":
    # Test Visualization
    plot_grid_state(12, 95.0, 45.0)
    logger.success("Visualization Engine Verified! Check 'data/' folder.")
