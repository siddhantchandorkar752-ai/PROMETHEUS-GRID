import numpy as np
import os
import sys

# Path logic for safety
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.logger import get_logger
logger = get_logger()

class PrometheusEnv:
    def __init__(self, nodes=10):
        self.nodes = nodes
        self.time = 0 
        logger.info(f"Environment Initialized with {self.nodes} Nodes.")

    def get_city_state(self):
        hour = self.time % 24
        solar = round(max(0, np.sin((hour - 6) * np.pi / 12)) * 100, 2)
        demand = round(50 + 20 * np.sin((hour - 7) * np.pi / 6), 2)
        return {"hour": hour, "solar": solar, "demand": demand, "net_load": round(demand - solar, 2)}

    def step(self):
        state = self.get_city_state()
        logger.info(f"Hour {state['hour']:02d}:00 | Solar: {state['solar']}MW | Demand: {state['demand']}MW")
        self.time += 1
        return state
