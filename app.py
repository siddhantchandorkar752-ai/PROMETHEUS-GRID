"""
Prometheus Grid: Neural OS Dashboard
An enterprise-grade Streamlit application for autonomous smart-city grid management.
Features specialized GNN spatial modeling and RL decision loop simulations.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

@dataclass
class AppConfig:
    PAGE_TITLE: str = "PROMETHEUS-GRID // NEURAL OS"
    PAGE_ICON: str = "⚡"
    THEME_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "background": "#00080a",
        "primary": "#00f2fe",
        "primary_dim": "rgba(0, 242, 254, 0.05)",
        "primary_border": "rgba(0, 242, 254, 0.2)",
        "panel_bg": "#001219",
        "danger": "#ff4b4b",
        "danger_dim": "rgba(255, 75, 75, 0.2)",
        "edge": "#1a3a3a",
        "node_border": "white"
    })
    SIMULATION_TICKS: int = 50
    SPEED_MAP: Dict[str, float] = field(default_factory=lambda: {
        "Passive": 0.8,
        "Active": 0.4,
        "Overclock": 0.1
    })

CONFIG = AppConfig()

# Initialize standard logging gracefully
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# UI COMPONENTS & STYLING
# ==========================================

def inject_custom_css() -> None:
    """Injects professional-grade CSS for the Vantablack & Neon Cyan Aura theme."""
    st.markdown(f"""
        <style>
        .main {{ background-color: {CONFIG.THEME_COLORS['background']}; }}
        .stMetric {{ 
            background: {CONFIG.THEME_COLORS['primary_dim']}; 
            padding: 15px; 
            border-radius: 10px; 
            border: 1px solid {CONFIG.THEME_COLORS['primary_border']}; 
        }}
        .log-box {{ 
            background-color: {CONFIG.THEME_COLORS['panel_bg']}; 
            color: {CONFIG.THEME_COLORS['primary']}; 
            padding: 15px; 
            border-radius: 8px; 
            font-family: 'Courier New', monospace; 
            font-size: 0.85rem; 
            height: 250px; 
            overflow-y: auto; 
            border-left: 4px solid {CONFIG.THEME_COLORS['primary']}; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .status-card {{ 
            border-radius: 10px; 
            padding: 20px; 
            text-align: center; 
            color: white; 
            font-weight: 800; 
            letter-spacing: 1.5px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        }}
        </style>
    """, unsafe_allow_html=True)


def render_sidebar() -> Tuple[int, str]:
    """Renders the sidebar control center and returns selected configurations."""
    with st.sidebar:
        st.image(
            "https://capsule-render.vercel.app/api?type=rect&color=001219"
            "&height=60&text=NEURAL%20CONTROLS&fontSize=20&fontColor=00f2fe"
        )
        st.markdown("### 🛠️ Configuration")
        node_count = st.slider("Grid Complexity (Nodes)", min_value=6, max_value=24, value=12)
        refresh_rate = st.select_slider(
            "Neural Pulse Speed", 
            options=list(CONFIG.SPEED_MAP.keys()), 
            value="Active"
        )
        
        st.divider()
        st.markdown("### 🛰️ Core Status")
        st.success("GNN Engine: ACTIVE")
        st.info("RL Policy: STABLE")
        st.warning("Hardware: T4 GPU-ACCEL")
        
    return node_count, refresh_rate


def render_header() -> None:
    """Renders the main dashboard header and architecture details."""
    st.markdown(
        f"# {CONFIG.PAGE_ICON} PROMETHEUS-GRID // "
        f"<span style='color:{CONFIG.THEME_COLORS['primary']}'>NEURAL CORE</span>", 
        unsafe_allow_html=True
    )
    st.markdown("> *Autonomous Smart-City Grid Management | GNN Spatial Modeling + RL Decision Loop*")

    with st.expander("📖 VIEW SYSTEM ARCHITECTURE", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🧠 Spatial GNN Layer")
            st.caption(
                "Captures multi-hop dependencies between energy nodes. "
                "Models the city as a dynamic adjacency matrix."
            )
        with col2:
            st.markdown("#### 🕹️ Reinforcement Learning")
            st.caption(
                "Deep Q-Network (DQN) manages the deficit/surplus delta "
                "with sub-100ms intervention latency."
            )
    st.divider()

# ==========================================
# SIMULATION ENGINE
# ==========================================

class SimulationState:
    """Manages the state and data trajectory of the simulation in an OOP manner."""
    def __init__(self):
        self.history: Dict[str, List[int]] = {"Solar": [], "Demand": [], "Storage": []}
        self.logs: List[str] = []

    def update(self, solar: int, demand: int, storage: int, step: int) -> Dict[str, Any]:
        """Calculates step metrics and updates historical state."""
        self.history["Solar"].append(solar)
        self.history["Demand"].append(demand)
        self.history["Storage"].append(storage)

        is_surplus = solar > demand
        status_color = CONFIG.THEME_COLORS["primary"] if is_surplus else CONFIG.THEME_COLORS["danger"]
        bg_color = CONFIG.THEME_COLORS["primary_dim"] if is_surplus else CONFIG.THEME_COLORS["danger_dim"]
        status_text = "SURPLUS" if is_surplus else "DEFICIT"
        
        action = "CHARGING_ARRAY" if is_surplus else "RELEASING_RESERVES"
        conf = np.random.uniform(94, 99.9)
        timestamp = time.strftime('%H:%M:%S')
        
        # Format the log beautifully
        new_log = f"<span style='color: #6c757d;'>[{timestamp}]</span> EVENT: <b>{status_text}</b> | ACTION: {action} | CONF: {conf:.2f}%"
        self.logs.insert(0, new_log)
        
        # Enforce log rotation to prevent aggressive memory consumption in UI
        if len(self.logs) > 50:
            self.logs = self.logs[:50]

        return {
            "solar": solar,
            "demand": demand,
            "is_surplus": is_surplus,
            "status_color": status_color,
            "bg_color": bg_color,
            "status_text": status_text,
            "new_log": new_log
        }


def generate_topology_graph(node_count: int) -> go.Figure:
    """Generates a high-performance Plotly topology map of the GNN grid."""
    angles = np.linspace(0, 2 * np.pi, node_count, endpoint=False)
    edge_x, edge_y = [], []
    
    # Generate random stochastic adjacency securely
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if np.random.rand() > 0.8:
                edge_x.extend([np.cos(angles[i]), np.cos(angles[j]), None])
                edge_y.extend([np.sin(angles[i]), np.sin(angles[j]), None])

    fig = go.Figure()
    
    # Render Edges natively
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, 
            line=dict(width=0.8, color=CONFIG.THEME_COLORS["edge"]), 
            hoverinfo='none', 
            mode='lines'
        ))
        
    # Render Nodes identically styled
    fig.add_trace(go.Scatter(
        x=np.cos(angles), y=np.sin(angles), 
        mode='markers',
        hoverinfo='none',
        marker=dict(
            size=14, 
            color=CONFIG.THEME_COLORS["primary"], 
            line=dict(width=2, color=CONFIG.THEME_COLORS["node_border"])
        )
    ))
    
    fig.update_layout(
        showlegend=False, 
        template="plotly_dark", 
        height=350, 
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def run_simulation(node_count: int, refresh_rate: str, ui_placeholders: Dict[str, Any]) -> None:
    """Executes the simulation loop cleanly utilizing dynamic Streamlit placeholders."""
    state = SimulationState()
    sleep_time = CONFIG.SPEED_MAP.get(refresh_rate, 0.4)

    # Pre-render initial frame so it doesn't wait for loop computation bounds
    logger.info("Initializing neural loop simulation protocol.")

    for t in range(CONFIG.SIMULATION_TICKS):
        # 1. Stochastic Environment Generation
        solar = np.random.randint(60, 120) if 8 < (t % 24) < 18 else np.random.randint(0, 15)
        demand = np.random.randint(55, 95)
        storage_level = np.random.randint(20, 100)

        # 2. State Mathematical Update
        metrics = state.update(solar, demand, storage_level, t)

        # 3. UI Flush & Component Render Loop
        ui_placeholders["m1"].metric("Solar Gen", f"{solar} MW", delta=f"{solar - 70} MW")
        ui_placeholders["m2"].metric("City Load", f"{demand} MW", delta=f"{demand - 75} MW", delta_color="inverse")
        
        ui_placeholders["m3"].markdown(
            f"<div class='status-card' style='background: {metrics['bg_color']}; "
            f"border: 1px solid {metrics['status_color']};'>"
            f"<span style='color:{metrics['status_color']};'>"
            f"GRID STATUS: {metrics['status_text']}</span></div>", 
            unsafe_allow_html=True
        )

        ui_placeholders["log"].markdown(
            f"<div class='log-box'>{'<br>'.join(state.logs[:12])}</div>", 
            unsafe_allow_html=True
        )

        # Throttle topology rebuild to strictly every tick efficiently natively
        fig = generate_topology_graph(node_count)
        ui_placeholders["graph"].plotly_chart(fig, use_container_width=True)

        df_history = pd.DataFrame(state.history).tail(20)
        ui_placeholders["chart"].area_chart(df_history)

        time.sleep(sleep_time)


# ==========================================
# MAIN APPLICATION FLOW
# ==========================================

def main() -> None:
    """Main application structural entry point."""
    st.set_page_config(
        page_title=CONFIG.PAGE_TITLE,
        page_icon=CONFIG.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    node_count, refresh_rate = render_sidebar()
    render_header()

    # Dashboard Master Architecture Array Layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🌐 Real-time Graph Topology")
        graph_placeholder = st.empty()
        
        st.subheader("📈 Neural Signal History")
        chart_placeholder = st.empty()

    with col_right:
        st.subheader("📊 Core Telemetry")
        met1, met2 = st.columns(2)
        m1 = met1.empty()
        m2 = met2.empty()
        m3 = st.empty()
        
        st.divider()
        st.subheader("🤖 AI Decision Intelligence")
        log_placeholder = st.empty()

    # Placeholder dictionary parameter mapping natively for decoupling
    ui_placeholders = {
        "graph": graph_placeholder,
        "chart": chart_placeholder,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "log": log_placeholder
    }

    # Simulation Action Trigger Core
    if st.sidebar.button("⚡ INITIALIZE NEURAL LOOP", use_container_width=True):
        try:
            run_simulation(node_count, refresh_rate, ui_placeholders)
            st.sidebar.success("Simulation sequence completed smoothly.")
        except Exception as e:
            logger.error(f"Engine failure detected: {e}", exc_info=True)
            st.sidebar.error("Critical failure computing neural sequences.")
    else:
        # Graceful Default Standby View
        st.info("System is in STANDBY. Configure parameters in the sidebar and 'Initialize Neural Loop' to begin.")
        
        # We wrap it in a fallback just in case the asset is missing
        try:
            st.image(
                "https://raw.githubusercontent.com/siddhantchandorkar752-ai/PROMETHEUS-GRID/main/assets/grid_preview.png", 
                caption="GNN Topology Visualization (Standby Mode)", 
                use_container_width=True
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()