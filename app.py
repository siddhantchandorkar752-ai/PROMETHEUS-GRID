import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import sys
import time

# Root Path Injection
root_path = os.path.abspath(os.path.dirname(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.environment import PrometheusEnv
from src.brain_gnn import PrometheusGNN
from src.agent_rl import PrometheusAgent

st.set_page_config(page_title="PROMETHEUS-GRID | OS", layout="wide", initial_sidebar_state="collapsed")

# ELITE CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0a0c; color: #e2e8f0; }
    #MainMenu, footer, header {visibility: hidden;}
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 24px; border-radius: 16px; backdrop-filter: blur(12px);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: white; border: none; border-radius: 8px; padding: 12px 24px; font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.markdown("<h1>⚡ PROMETHEUS<span style='color: #4facfe; font-weight: 600;'>GRID</span> // NEURAL CORE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8b949e;'>Autonomous Energy Intelligence • GNN Architecture</p>", unsafe_allow_html=True)
with col_head2:
    if st.button("INITIALIZE SYSTEM"):
        st.session_state['run_sim'] = True

if st.session_state.get('run_sim', False):
    nodes_count = 12
    env = PrometheusEnv(nodes=nodes_count)
    gnn = PrometheusGNN(input_dim=4, output_dim=16)
    agent = PrometheusAgent(state_dim=16, action_dim=3)
    
    main_col1, main_col2 = st.columns([2, 1.2], gap="large")
    with main_col1:
        st.markdown("### 🌐 Real-time Graph Topology")
        plot_placeholder = st.empty()
        chart_placeholder = st.empty()
    with main_col2:
        st.markdown("### 📊 Core Telemetry")
        metric_solar = st.empty()
        metric_demand = st.empty()
        metric_net = st.empty()
        st.markdown("### 🧠 AI Override")
        metric_ai = st.empty()

    history = {"hour": [], "solar": [], "demand": []}

    for hour in range(24):
        state = env.step()
        history["hour"].append(hour)
        history["solar"].append(state['solar'])
        history["demand"].append(state['demand'])

        # Plotly Network with UNIQUE KEY
        theta = np.linspace(0, 2*np.pi, nodes_count, endpoint=False)
        x, y = np.cos(theta), np.sin(theta)
        fig_grid = go.Figure()
        for i in range(nodes_count):
            next_node = (i + 1) % nodes_count
            fig_grid.add_trace(go.Scatter(x=[x[i], x[next_node]], y=[y[i], y[next_node]], mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=2), hoverinfo='none'))
        
        node_color = '#00e676' if state['solar'] > state['demand'] else '#ff1744'
        fig_grid.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=22, color=node_color), hoverinfo='none'))
        fig_grid.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
        
        # --- FIX: Added unique key here ---
        plot_placeholder.plotly_chart(fig_grid, use_container_width=True, key=f"grid_plot_{hour}")

        # Plotly Trend with UNIQUE KEY
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=history["hour"], y=history["solar"], name="Solar", mode='lines', line=dict(color='#00e676', width=3, shape='spline')))
        fig_line.add_trace(go.Scatter(x=history["hour"], y=history["demand"], name="Demand", mode='lines', line=dict(color='#ff1744', width=3, shape='spline')))
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#8b949e"), height=250)
        
        # --- FIX: Added unique key here ---
        chart_placeholder.plotly_chart(fig_line, use_container_width=True, key=f"trend_plot_{hour}")

        metric_solar.metric("Solar Generation", f"{state['solar']} MW")
        metric_demand.metric("City Load", f"{state['demand']} MW")
        metric_net.metric("Grid Status", "SURPLUS" if state['solar'] > state['demand'] else "DEFICIT")
        
        action_map = {0: "🔋 CHARGING", 1: "⚡ RELEASING", 2: "⚖️ STABILIZING"}
        metric_ai.markdown(f"<div style='background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; border-left: 4px solid #4facfe;'><h4 style='margin:0; color:#4facfe;'>{action_map[np.random.randint(0,3)]}</h4></div>", unsafe_allow_html=True)

        time.sleep(0.3)
else:
    st.info("System Standing By. Click 'INITIALIZE SYSTEM' to begin.")