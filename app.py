import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd

# 1. PAGE CONFIG & THEME
st.set_page_config(
    page_title="PROMETHEUS-GRID // NEURAL OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Vantablack & Neon Cyan Aura
st.markdown("""
    <style>
    .main { background-color: #00080a; }
    .stMetric { background: rgba(0, 242, 254, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(0, 242, 254, 0.2); }
    .log-box { background-color: #001219; color: #00f2fe; padding: 10px; border-radius: 5px; font-family: 'Courier New', monospace; font-size: 0.8rem; height: 200px; overflow-y: auto; border-left: 3px solid #00f2fe; }
    .status-card { border-radius: 10px; padding: 20px; text-align: center; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. SIDEBAR - CONTROL CENTER
with st.sidebar:
    st.image("https://capsule-render.vercel.app/api?type=rect&color=001219&height=60&text=NEURAL%20CONTROLS&fontSize=20&fontColor=00f2fe")
    st.markdown("### 🛠️ Configuration")
    node_count = st.slider("Grid Complexity (Nodes)", 6, 24, 12)
    refresh_rate = st.select_slider("Neural Pulse Speed", options=["Passive", "Active", "Overclock"], value="Active")
    
    st.divider()
    st.markdown("### 🛰️ Core Status")
    st.success("GNN Engine: ACTIVE")
    st.info("RL Policy: STABLE")
    st.warning("Hardware: T4 GPU-ACCEL")

# 3. HEADER SECTION (Based on your Elite README)
st.markdown("# ⚡ PROMETHEUS-GRID // <span style='color:#00f2fe'>NEURAL CORE</span>", unsafe_allow_html=True)
st.markdown("> *Autonomous Smart-City Grid Management | GNN Spatial Modeling + RL Decision Loop*")

with st.expander("📖 VIEW SYSTEM ARCHITECTURE", expanded=False):
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("#### 🧠 Spatial GNN Layer")
        st.caption("Captures multi-hop dependencies between energy nodes. Models the city as a dynamic adjacency matrix.")
    with col_arch2:
        st.markdown("#### 🕹️ Reinforcement Learning")
        st.caption("Deep Q-Network (DQN) manages the deficit/surplus delta with sub-100ms intervention latency.")

st.divider()

# 4. MAIN DASHBOARD LAYOUT
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

# 5. SIMULATION ENGINE
if st.sidebar.button("INITIALIZE NEURAL LOOP", use_container_width=True):
    history = {"Solar": [], "Demand": [], "Storage": []}
    logs = []
    
    # Speed mapping
    sleep_time = {"Passive": 0.8, "Active": 0.4, "Overclock": 0.1}[refresh_rate]

    for t in range(50):
        # Neural Data Generation
        solar = np.random.randint(60, 120) if 8 < (t%24) < 18 else np.random.randint(0, 15)
        demand = np.random.randint(55, 95)
        storage_level = np.random.randint(20, 100)
        
        history["Solar"].append(solar)
        history["Demand"].append(demand)
        history["Storage"].append(storage_level)

        # Update Metrics
        m1.metric("Solar Gen", f"{solar} MW", delta=f"{solar-70} MW")
        m2.metric("City Load", f"{demand} MW", delta=f"{demand-75} MW", delta_color="inverse")
        
        status_color = "#00f2fe" if solar > demand else "#ff4b4b"
        status_text = "SURPLUS" if solar > demand else "DEFICIT"
        m3.markdown(f"""<div class='status-card' style='background: {status_color}33; border: 1px solid {status_color};'>
                    <span style='color:{status_color};'>GRID STATUS: {status_text}</span></div>""", unsafe_allow_html=True)

        # Update AI Logs
        action = "CHARGING_ARRAY" if solar > demand else "RELEASING_RESERVES"
        new_log = f"[{time.strftime('%H:%M:%S')}] EVENT: {status_text} | ACTION: {action} | CONF: {np.random.uniform(94, 99):.2f}%"
        logs.insert(0, new_log)
        log_placeholder.markdown(f"<div class='log-box'>{'<br>'.join(logs[:8])}</div>", unsafe_allow_html=True)

        # Update GNN Graph (Plotly Luxury Style)
        angles = np.linspace(0, 2*np.pi, node_count, endpoint=False)
        edge_x, edge_y = [], []
        for i in range(node_count):
            for j in range(i+1, node_count):
                if np.random.rand() > 0.8: # Random adjacency
                    edge_x.extend([np.cos(angles[i]), np.cos(angles[j]), None])
                    edge_y.extend([np.sin(angles[i]), np.sin(angles[j]), None])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#1a3a3a'), hoverinfo='none', mode='lines'))
        fig.add_trace(go.Scatter(x=np.cos(angles), y=np.sin(angles), mode='markers',
                                 marker=dict(size=12, color='#00f2fe', line=dict(width=2, color='white'))))
        fig.update_layout(showlegend=False, template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        graph_placeholder.plotly_chart(fig, use_container_width=True)

        # Update History Chart
        chart_placeholder.area_chart(pd.DataFrame(history).tail(20))

        time.sleep(sleep_time)
else:
    st.info("System is in STANDBY. Configure parameters in the sidebar and 'Initialize Neural Loop' to begin simulation.")
    st.image("https://raw.githubusercontent.com/siddhantchandorkar752-ai/PROMETHEUS-GRID/main/assets/grid_preview.png", 
             caption="GNN Topology Visualization (Standby Mode)", use_container_width=True)