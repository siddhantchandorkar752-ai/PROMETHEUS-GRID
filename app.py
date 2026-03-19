import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time

# FAANG Layout Config
st.set_page_config(page_title="PROMETHEUS-GRID // CORE", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Luxury Aura
st.markdown("""
    <style>
    .metric-card { background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; border: 1px solid rgba(0,242,254,0.1); }
    .stButton>button { width: 100%; background: linear-gradient(45deg, #005f73, #00f2fe); color: white; border: none; }
    </style>
""", unsafe_allow_html=True)

# 1. SIDEBAR: Controls and Status
with st.sidebar:
    st.image("https://capsule-render.vercel.app/api?type=rect&color=001219&height=60&text=SYSTEM%20CONTROLLERS&fontSize=20&fontColor=00f2fe")
    nodes = st.slider("Neural Nodes", 5, 20, 12)
    speed = st.select_slider("Simulation Pulse", options=["Slow", "Real-time", "Hyper-drive"], value="Real-time")
    st.divider()
    st.markdown("### 🛰️ Connectivity\n**Status:** `ONLINE`  \n**Core:** `GNN-v1.0`  \n**Agent:** `DQN-Stabilizer`")

# 2. HEADER: Architecture Briefing
st.markdown("# ⚡ PROMETHEUS-GRID // <span style='color:#00f2fe'>NEURAL CORE</span>", unsafe_allow_html=True)
with st.expander("📖 SYSTEM ARCHITECTURE & INTELLIGENCE BRIEFING", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🧠 Graph Neural Network (GNN)")
        st.write("Models spatial dependencies between city nodes. Understands energy flow topology beyond linear paths.")
    with col_b:
        st.markdown("#### 🕹️ Reinforcement Learning (RL)")
        st.write("Deep Q-Network (DQN) optimizes storage-to-release ratios based on real-time deficit/surplus delta.")

st.divider()

# 3. MAIN STAGE
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌐 Real-time Graph Topology")
    plot_spot = st.empty()
    st.markdown("---")
    st.subheader("📈 Neural Signal Analysis")
    chart_spot = st.empty()

with col2:
    st.subheader("📊 Live Telemetry")
    m1, m2, m3 = st.empty(), st.empty(), st.empty()
    st.markdown("---")
    st.subheader("🤖 AI Decision Log")
    log_spot = st.empty()

# 4. SIMULATION LOGIC
if st.sidebar.button("INITIALIZE NEURAL LOOP"):
    # Simulation data placeholders
    history = {"solar": [], "demand": []}
    
    for i in range(24):
        # Neural Simulation Mock
        solar = np.random.randint(40, 110) if 10 < i < 17 else np.random.randint(0, 20)
        demand = np.random.randint(50, 90)
        history["solar"].append(solar)
        history["demand"].append(demand)
        
        # Update Metrics
        m1.metric("Solar Generation", f"{solar} MW", delta=f"{solar-50} MW")
        m2.metric("City Load", f"{demand} MW", delta=f"{demand-70} MW", delta_color="inverse")
        status = "🟢 SURPLUS" if solar > demand else "🔴 DEFICIT"
        m3.markdown(f"<div class='metric-card'><h4>Grid Status</h4><h2>{status}</h2></div>", unsafe_allow_html=True)
        
        # Update Decision Log
        action = "CHARGING STORAGE" if solar > demand else "STABILIZING GRID"
        log_spot.code(f"[HOUR {i:02d}:00] AGENT_ACTION: {action}\n[CONFIDENCE]: {np.random.uniform(92, 99):.2f}%")
        
        # Update Graph (Plotly Luxury Style)
        fig = go.Figure()
        theta = np.linspace(0, 2*np.pi, nodes, endpoint=False)
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='markers+text', 
                                 marker=dict(size=25, color='#00f2fe', line=dict(width=2, color='white')),
                                 text=[str(x) for x in range(nodes)]))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0))
        plot_spot.plotly_chart(fig, use_container_width=True)
        
        # Update Line Chart
        chart_spot.line_chart(history)
        
        time.sleep(0.5 if speed == "Real-time" else 0.1)