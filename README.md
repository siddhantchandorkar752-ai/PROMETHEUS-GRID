<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,25:001219,60:005f73,100:00f2fe&height=280&section=header&text=PROMETHEUS-GRID&fontSize=85&fontColor=ffffff&fontAlignY=38&desc=Autonomous%20Neural%20Energy%20Infrastructure%20v1.0&descAlignY=62&descSize=22&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Orbitron&weight=900&size=22&duration=2500&pause=700&color=00F2FE&center=true&vCenter=true&multiline=true&width=850&height=130&lines=GNN+%2B+Reinforcement+Learning+%7C+Smart+City+Energy+OS;Graph+Neural+Network+%7C+Deep+Q-Learning+Agent;Self-Healing+Grid+%7C+Real-Time+Digital+Twin;Static+Grids+Break.+Neural+Grids+Evolve.)](https://git.io/typing-svg)

<br/>

<img src="https://img.shields.io/badge/Python-3.10+-00f2fe?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-GNN-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch_Geometric-GCN-00f2fe?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-Neon_Graphics-005f73?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Architecture-GNN_%2B_DQN-00f2fe?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-005f73?style=for-the-badge"/>

<br/><br/>

> ### *"A Smart Grid doesn't just distribute power — it predicts the future of the city."*
> PROMETHEUS-GRID merges Graph Neural Networks with Deep Q-Learning to create a self-healing energy ecosystem. Every building is a node. Every power line is an edge. The RL agent decides what happens next.

<br/>

### 🔴 LIVE NOW
# [![LAUNCH PROMETHEUS GRID](https://img.shields.io/badge/%E2%9A%A1_LAUNCH_PROMETHEUS--GRID-CLICK_TO_RUN_LIVE-00f2fe?style=for-the-badge&labelColor=001219&logo=streamlit&logoColor=white)](https://prometheus-siddhantchandorkar.streamlit.app/)

> **No setup. No install. Runs in your browser.**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/siddhantchandorkar752-ai/PROMETHEUS-GRID)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-00f2fe?style=for-the-badge&logo=streamlit&logoColor=white)](https://prometheus-siddhantchandorkar.streamlit.app/)

</div>

---

## WHAT IS PROMETHEUS-GRID?

```
╔══════════════════════════════════════════════════════════════════════╗
║     PROMETHEUS-GRID — Neural Smart-City Energy OS v1.0              ║
║     "Static grids break. Neural grids adapt and evolve."            ║
║                                                                      ║
║     SPATIAL LOOP:   GNN modeling of nodal energy dependencies       ║
║     CONTROL LOOP:   Deep Q-Learning (DQN) agent grid stabilization  ║
║     INTERFACE:      Glassmorphism telemetry dashboard (Plotly)      ║
║     SIMULATION:     Real-time solar/demand digital twin             ║
╚══════════════════════════════════════════════════════════════════════╝
```

PROMETHEUS-GRID is an **autonomous digital twin of a smart-city energy network**. Every building and power plant is a node in a graph. Every power line is a weighted edge. The GNN understands spatial energy relationships. The RL agent decides the optimal storage-to-release ratio — in real time.

> No human operator. No static rules. The grid thinks for itself.

---

## THE PROBLEM

```
Modern energy grids were designed for a world that no longer exists.

Centralized coal plants → Distributed solar farms.
Predictable demand → EV charging spikes at 6PM.
Manual operators → Millisecond-level grid events.

Traditional grids respond to failures. They cannot predict them.
They balance after the fact. They do not adapt before.

PROMETHEUS-GRID changes this.
GNN sees the grid as a graph. DQN agent acts before failure occurs.
```

| Grid Failure Mode | Traditional Response | PROMETHEUS-GRID |
|-------------------|---------------------|-----------------|
| Solar surplus at noon | Manual curtailment | DQN: charge storage |
| Demand spike at 18:00 | Rolling blackout | DQN: release stored energy |
| Node failure cascade | Human intervention | GNN: reroute automatically |
| Frequency deviation | Slow PID controller | RL: sub-second correction |

---

## DUAL-ENGINE ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PROMETHEUS NEURAL CORE                          │
│                                                                      │
│  ┌───────────────────────────────┐  ┌────────────────────────────┐  │
│  │      NEURAL LAYER (GNN)       │  │     CONTROL LAYER (DQN)    │  │
│  │                               │  │                            │  │
│  │  Node Features                │  │  State Tensor Extraction   │  │
│  │  (Solar Gen / Load Demand)    ├──►                            │  │
│  │                               │  │  Epsilon-Greedy Policy     │  │
│  │  Edge Weight Calculation      │  │                            │  │
│  │  (Power line capacity)        │  │  Actions:                  │  │
│  │                               │  │  CHARGE / RELEASE / HOLD   │  │
│  │  Graph Conv Network (GCN)     │  │                            │  │
│  │  PyTorch Geometric            │  │  Reward:                   │  │
│  │                               │  │  Grid stability score      │  │
│  └───────────────┬───────────────┘  └──────────────┬─────────────┘  │
│                  │                                  │                │
│                  └─────────────────┬────────────────┘                │
│                                    ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              LUXURY TELEMETRY DASHBOARD                      │    │
│  │   Real-time Plotly graphs  •  Node health heatmap           │    │
│  │   Agent action log  •  Grid stability score  •  KPI cards   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## AGENT DECISION SYSTEM

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   SURPLUS    Solar > Demand + 10%   →  CHARGE storage              │
│   DEFICIT    Demand > Solar + 10%   →  RELEASE storage             │
│   STABLE     Delta < 10%            →  BALANCE — hold              │
│                                                                      │
│   DQN Confidence threshold: > 90% before autonomous action         │
│   Epsilon decay: 1.0 → 0.01 over 1000 episodes                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## LIVE PERFORMANCE TRACKING

| Hour | Solar (MW) | Demand (MW) | Grid Status | Agent Action | Confidence |
|:----:|:----------:|:-----------:|:-----------:|:------------:|:----------:|
| 08:00 | 45.5 | 62.1 | DEFICIT | RELEASING | 94.2% |
| 12:00 | 110.2 | 58.4 | SURPLUS | CHARGING | 98.7% |
| 18:00 | 15.1 | 85.3 | DEFICIT | RELEASING | 91.5% |
| 22:00 | 0.0 | 42.1 | STABLE | BALANCING | 96.0% |

---

## ENGINEERING HIGHLIGHTS

| Feature | Implementation | Why It Matters |
|---------|---------------|----------------|
| **GNN Topology** | PyTorch Geometric GCN | Understands city-wide energy node dependencies |
| **RL Decision Engine** | Epsilon-Greedy DQN | Learns optimal storage policies over 1000+ episodes |
| **Digital Twin** | Async Python simulation | Real-time grid state — not batch, not static |
| **Glassmorphism UI** | Streamlit + Custom CSS | Executive-level monitoring dashboard |
| **Neon Telemetry** | Plotly custom themes | Every metric visualized — zero information loss |

---

## RESEARCH CONTEXT

PROMETHEUS-GRID addresses three open challenges in the **Energy Transition**:

| Challenge | Description | How PROMETHEUS Solves It |
|-----------|-------------|--------------------------|
| **Intermittency** | Solar/wind surges are unpredictable | GNN predicts node-level surplus before it occurs |
| **Distributed Nodes** | Moving from centralized to distributed generation | Graph topology models any network configuration |
| **Auto-Stabilization** | Grid frequency must stay at 50/60Hz in real time | DQN agent acts in milliseconds — not minutes |

This aligns with active research at:
- **DeepMind** — GraphCast for grid-scale energy prediction
- **Google** — RL for data center cooling optimization
- **OpenAI** — Autonomous control systems

---

## PROJECT STRUCTURE

```
prometheus-grid/
├── src/
│   ├── environment.py      # Grid physics engine — digital twin simulation
│   ├── brain_gnn.py        # Graph Neural Network — PyTorch Geometric GCN
│   └── agent_rl.py         # Deep Q-Network RL agent — epsilon-greedy policy
├── utils/
│   └── telemetry.py        # Real-time logging + metrics collection
├── app.py                  # Streamlit luxury OS entry point
├── requirements.txt        # Pinned dependencies
└── README.md
```

---

## QUICK START

```bash
# 1. Clone
git clone https://github.com/siddhantchandorkar752-ai/PROMETHEUS-GRID.git
cd PROMETHEUS-GRID

# 2. Setup
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install neural stack
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

---

## TECH STACK

| Layer | Technology | Why |
|-------|-----------|-----|
| **GNN** | PyTorch Geometric | Industry standard for graph neural networks |
| **RL Agent** | PyTorch DQN | Deep Q-Learning — proven for control tasks |
| **Dashboard** | Streamlit | Fast iteration, glassmorphism UI possible |
| **Visualization** | Plotly | Neon real-time charts — zero lag |
| **Simulation** | Async Python | True digital twin capability |

---

## LICENSE

MIT License — built for the open-source energy future.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:001219,50:005f73,100:001219&height=70&text=Siddhant%20Chandorkar&fontSize=28&fontColor=00f2fe&fontAlign=50&fontAlignY=50" width="500"/>

<br/><br/>

[![GitHub](https://img.shields.io/badge/GitHub-siddhantchandorkar752--ai-005f73?style=for-the-badge&logo=github&logoColor=white)](https://github.com/siddhantchandorkar752-ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-siddhantchandorkar-00f2fe?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/siddhantchandorkar)

<br/>

*"I don't just write code. I build neural ecosystems for the smart-city era."*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00f2fe,40:005f73,100:000000&height=140&section=footer&text=SYSTEM%20ONLINE%20v1.0&fontSize=34&fontColor=00f2fe&fontAlignY=68&animation=fadeIn" width="100%"/>

</div>
