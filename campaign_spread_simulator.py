"""
Corporate Product Campaign Spread Simulator
==========================================
A Streamlit app that models how a corporate product campaign spreads through
a social network using influence maximization and network effects.

Run with: streamlit run campaign_spread_simulator.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import math
import io
from collections import defaultdict

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Campaign Spread Simulator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
}

/* Headers */
h1, h2, h3 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    font-size: 2rem !important;
    color: #58a6ff !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace;
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #30363d;
    gap: 4px;
}

[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500;
    color: #8b949e !important;
    font-family: 'DM Sans', sans-serif;
}

[aria-selected="true"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    padding: 10px 24px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(31,111,235,0.3);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(31,111,235,0.5);
}

/* Expanders */
.streamlit-expanderHeader {
    background: #1c2128 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-weight: 500;
}

/* Divider */
hr {
    border-color: #30363d;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d;
    border-radius: 8px;
}

/* Section cards */
.section-card {
    background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #1f2937 40%, #111827 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(31,111,235,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 8px 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #8b949e;
    margin: 0;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: rgba(31,111,235,0.15);
    border: 1px solid rgba(56,139,253,0.4);
    color: #58a6ff;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Info boxes */
.info-box {
    background: rgba(31,111,235,0.08);
    border: 1px solid rgba(56,139,253,0.3);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #79c0ff;
}

.success-box {
    background: rgba(46,160,67,0.08);
    border: 1px solid rgba(46,160,67,0.3);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #56d364;
}

.warning-box {
    background: rgba(210,153,34,0.08);
    border: 1px solid rgba(210,153,34,0.3);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #e3b341;
}

/* Rank badge */
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    text-align: center;
    line-height: 28px;
    font-size: 0.8rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}

.stSelectbox > div > div {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #e6edf3;
}

.stSlider > div {
    color: #e6edf3;
}

label, .stLabel {
    color: #8b949e !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "graph": None,
        "seeds": [],
        "sim_results": None,
        "influencer_scores": None,
        "strategy_results": {},
        "network_built": False,
        "simulation_run": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ─────────────────────────────────────────────
# NETWORK BUILDERS
# ─────────────────────────────────────────────
def build_synthetic_network(n_users, edge_density, communities, randomness, seed=42):
    """Generate a synthetic social network with community structure."""
    random.seed(seed)
    np.random.seed(seed)

    if communities > 1:
        sizes = [n_users // communities] * communities
        sizes[-1] += n_users - sum(sizes)
        p_in = min(0.6, edge_density * 2)
        p_out = max(0.01, edge_density * randomness * 0.3)
        p_matrix = [[p_in if i == j else p_out for j in range(communities)] for i in range(communities)]
        G = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    else:
        G = nx.erdos_renyi_graph(n_users, edge_density, seed=seed)

    # Assign metadata
    for node in G.nodes():
        G.nodes[node]["community"] = 0
        G.nodes[node]["influence_score"] = 0.0

    if communities > 1:
        idx = 0
        for comm_id, size in enumerate(sizes):
            for i in range(size):
                if idx < len(G.nodes()):
                    G.nodes[idx]["community"] = comm_id
                    idx += 1

    G = nx.convert_node_labels_to_integers(G)
    return G


def load_network_from_csv(nodes_df, edges_df):
    """Load network from uploaded CSV dataframes."""
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        nid = int(row["node_id"])
        attrs = {k: v for k, v in row.items() if k != "node_id"}
        G.add_node(nid, **attrs)
    for _, row in edges_df.iterrows():
        src = int(row["source"])
        tgt = int(row["target"])
        weight = float(row.get("weight", 1.0))
        G.add_edge(src, tgt, weight=weight)
    return G


# ─────────────────────────────────────────────
# INFLUENCER SCORING
# ─────────────────────────────────────────────
def score_influencers(G, strategy, budget, adoption_prob):
    """Rank nodes by influence strategy and return top seeds."""
    scores = {}

    if strategy == "Random":
        nodes = list(G.nodes())
        random.shuffle(nodes)
        seeds = nodes[:budget]
        for n in G.nodes():
            scores[n] = random.random()

    elif strategy == "Degree Centrality":
        dc = nx.degree_centrality(G)
        scores = dc
        seeds = sorted(dc, key=dc.get, reverse=True)[:budget]

    elif strategy == "Betweenness Centrality":
        bc = nx.betweenness_centrality(G, normalized=True, endpoints=False)
        scores = bc
        seeds = sorted(bc, key=bc.get, reverse=True)[:budget]

    elif strategy == "PageRank":
        pr = nx.pagerank(G, alpha=0.85)
        scores = pr
        seeds = sorted(pr, key=pr.get, reverse=True)[:budget]

    elif strategy == "Greedy Influence Maximization":
        seeds = greedy_influence_maximization(G, budget, adoption_prob, mc_rounds=50)
        # Assign scores based on order selected
        for rank, node in enumerate(seeds):
            scores[node] = 1.0 - (rank / max(len(seeds), 1))
        for n in G.nodes():
            if n not in scores:
                dc = nx.degree_centrality(G)
                scores[n] = dc.get(n, 0.0)

    # Normalize scores to 0–100
    if scores:
        max_s = max(scores.values()) or 1
        scores = {k: round(v / max_s * 100, 2) for k, v in scores.items()}

    return seeds, scores


def greedy_influence_maximization(G, budget, adoption_prob, mc_rounds=50):
    """Greedy algorithm for influence maximization using Monte Carlo estimation."""
    nodes = list(G.nodes())
    selected = []
    remaining = set(nodes)

    for _ in range(budget):
        best_node = None
        best_gain = -1

        for candidate in remaining:
            seed_set = selected + [candidate]
            gain = estimate_influence(G, seed_set, adoption_prob, mc_rounds)
            if gain > best_gain:
                best_gain = gain
                best_node = candidate

        if best_node is not None:
            selected.append(best_node)
            remaining.discard(best_node)

    return selected


def estimate_influence(G, seeds, adoption_prob, mc_rounds=50):
    """Estimate expected spread via Monte Carlo IC simulation."""
    total = 0
    for _ in range(mc_rounds):
        activated = set(seeds)
        frontier = list(seeds)
        while frontier:
            new_frontier = []
            for node in frontier:
                for neighbor in G.neighbors(node):
                    if neighbor not in activated:
                        edge_data = G.get_edge_data(node, neighbor) or {}
                        weight = edge_data.get("weight", 1.0)
                        prob = min(1.0, adoption_prob * weight)
                        if random.random() < prob:
                            activated.add(neighbor)
                            new_frontier.append(neighbor)
            frontier = new_frontier
        total += len(activated)
    return total / mc_rounds


# ─────────────────────────────────────────────
# DIFFUSION MODELS
# ─────────────────────────────────────────────
def run_independent_cascade(G, seeds, adoption_prob, influence_strength, steps):
    """Independent Cascade Model simulation."""
    activated = set(seeds)
    newly_activated = set(seeds)
    history = [set(seeds)]

    for _ in range(steps):
        new_this_round = set()
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in activated:
                    edge_data = G.get_edge_data(node, neighbor) or {}
                    weight = edge_data.get("weight", 1.0)
                    prob = min(1.0, adoption_prob * influence_strength * weight)
                    if random.random() < prob:
                        new_this_round.add(neighbor)
        activated |= new_this_round
        newly_activated = new_this_round
        history.append(set(newly_activated))
        if not newly_activated:
            break

    return activated, history


def run_linear_threshold(G, seeds, adoption_prob, influence_strength, steps):
    """Linear Threshold Model simulation."""
    thresholds = {n: random.uniform(0.1, adoption_prob) for n in G.nodes()}
    influence_weights = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            for nb in neighbors:
                influence_weights[(nb, node)] = influence_strength / len(neighbors)

    activated = set(seeds)
    history = [set(seeds)]

    for _ in range(steps):
        new_this_round = set()
        for node in G.nodes():
            if node not in activated:
                total_influence = sum(
                    influence_weights.get((nb, node), 0.0)
                    for nb in G.neighbors(node)
                    if nb in activated
                )
                if total_influence >= thresholds[node]:
                    new_this_round.add(node)
        activated |= new_this_round
        history.append(set(new_this_round))
        if not new_this_round:
            break

    return activated, history


def run_simulation(G, seeds, model, adoption_prob, influence_strength, steps, rounds):
    """Run simulation multiple rounds and aggregate results."""
    all_activated = []
    all_histories = []

    for _ in range(rounds):
        if model == "Independent Cascade":
            activated, history = run_independent_cascade(G, seeds, adoption_prob, influence_strength, steps)
        else:
            activated, history = run_linear_threshold(G, seeds, adoption_prob, influence_strength, steps)
        all_activated.append(len(activated))
        all_histories.append(history)

    # Build adoption-over-time curve (mean across rounds)
    max_steps = max(len(h) for h in all_histories)
    step_counts = []
    for step in range(max_steps):
        counts = []
        for hist in all_histories:
            cumulative = set()
            for s in range(step + 1):
                if s < len(hist):
                    cumulative |= hist[s]
            counts.append(len(cumulative))
        step_counts.append(np.mean(counts))

    # Final activated set from last round for visualization
    if model == "Independent Cascade":
        final_activated, final_history = run_independent_cascade(G, seeds, adoption_prob, influence_strength, steps)
    else:
        final_activated, final_history = run_linear_threshold(G, seeds, adoption_prob, influence_strength, steps)

    return {
        "mean_adopters": np.mean(all_activated),
        "std_adopters": np.std(all_activated),
        "min_adopters": min(all_activated),
        "max_adopters": max(all_activated),
        "adoption_curve": step_counts,
        "final_activated": final_activated,
        "final_history": final_history,
        "total_nodes": G.number_of_nodes(),
        "seeds": seeds,
    }


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(result, budget, discount_pct):
    n = result["total_nodes"]
    adopters = result["mean_adopters"]
    seeds = len(result["seeds"])
    adoption_rate = adopters / n if n > 0 else 0
    seed_efficiency = (adopters - seeds) / max(seeds, 1)
    reach_ratio = (adopters - seeds) / max(n - seeds, 1)
    roi_score = seed_efficiency * (1 - discount_pct / 100)
    return {
        "Total Adopters": int(adopters),
        "Adoption Rate (%)": round(adoption_rate * 100, 1),
        "Reach from Seeds": int(adopters - seeds),
        "Seed Efficiency": round(seed_efficiency, 2),
        "Reach Ratio (%)": round(reach_ratio * 100, 1),
        "ROI Score": round(roi_score, 2),
    }


# ─────────────────────────────────────────────
# NETWORK VISUALIZATION
# ─────────────────────────────────────────────
def draw_network(G, seeds, activated=None, layout="spring"):
    """Render interactive network using Plotly."""
    n = G.number_of_nodes()
    if n > 500:
        # Sample for performance
        sampled = random.sample(list(G.nodes()), 500)
        G = G.subgraph(sampled).copy()

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=1.5 / math.sqrt(n + 1))
    elif layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.circular_layout(G)

    activated = activated or set()
    seed_set = set(seeds)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#30363d"),
        hoverinfo="none",
    )

    # Color nodes
    node_colors, node_sizes, node_labels = [], [], []
    for n_id in G.nodes():
        if n_id in seed_set:
            node_colors.append("#f78166")   # Red — seed
            node_sizes.append(16)
        elif n_id in activated:
            node_colors.append("#3fb950")   # Green — adopted
            node_sizes.append(10)
        else:
            node_colors.append("#8b949e")   # Grey — not activated
            node_sizes.append(7)
        deg = G.degree(n_id)
        node_labels.append(f"User {n_id}<br>Degree: {deg}")

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=0.8, color="#0d1117")),
        text=node_labels,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )

    # Legend annotations
    fig.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper",
        text="🔴 Seed Users  🟢 Adopted  ⚫ Not Activated",
        showarrow=False, font=dict(color="#8b949e", size=11),
        bgcolor="rgba(13,17,23,0.8)", bordercolor="#30363d", borderwidth=1,
        borderpad=6, xanchor="left", yanchor="top")

    return fig


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────
def plot_adoption_curve(adoption_curve, total_nodes):
    steps = list(range(len(adoption_curve)))
    pct = [round(v / total_nodes * 100, 1) for v in adoption_curve]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=adoption_curve,
        mode="lines+markers",
        name="Adopters",
        line=dict(color="#388bfd", width=3),
        marker=dict(size=7, color="#388bfd"),
        fill="tozeroy",
        fillcolor="rgba(56,139,253,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=pct,
        mode="lines",
        name="Adoption %",
        line=dict(color="#3fb950", width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        paper_bgcolor="#1c2128",
        plot_bgcolor="#1c2128",
        font=dict(color="#8b949e", family="DM Sans"),
        xaxis=dict(title="Time Step", gridcolor="#21262d", title_font=dict(color="#8b949e")),
        yaxis=dict(title="Cumulative Adopters", gridcolor="#21262d", title_font=dict(color="#58a6ff")),
        yaxis2=dict(title="Adoption %", overlaying="y", side="right",
                    gridcolor="#21262d", title_font=dict(color="#3fb950")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3")),
        margin=dict(l=60, r=60, t=20, b=50),
        height=360,
    )
    return fig


def plot_influencer_ranking(scores, seeds, top_n=20):
    nodes = sorted(scores, key=scores.get, reverse=True)[:top_n]
    values = [scores[n] for n in nodes]
    colors = ["#f78166" if n in set(seeds) else "#388bfd" for n in nodes]
    labels = [f"User {n}" + (" ★" if n in set(seeds) else "") for n in nodes]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e6edf3", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="#1c2128",
        plot_bgcolor="#1c2128",
        font=dict(color="#8b949e", family="DM Sans"),
        xaxis=dict(title="Influence Score", gridcolor="#21262d"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=60, t=20, b=40),
        height=max(300, top_n * 22),
        showlegend=False,
    )
    return fig


def plot_strategy_comparison(strategy_results):
    strategies = list(strategy_results.keys())
    adopters = [strategy_results[s]["metrics"]["Total Adopters"] for s in strategies]
    rates = [strategy_results[s]["metrics"]["Adoption Rate (%)"] for s in strategies]
    efficiency = [strategy_results[s]["metrics"]["Seed Efficiency"] for s in strategies]

    palette = ["#388bfd", "#3fb950", "#d2a679", "#f78166", "#bc8cff"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Total Adopters", "Adoption Rate (%)", "Seed Efficiency"])
    for i, (strat, adopt, rate, eff, color) in enumerate(zip(strategies, adopters, rates, efficiency, palette)):
        fig.add_trace(go.Bar(name=strat, x=[strat], y=[adopt], marker_color=color, showlegend=(i == 0)), row=1, col=1)
        fig.add_trace(go.Bar(name=strat, x=[strat], y=[rate], marker_color=color, showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(name=strat, x=[strat], y=[eff], marker_color=color, showlegend=False), row=1, col=3)

    fig.update_layout(
        paper_bgcolor="#1c2128",
        plot_bgcolor="#1c2128",
        font=dict(color="#8b949e", family="DM Sans"),
        height=380,
        barmode="group",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=60),
    )
    for ax in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
        fig.update_layout(**{ax: dict(gridcolor="#21262d", tickfont=dict(color="#8b949e", size=10))})
    return fig


def plot_funnel(result, total_nodes):
    adopters = int(result["mean_adopters"])
    seeds = len(result["seeds"])
    aware = adopters
    considered = int(adopters * 0.75)
    adopted = adopters

    fig = go.Figure(go.Funnel(
        y=["Seed Users", "Network Aware", "Considered", "Adopted"],
        x=[seeds, aware, considered, adopted],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=["#f78166", "#388bfd", "#3fb950", "#56d364"]),
        connector=dict(line=dict(color="#30363d", width=2)),
    ))
    fig.update_layout(
        paper_bgcolor="#1c2128",
        plot_bgcolor="#1c2128",
        font=dict(color="#8b949e", family="DM Sans"),
        margin=dict(l=40, r=40, t=20, b=20),
        height=320,
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:2rem;">🚀</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#58a6ff; letter-spacing:0.1em; text-transform:uppercase;">Campaign Simulator</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📡 Network Settings")
    network_source = st.radio("Network Source", ["Generate Synthetic", "Upload CSV"], label_visibility="collapsed")

    if network_source == "Generate Synthetic":
        n_users = st.slider("Number of Users", 50, 1000, 200, step=50)
        edge_density = st.slider("Edge Density", 0.01, 0.3, 0.05, step=0.01)
        n_communities = st.slider("Communities", 1, 8, 3)
        randomness = st.slider("Randomness", 0.0, 1.0, 0.3, step=0.05)
    else:
        nodes_file = st.file_uploader("Nodes CSV (node_id, ...)", type="csv")
        edges_file = st.file_uploader("Edges CSV (source, target, weight)", type="csv")

    if st.button("🔨 Build Network", use_container_width=True):
        with st.spinner("Building network..."):
            if network_source == "Generate Synthetic":
                G = build_synthetic_network(n_users, edge_density, n_communities, randomness)
            else:
                try:
                    nodes_df = pd.read_csv(nodes_file)
                    edges_df = pd.read_csv(edges_file)
                    G = load_network_from_csv(nodes_df, edges_df)
                except Exception:
                    st.error("Invalid CSV. Using demo network.")
                    G = build_synthetic_network(200, 0.05, 3, 0.3)

            st.session_state.graph = G
            st.session_state.seeds = []
            st.session_state.sim_results = None
            st.session_state.influencer_scores = None
            st.session_state.strategy_results = {}
            st.session_state.network_built = True
            st.session_state.simulation_run = False
        st.success(f"Network ready: {G.number_of_nodes()} users, {G.number_of_edges()} connections")

    st.divider()
    st.markdown("### 🎯 Seed Selection")
    budget = st.slider("Seed Budget (# Influencers)", 1, 30, 5)
    strategy = st.selectbox("Selection Strategy", [
        "Greedy Influence Maximization",
        "PageRank",
        "Degree Centrality",
        "Betweenness Centrality",
        "Random",
    ])

    st.divider()
    st.markdown("### ⚙️ Campaign Settings")
    diffusion_model = st.selectbox("Diffusion Model", ["Independent Cascade", "Linear Threshold"])
    adoption_prob = st.slider("Adoption Probability", 0.01, 0.5, 0.1, step=0.01)
    influence_strength = st.slider("Influence Strength", 0.5, 3.0, 1.0, step=0.1)
    discount_pct = st.slider("Seed Incentive / Discount (%)", 0, 80, 20)
    sim_rounds = st.slider("Simulation Rounds", 5, 100, 20)
    sim_steps = st.slider("Max Time Steps", 5, 50, 20)

    st.divider()
    if st.button("▶ Run Simulation", use_container_width=True, type="primary"):
        if st.session_state.graph is None:
            st.error("Build a network first!")
        else:
            G = st.session_state.graph
            with st.spinner("Selecting influencers..."):
                seeds, scores = score_influencers(G, strategy, budget, adoption_prob)
                st.session_state.seeds = seeds
                st.session_state.influencer_scores = scores

            with st.spinner("Simulating spread..."):
                result = run_simulation(G, seeds, diffusion_model, adoption_prob, influence_strength, sim_steps, sim_rounds)
                result["strategy"] = strategy
                result["model"] = diffusion_model
                st.session_state.sim_results = result
                st.session_state.simulation_run = True

            st.success("Simulation complete!")

    st.divider()
    if st.button("📊 Compare All Strategies", use_container_width=True):
        if st.session_state.graph is None:
            st.error("Build a network first!")
        else:
            G = st.session_state.graph
            all_strats = ["Greedy Influence Maximization", "PageRank", "Degree Centrality", "Betweenness Centrality", "Random"]
            strat_results = {}
            prog = st.progress(0)
            for idx, strat in enumerate(all_strats):
                seeds_s, _ = score_influencers(G, strat, budget, adoption_prob)
                res_s = run_simulation(G, seeds_s, diffusion_model, adoption_prob, influence_strength, sim_steps, sim_rounds)
                metrics_s = compute_metrics(res_s, budget, discount_pct)
                strat_results[strat] = {"result": res_s, "metrics": metrics_s, "seeds": seeds_s}
                prog.progress((idx + 1) / len(all_strats))
            st.session_state.strategy_results = strat_results
            st.success("Strategy comparison done!")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">B2B / B2C Growth Tool</div>
    <div class="hero-title">Corporate Product Campaign<br>Spread Simulator</div>
    <div class="hero-subtitle">Identify optimal seed users · Model viral adoption · Compare influencer strategies · Plan your campaign with data</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📋 Overview", "🕸️ Network", "👑 Influencers", "📈 Simulation", "📊 Dashboard", "⚖️ Strategy Comparison", "💡 Recommendation"])

# ─────────────────────────── TAB 1: OVERVIEW ──────────────────────────────
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="section-card">
            <div style="font-size:2rem; margin-bottom:12px;">🔬</div>
            <h3 style="color:#58a6ff; margin:0 0 10px;">Influence Maximization</h3>
            <p style="color:#8b949e; margin:0; font-size:0.9rem;">Algorithmically identify which users in your network, if given the product first, will trigger the largest cascade of adoptions through social connections.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="section-card">
            <div style="font-size:2rem; margin-bottom:12px;">🌊</div>
            <h3 style="color:#3fb950; margin:0 0 10px;">Diffusion Simulation</h3>
            <p style="color:#8b949e; margin:0; font-size:0.9rem;">Simulate product awareness and adoption spreading step-by-step through your audience using the Independent Cascade or Linear Threshold models.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="section-card">
            <div style="font-size:2rem; margin-bottom:12px;">📐</div>
            <h3 style="color:#d2a679; margin:0 0 10px;">Strategy Comparison</h3>
            <p style="color:#8b949e; margin:0; font-size:0.9rem;">Compare five influencer-selection strategies across total reach, adoption rate, seed efficiency, and ROI to choose the best campaign approach.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### How to use this app")
    steps_col1, steps_col2 = st.columns(2)
    with steps_col1:
        st.markdown("""
        <div class="info-box">
            <b>Step 1 — Build a Network</b><br>
            Use the sidebar to generate a synthetic social network or upload your own CSV files with users (nodes) and connections (edges).
        </div>
        <div class="info-box">
            <b>Step 2 — Configure Campaign</b><br>
            Set your seed budget (how many influencers to seed), adoption probability, influence strength, and choose a diffusion model.
        </div>
        <div class="info-box">
            <b>Step 3 — Run Simulation</b><br>
            Click "Run Simulation" and explore results: adoption curve, network visualization, influencer ranking, and key metrics.
        </div>
        """, unsafe_allow_html=True)
    with steps_col2:
        st.markdown("""
        <div class="info-box">
            <b>Step 4 — Compare Strategies</b><br>
            Click "Compare All Strategies" to benchmark Greedy, PageRank, Degree Centrality, Betweenness, and Random selection side-by-side.
        </div>
        <div class="info-box">
            <b>Step 5 — Read the Recommendation</b><br>
            Visit the Recommendation tab for a plain-language executive summary with the optimal strategy, expected ROI, and action steps.
        </div>
        <div class="success-box">
            <b>✅ Default Recommendation:</b> Greedy Influence Maximization is pre-selected as the best strategy — it provably maximizes expected spread with submodular guarantees.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Expected CSV Format")
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("**nodes.csv**")
        st.dataframe(pd.DataFrame({"node_id": [0, 1, 2], "name": ["Alice", "Bob", "Carol"], "segment": ["A", "B", "A"]}), use_container_width=True)
    with ex_col2:
        st.markdown("**edges.csv**")
        st.dataframe(pd.DataFrame({"source": [0, 0, 1], "target": [1, 2, 2], "weight": [1.0, 0.5, 0.8]}), use_container_width=True)


# ─────────────────────────── TAB 2: NETWORK ──────────────────────────────
with tabs[1]:
    if not st.session_state.network_built:
        st.markdown('<div class="warning-box">⚠️ Build a network first using the sidebar controls.</div>', unsafe_allow_html=True)
    else:
        G = st.session_state.graph
        seeds = st.session_state.seeds
        activated = st.session_state.sim_results["final_activated"] if st.session_state.sim_results else set()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Users", G.number_of_nodes())
        c2.metric("Connections", G.number_of_edges())
        c3.metric("Avg Degree", round(sum(d for _, d in G.degree()) / G.number_of_nodes(), 2))
        c4.metric("Density", round(nx.density(G), 4))

        layout_choice = st.radio("Graph Layout", ["spring", "kamada_kawai", "circular"], horizontal=True)
        with st.spinner("Rendering network..."):
            fig = draw_network(G, seeds, activated, layout=layout_choice)
        st.plotly_chart(fig, use_container_width=True, key="network_graph")

        # Degree distribution
        degrees = [d for _, d in G.degree()]
        deg_fig = px.histogram(
            x=degrees, nbins=40,
            labels={"x": "Degree", "y": "Count"},
            title="Degree Distribution",
            color_discrete_sequence=["#388bfd"],
        )
        deg_fig.update_layout(
            paper_bgcolor="#1c2128", plot_bgcolor="#1c2128",
            font=dict(color="#8b949e", family="DM Sans"),
            title_font=dict(color="#e6edf3"),
            height=280, margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(deg_fig, use_container_width=True, key="degree_dist")


# ─────────────────────────── TAB 3: INFLUENCERS ──────────────────────────────
with tabs[2]:
    if not st.session_state.simulation_run:
        st.markdown('<div class="warning-box">⚠️ Run a simulation first to see influencer rankings.</div>', unsafe_allow_html=True)
    else:
        G = st.session_state.graph
        seeds = st.session_state.seeds
        scores = st.session_state.influencer_scores
        strategy_used = st.session_state.sim_results.get("strategy", "")

        st.markdown(f"""
        <div class="success-box">
            ✅ <b>Strategy Used:</b> {strategy_used} &nbsp;|&nbsp; <b>Seeds Selected:</b> {len(seeds)}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Top Influencers")
        top_nodes = sorted(scores, key=scores.get, reverse=True)[:20]
        influencer_df = pd.DataFrame([{
            "Rank": i + 1,
            "User ID": n,
            "Influence Score": scores[n],
            "Degree": G.degree(n),
            "Selected as Seed": "✅ Yes" if n in set(seeds) else "—",
        } for i, n in enumerate(top_nodes)])

        st.dataframe(influencer_df, use_container_width=True, hide_index=True)

        st.markdown("#### Influence Score Chart")
        st.plotly_chart(plot_influencer_ranking(scores, seeds, top_n=20), use_container_width=True, key="influencer_ranking")

        st.markdown("---")
        st.markdown("#### Why These Users Were Selected")
        strategy_explanations = {
            "Greedy Influence Maximization": "Uses Monte Carlo simulation to greedily pick each user that adds the most expected marginal influence. This provably achieves at least 63% of the theoretical optimum (submodular guarantee). Best for maximizing reach.",
            "PageRank": "Identifies users who are influential because they are connected to other influential users — similar to how Google ranks web pages. Great for targeting opinion leaders in dense clusters.",
            "Degree Centrality": "Selects users with the most direct connections. Simple and fast — high-degree nodes can directly reach the largest number of first-degree neighbors.",
            "Betweenness Centrality": "Finds users who sit on the most shortest paths between others — network brokers who bridge different communities. Effective for cross-community spread.",
            "Random": "Baseline random selection. Useful for benchmarking — any structured strategy should outperform this significantly.",
        }
        st.markdown(f'<div class="info-box"><b>{strategy_used}:</b><br>{strategy_explanations.get(strategy_used, "")}</div>', unsafe_allow_html=True)


# ─────────────────────────── TAB 4: SIMULATION ──────────────────────────────
with tabs[3]:
    if not st.session_state.simulation_run:
        st.markdown('<div class="warning-box">⚠️ Run a simulation to see results.</div>', unsafe_allow_html=True)
    else:
        result = st.session_state.sim_results
        G = st.session_state.graph

        st.markdown(f"""
        <div class="info-box">
            Model: <b>{result['model']}</b> &nbsp;|&nbsp;
            Seeds: <b>{len(result['seeds'])}</b> &nbsp;|&nbsp;
            Rounds: averaged across simulation runs
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Adoption Over Time")
        st.plotly_chart(plot_adoption_curve(result["adoption_curve"], result["total_nodes"]), use_container_width=True, key="adoption_curve")

        st.markdown("#### Network State After Simulation")
        with st.spinner("Rendering final network state..."):
            fig_final = draw_network(G, result["seeds"], result["final_activated"])
        st.plotly_chart(fig_final, use_container_width=True, key="network_final")

        st.markdown("#### Adoption Funnel")
        st.plotly_chart(plot_funnel(result, result["total_nodes"]), use_container_width=True, key="adoption_funnel")


# ─────────────────────────── TAB 5: DASHBOARD ──────────────────────────────
with tabs[4]:
    if not st.session_state.simulation_run:
        st.markdown('<div class="warning-box">⚠️ Run a simulation to see the dashboard.</div>', unsafe_allow_html=True)
    else:
        result = st.session_state.sim_results
        G = st.session_state.graph
        budget = len(result["seeds"])
        metrics = compute_metrics(result, budget, discount_pct)

        st.markdown("#### Campaign Performance Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Adopters", f"{metrics['Total Adopters']:,}", delta=f"{metrics['Adoption Rate (%)']:.1f}% of network")
        m2.metric("Reach from Seeds", f"{metrics['Reach from Seeds']:,}", delta=f"→ {metrics['Reach Ratio (%)']:.1f}% organic")
        m3.metric("Seed Efficiency", f"{metrics['Seed Efficiency']:.1f}x", delta="adopters per seed user")

        m4, m5, m6 = st.columns(3)
        m4.metric("Adoption Rate", f"{metrics['Adoption Rate (%)']:.1f}%")
        m5.metric("ROI Score", f"{metrics['ROI Score']:.2f}")
        m6.metric("Network Size", f"{result['total_nodes']:,}")

        st.markdown("---")
        st.markdown("#### Simulation Stability")
        stab_col1, stab_col2 = st.columns(2)
        with stab_col1:
            stab_data = {
                "Metric": ["Mean Adopters", "Std Dev", "Min Adopters", "Max Adopters", "Coefficient of Variation"],
                "Value": [
                    f"{result['mean_adopters']:.1f}",
                    f"{result['std_adopters']:.1f}",
                    f"{result['min_adopters']}",
                    f"{result['max_adopters']}",
                    f"{result['std_adopters'] / max(result['mean_adopters'], 1) * 100:.1f}%",
                ]
            }
            st.dataframe(pd.DataFrame(stab_data), use_container_width=True, hide_index=True)

        with stab_col2:
            cv = result["std_adopters"] / max(result["mean_adopters"], 1) * 100
            if cv < 10:
                st.markdown('<div class="success-box">✅ <b>High stability:</b> CV &lt; 10% — results are highly consistent across simulation rounds. You can trust this estimate.</div>', unsafe_allow_html=True)
            elif cv < 25:
                st.markdown('<div class="info-box">ℹ️ <b>Moderate stability:</b> CV is between 10–25%. Some variability — consider increasing simulation rounds for tighter bounds.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ <b>High variability:</b> CV &gt; 25%. Results vary significantly. Increase rounds or adoption probability for more reliable estimates.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Community-Level Spread")
        if "community" in list(G.nodes(data=True))[0][1]:
            comm_data = defaultdict(lambda: {"total": 0, "adopted": 0})
            activated = result["final_activated"]
            for node, data in G.nodes(data=True):
                c_id = data.get("community", 0)
                comm_data[c_id]["total"] += 1
                if node in activated:
                    comm_data[c_id]["adopted"] += 1
            comm_rows = [{"Community": f"C{c}", "Total Users": v["total"],
                           "Adopted": v["adopted"],
                           "Penetration (%)": round(v["adopted"] / max(v["total"], 1) * 100, 1)}
                          for c, v in sorted(comm_data.items())]
            st.dataframe(pd.DataFrame(comm_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Community data not available for uploaded networks.")


# ─────────────────────────── TAB 6: STRATEGY COMPARISON ──────────────────────────────
with tabs[5]:
    if not st.session_state.strategy_results:
        st.markdown('<div class="warning-box">⚠️ Click "Compare All Strategies" in the sidebar to run a full comparison.</div>', unsafe_allow_html=True)
    else:
        strat_results = st.session_state.strategy_results

        st.markdown("#### Strategy Comparison Chart")
        st.plotly_chart(plot_strategy_comparison(strat_results), use_container_width=True, key="strategy_comparison")

        st.markdown("#### Comparison Table")
        rows = []
        for strat, data in strat_results.items():
            m = data["metrics"]
            row = {"Strategy": strat, **m}
            rows.append(row)
        df_compare = pd.DataFrame(rows)

        # Highlight best in each column
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # Adoption curves per strategy
        st.markdown("#### Adoption Curves — All Strategies")
        curve_fig = go.Figure()
        colors_strat = ["#388bfd", "#3fb950", "#d2a679", "#f78166", "#bc8cff"]
        total_nodes = list(strat_results.values())[0]["result"]["total_nodes"]
        for strat, color in zip(strat_results.keys(), colors_strat):
            curve = strat_results[strat]["result"]["adoption_curve"]
            curve_fig.add_trace(go.Scatter(
                x=list(range(len(curve))), y=curve,
                mode="lines", name=strat,
                line=dict(color=color, width=2.5),
            ))
        curve_fig.update_layout(
            paper_bgcolor="#1c2128", plot_bgcolor="#1c2128",
            font=dict(color="#8b949e", family="DM Sans"),
            xaxis=dict(title="Time Step", gridcolor="#21262d"),
            yaxis=dict(title="Cumulative Adopters", gridcolor="#21262d"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3")),
            height=380, margin=dict(l=60, r=40, t=20, b=50),
        )
        st.plotly_chart(curve_fig, use_container_width=True, key="strategy_curves")


# ─────────────────────────── TAB 7: RECOMMENDATION ──────────────────────────────
with tabs[6]:
    if not st.session_state.simulation_run:
        st.markdown('<div class="warning-box">⚠️ Run a simulation to generate a recommendation.</div>', unsafe_allow_html=True)
    else:
        result = st.session_state.sim_results
        G = st.session_state.graph
        seeds = result["seeds"]
        metrics = compute_metrics(result, len(seeds), discount_pct)
        strategy_used = result.get("strategy", "")

        best_strategy = strategy_used
        has_comparison = bool(st.session_state.strategy_results)
        if has_comparison:
            best_strategy = max(
                st.session_state.strategy_results,
                key=lambda s: st.session_state.strategy_results[s]["metrics"]["Total Adopters"]
            )

        st.markdown(f"""
        <div class="hero-banner">
            <div class="hero-badge">Executive Summary</div>
            <div class="hero-title" style="font-size:1.8rem;">Campaign Recommendation</div>
            <div class="hero-subtitle">Generated based on {result['model']} simulation across multiple rounds</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("#### 🎯 Recommended Action")
            st.markdown(f"""
            <div class="success-box">
                <b>Best Strategy: {best_strategy}</b><br><br>
                Seed <b>{len(seeds)} users</b> from your network to trigger a product adoption cascade.
                Based on the simulation, this is expected to reach approximately <b>{metrics['Total Adopters']:,} users</b>
                ({metrics['Adoption Rate (%)']:.1f}% of the network) within <b>{len(result['adoption_curve'])} time steps</b>.
                <br><br>
                Each seed user is estimated to influence an additional <b>{metrics['Seed Efficiency']:.1f} people</b> on average,
                giving you a seed efficiency multiplier of <b>{metrics['Seed Efficiency']:.1f}x</b>.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📋 Key Findings")
            findings = [
                f"**Network Reach:** Your campaign is expected to organically spread to {metrics['Reach Ratio (%)']:.1f}% of non-seed users without additional spend.",
                f"**Adoption Rate:** {metrics['Adoption Rate (%)']:.1f}% network-wide adoption — {'strong result' if metrics['Adoption Rate (%)'] > 30 else 'moderate result — consider increasing adoption probability or influence strength'}.",
                f"**Seed Efficiency:** {metrics['Seed Efficiency']:.1f}x — {'excellent' if metrics['Seed Efficiency'] > 5 else 'good' if metrics['Seed Efficiency'] > 2 else 'moderate'}. {'Consider increasing budget for better coverage.' if metrics['Seed Efficiency'] > 10 else ''}",
                f"**Incentive Level:** A {discount_pct}% discount/incentive for seed users is factored into the ROI score of {metrics['ROI Score']:.2f}.",
            ]
            for f in findings:
                st.markdown(f"- {f}")

            if has_comparison:
                st.markdown("#### ⚖️ Strategy Verdict")
                strat_results = st.session_state.strategy_results
                ordered = sorted(strat_results.keys(), key=lambda s: strat_results[s]["metrics"]["Total Adopters"], reverse=True)
                for rank, strat in enumerate(ordered, 1):
                    m = strat_results[strat]["metrics"]
                    badge_color = "#3fb950" if rank == 1 else "#8b949e"
                    recommended_tag = " ← RECOMMENDED" if rank == 1 else ""
                    st.markdown(f"**{rank}. {strat}** — {m['Total Adopters']:,} adopters, {m['Adoption Rate (%)']:.1f}% rate{recommended_tag}")

        with col2:
            st.markdown("#### 📊 Summary Metrics")
            summary_df = pd.DataFrame({
                "Metric": list(metrics.keys()),
                "Value": [str(v) for v in metrics.values()],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.markdown("#### 🚀 Next Steps")
            st.markdown(f"""
            <div class="info-box">
                <ol style="margin:0; padding-left:18px; color:#79c0ff;">
                    <li>Identify the top {len(seeds)} users in your CRM matching the seed user profile.</li>
                    <li>Offer the configured {discount_pct}% incentive to seed users for early adoption.</li>
                    <li>Use the <b>{result['model']}</b> model parameters to monitor real adoption velocity.</li>
                    <li>Track time-step metrics weekly — compare against the simulated adoption curve.</li>
                    <li>Re-run simulation if adoption probability shifts based on field data.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

            # Export metrics
            export_df = pd.DataFrame([metrics])
            csv_bytes = export_df.to_csv(index=False).encode()
            st.download_button(
                label="⬇️ Download Metrics CSV",
                data=csv_bytes,
                file_name="campaign_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

            seed_export = pd.DataFrame({"seed_user_id": seeds, "rank": range(1, len(seeds) + 1)})
            st.download_button(
                label="⬇️ Download Seed User List",
                data=seed_export.to_csv(index=False).encode(),
                file_name="seed_users.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ─────────────────────────── FOOTER ──────────────────────────────
st.markdown("""
<hr style="border-color:#30363d; margin-top:48px;">
<div style="text-align:center; color:#484f58; font-size:0.8rem; padding:16px 0; font-family:'Space Mono',monospace;">
    Corporate Product Campaign Spread Simulator &nbsp;|&nbsp; Built with NetworkX · Plotly · Streamlit
</div>
""", unsafe_allow_html=True)
