# campaign_spread_simulator_small.py
# Run with: streamlit run campaign_spread_simulator_small.py
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import random, math

st.set_page_config(page_title="Campaign Spread Simulator", layout="wide")

# Helpers
def build_network(n_users=200, edge_density=0.05, communities=3, seed=42):
    random.seed(seed); np.random.seed(seed)
    if communities > 1:
        sizes = [n_users // communities] * communities
        sizes[-1] += n_users - sum(sizes)
        p_in = min(0.6, edge_density * 2)
        p_out = edge_density * 0.2
        p_mat = [[p_in if i==j else p_out for j in range(communities)] for i in range(communities)]
        G = nx.stochastic_block_model(sizes, p_mat, seed=seed)
    else:
        G = nx.erdos_renyi_graph(n_users, edge_density, seed=seed)
    for i, n in enumerate(G.nodes()):
        G.nodes[n]["community"] = i % communities
    return nx.convert_node_labels_to_integers(G)

def score_influencers(G, strategy="PageRank", budget=5):
    if strategy=="Degree Centrality":
        s = nx.degree_centrality(G)
    elif strategy=="Betweenness Centrality":
        s = nx.betweenness_centrality(G)
    else:
        s = nx.pagerank(G)
    seeds = sorted(s, key=s.get, reverse=True)[:budget]
    return seeds, s

def run_ic(G, seeds, p=0.1, steps=10):
    activated = set(seeds); history=[set(seeds)]
    for _ in range(steps):
        new=set()
        for n in history[-1]:
            for neigh in G.neighbors(n):
                if neigh not in activated and random.random()<p:
                    new.add(neigh)
        if not new: break
        activated|=new; history.append(new)
    return activated, history

def plot_network(G, seeds, activated):
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y=[],[]
    for u,v in G.edges():
        x0,y0=pos[u]; x1,y1=pos[v]
        edge_x+=[x0,x1,None]; edge_y+=[y0,y1,None]
    edge_trace=go.Scatter(x=edge_x,y=edge_y,mode="lines",line=dict(width=0.5,color="#444"),hoverinfo="none")
    node_x=[pos[n][0] for n in G.nodes()]; node_y=[pos[n][1] for n in G.nodes()]
    colors=[]
    for n in G.nodes():
        if n in seeds: colors.append("#ff7b72")
        elif n in activated: colors.append("#2ecc71")
        else: colors.append("#95a5a6")
    node_trace=go.Scatter(x=node_x,y=node_y,mode="markers",
                          marker=dict(size=8,color=colors,line=dict(width=0.5,color="#000")))
    fig=go.Figure(data=[edge_trace,node_trace])
    fig.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",xaxis=dict(visible=False),
                      yaxis=dict(visible=False),margin=dict(l=0,r=0,t=0,b=0),height=500)
    return fig

def plot_adoption_curve(hist, total_nodes):
    cum=[len(set.union(*hist[:i+1])) for i in range(len(hist))]
    pct=[c/total_nodes*100 for c in cum]
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=cum,mode="lines+markers",name="Adopters",line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(y=pct,mode="lines",name="% Adoption",line=dict(color="#2ecc71",dash="dot"),yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying="y",side="right",title="%"),xaxis_title="Step",
                      paper_bgcolor="#111",plot_bgcolor="#111",font=dict(color="#ddd"))
    return fig

def compute_metrics(G, seeds, activated):
    total = G.number_of_nodes()
    adopters = len(activated)
    rate = adopters / total * 100
    eff = (adopters - len(seeds)) / len(seeds)
    return {"Total Adopters": adopters, "Adoption Rate (%)": round(rate,1), "Seed Efficiency": round(eff,2)}

# Sidebar
st.sidebar.title("🎯 Campaign Settings")
n_users = st.sidebar.slider("Users",50,1000,200,step=50)
density = st.sidebar.slider("Edge Density",0.01,0.2,0.05)
communities = st.sidebar.slider("Communities",1,6,3)
budget = st.sidebar.slider("Seed Budget",1,30,5)
strategy = st.sidebar.selectbox("Seed Strategy",["PageRank","Degree Centrality","Betweenness Centrality"])
adopt_p = st.sidebar.slider("Adoption Probability",0.01,0.5,0.1)
steps = st.sidebar.slider("Simulation Steps",5,30,10)
run_button = st.sidebar.button("▶ Run Simulation")

# Main UI
st.title("🚀 Corporate Product Campaign Spread Simulator (Compact Version)")
st.write("Model viral product adoption in a social network with network effects.")

if run_button:
    with st.spinner("Building network..."):
        G = build_network(n_users,density,communities)
    st.success(f"Network ready ({G.number_of_nodes()} users, {G.number_of_edges()} edges).")

    with st.spinner("Scoring users..."):
        seeds,scores = score_influencers(G,strategy,budget)
    with st.spinner("Running simulation..."):
        activated,history = run_ic(G,seeds,adopt_p,steps)

    met = compute_metrics(G,seeds,activated)

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Adopters",met["Total Adopters"])
    c2.metric("Adoption Rate",f"{met['Adoption Rate (%)']}%")
    c3.metric("Seed Efficiency",f"{met['Seed Efficiency']}x")

    st.subheader("Network Visualization")
    st.plotly_chart(plot_network(G,seeds,activated),use_container_width=True)

    st.subheader("Adoption Curve")
    st.plotly_chart(plot_adoption_curve(history,G.number_of_nodes()),use_container_width=True)

    st.subheader("Top Influencers")
    top_nodes=sorted(scores,key=scores.get,reverse=True)[:20]
    df=pd.DataFrame({"User":top_nodes,"Score":[round(scores[n]*100,2) for n in top_nodes],
                     "Seed":["Yes" if n in seeds else "No" for n in top_nodes]})
    st.dataframe(df,use_container_width=True,hide_index=True)

    st.download_button("⬇️ Export Metrics",pd.DataFrame([met]).to_csv(index=False), "metrics.csv")
else:
    st.info("Adjust settings in sidebar, then click **Run Simulation** to begin.")

st.markdown("---")
st.caption("© 2024 Compact Campaign Spread Simulator | NetworkX · Plotly · Streamlit")

