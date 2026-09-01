"""
Nectar Intelligent Facilities Platform — Operations Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import joblib
import json
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
sys.path.insert(0, os.path.join(BASE, "src"))
from predictive_features import engineer_predictive_features

st.set_page_config(page_title="Nectar Facilities Platform", layout="wide")

@st.cache_data
def load_data():
    telem = pd.read_csv(f"{DATA}/telemetry_with_anomalies.csv", parse_dates=["timestamp"])
    assets = pd.read_csv(f"{DATA}/asset_metadata.csv")
    conn = pd.read_csv(f"{DATA}/asset_connectivity.csv")
    return telem, assets, conn

telem, assets, conn = load_data()

st.title("Nectar Intelligent Facilities Platform")
st.caption("Site overview • Asset health • Failure risk • Energy trends • Anomalies • Connectivity")

# ---------------- Sidebar filters ----------------
sites = sorted(telem["site_id"].unique())
site_sel = st.sidebar.selectbox("Site", ["All"] + sites)
df = telem if site_sel == "All" else telem[telem["site_id"] == site_sel]

# ---------------- Row 1: Site overview KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets", assets["asset_id"].nunique() if site_sel == "All" else assets[assets.site_id==site_sel].asset_id.nunique())
c2.metric("Total Energy (kWh)", f"{df['power_consumption'].sum():,.0f}")
c3.metric("Fault Events", int(df["fault_flag"].sum()))
c4.metric("Anomalies Flagged", int(df["any_anomaly"].sum()) if "any_anomaly" in df else "n/a")

# ---------------- Row 2: Energy trend ----------------
st.subheader("Energy Trends")
daily = df.groupby([pd.Grouper(key="timestamp", freq="D"), "site_id"])["power_consumption"].sum().reset_index()
fig = px.line(daily, x="timestamp", y="power_consumption", color="site_id", title="Daily Total Energy Consumption")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Row 3: Asset health / anomalies ----------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Anomaly Rate by Asset Type")
    if "any_anomaly" in df.columns:
        rate = df.merge(assets[["asset_id","asset_type"]], on="asset_id", how="left", suffixes=("","_m"))
        rate = rate.groupby("asset_type")["any_anomaly"].mean().reset_index()
        st.plotly_chart(px.bar(rate, x="asset_type", y="any_anomaly", title="Anomaly rate"), use_container_width=True)
with col2:
    st.subheader("Fault Rate by Asset Type")
    fr = df.merge(assets[["asset_id","asset_type"]], on="asset_id", how="left", suffixes=("","_m"))
    fr = fr.groupby("asset_type")["fault_flag"].mean().reset_index()
    st.plotly_chart(px.bar(fr, x="asset_type", y="fault_flag", title="Historical fault rate"), use_container_width=True)

# ---------------- Row 4: Failure prediction (same causal features as training/API) ----------------
st.subheader("Failure Risk — Latest Reading per Asset")
try:
    model = joblib.load(os.path.join(MODELS, "predictive_maintenance_model.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS, "predictive_maintenance_features.joblib"))
    prep = joblib.load(os.path.join(MODELS, "predictive_maintenance_preprocessing.joblib"))
    threshold = float(prep.get("threshold", 0.5))
    medians = prep.get("medians", {})
    scored=[]
    # Score each asset using its complete available history so rolling features are real, not placeholders.
    for aid, hist in telem.sort_values("timestamp").groupby("asset_id"):
        h=hist.merge(assets[["asset_id","asset_type","capacity"]],on="asset_id",how="left",suffixes=("","_meta"))
        if "asset_type_meta" in h and "asset_type" not in h: h=h.rename(columns={"asset_type_meta":"asset_type"})
        h["asset_type"]=h["asset_type"].fillna("Unknown")
        x=engineer_predictive_features(h, medians=medians)
        for c in feature_cols:
            if c not in x: x[c]=0
        prob=float(model.predict_proba(x[feature_cols].tail(1))[:,1][0])
        scored.append({"asset_id":aid,"risk":prob})
    top_risk=pd.DataFrame(scored).sort_values("risk",ascending=False).head(10)
    st.caption(f"Threshold learned on training data: {threshold:.3f}. Risk uses historical telemetry for each asset.")
    st.plotly_chart(px.bar(top_risk,x="asset_id",y="risk",title="Top-10 highest risk assets (next 24 clock hours)"),use_container_width=True)
except Exception as e:
    st.info(f"Model scoring unavailable in this session: {e}")

# ---------------- Row 5: Connectivity graph ----------------
st.subheader("Asset Connectivity")
site_for_graph = site_sel if site_sel != "All" else sites[0]
sub_assets = assets[assets.site_id == site_for_graph]
sub_conn = conn[conn.source_asset_id.isin(sub_assets.asset_id) & conn.target_asset_id.isin(sub_assets.asset_id)]
G = nx.from_pandas_edgelist(sub_conn, "source_asset_id", "target_asset_id", create_using=nx.DiGraph())
pos = nx.spring_layout(G, seed=42)
edge_x, edge_y = [], []
for e in G.edges():
    x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
import plotly.graph_objects as go
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888")))
node_x = [pos[n][0] for n in G.nodes()]
node_y = [pos[n][1] for n in G.nodes()]
fig2.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=list(G.nodes()),
                           textposition="top center", marker=dict(size=14, color="#4f81bd")))
fig2.update_layout(title=f"Connectivity graph — {site_for_graph}", showlegend=False, height=500)
st.plotly_chart(fig2, use_container_width=True)

st.caption("Data is synthetic, generated for the Nectar Data Scientist Challenge assessment.")
