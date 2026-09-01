"""
Task 5: Multi-Asset Connectivity Analysis
Builds a directed graph (site/building hierarchy + explicit connectivity
edges), analyzes dependencies, simulates failure propagation, and runs a
data-quality audit (orphans, duplicates, invalid references).
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
DATA, FIG, MODELS = ROOT / "data", ROOT / "figures", ROOT / "models"

assets = pd.read_csv(f"{DATA}/asset_metadata.csv")
conn = pd.read_csv(f"{DATA}/asset_connectivity.csv")

valid_ids = set(assets["asset_id"])

# --------------------------------------------------------------------
# 1. Data quality audit
# --------------------------------------------------------------------
dq = {}

# duplicate connections (identical source/target/type)
dup_mask = conn.duplicated(subset=["source_asset_id", "target_asset_id", "connection_type"], keep=False)
dq["duplicate_connections"] = conn[dup_mask].drop_duplicates().to_dict("records")

# invalid parent references in asset_metadata (parent not an existing asset)
invalid_parents = assets[
    assets["parent_asset_id"].notna() & ~assets["parent_asset_id"].isin(valid_ids)
]
dq["invalid_parent_references"] = invalid_parents[["asset_id", "asset_name", "parent_asset_id"]].to_dict("records")

# invalid connectivity edges (endpoints that don't exist)
invalid_edges = conn[
    ~conn["source_asset_id"].isin(valid_ids) | ~conn["target_asset_id"].isin(valid_ids)
]
dq["invalid_connectivity_edges"] = invalid_edges.to_dict("records")

# orphan assets: no parent AND not referenced as a target/source in connectivity,
# excluding asset types that are legitimately top-level (Chiller, HVAC, EnergyMeter)
referenced = set(conn["source_asset_id"]) | set(conn["target_asset_id"])
top_level_types = {"Chiller", "HVAC", "EnergyMeter"}
orphans = assets[
    assets["parent_asset_id"].isna()
    & ~assets["asset_id"].isin(referenced)
    & ~assets["asset_type"].isin(top_level_types)
]
dq["orphan_assets"] = orphans[["asset_id", "asset_name", "asset_type"]].to_dict("records")

# missing relationships: AHUs/Pumps/Sensors with no parent at all (structurally should have one)
should_have_parent = {"AHU", "Pump", "EnvSensor"}
missing_rel = assets[
    assets["asset_type"].isin(should_have_parent) & assets["parent_asset_id"].isna()
]
dq["missing_relationships"] = missing_rel[["asset_id", "asset_name", "asset_type"]].to_dict("records")

print("=" * 70)
print("DATA QUALITY AUDIT")
print("=" * 70)
for k, v in dq.items():
    print(f"\n{k} ({len(v)}):")
    for item in v:
        print("  ", item)

with open(f"{MODELS}/connectivity_data_quality.json", "w") as f:
    json.dump(dq, f, indent=2, default=str)

# --------------------------------------------------------------------
# 2. Build the graph (clean version: drop invalid edges/refs for analysis)
# --------------------------------------------------------------------
G = nx.DiGraph()
for _, row in assets.iterrows():
    G.add_node(row["asset_id"], name=row["asset_name"], type=row["asset_type"],
               site=row["site_id"], building=row["building_id"])

# hierarchy edges (parent -> child), skipping invalid parent refs
for _, row in assets.iterrows():
    if pd.notna(row["parent_asset_id"]) and row["parent_asset_id"] in valid_ids:
        G.add_edge(row["parent_asset_id"], row["asset_id"], kind="hierarchy")

# explicit connectivity edges, skipping invalid endpoints, de-duplicated
conn_clean = conn.dropna(subset=["source_asset_id", "target_asset_id"])
conn_clean = conn_clean[
    conn_clean["source_asset_id"].isin(valid_ids) & conn_clean["target_asset_id"].isin(valid_ids)
].drop_duplicates(subset=["source_asset_id", "target_asset_id", "connection_type"])
for _, row in conn_clean.iterrows():
    G.add_edge(row["source_asset_id"], row["target_asset_id"],
               kind=row["connection_type"], weight=row["relationship_strength"])

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# isolated assets (no connections at all, in a fully clean sense)
isolated = [n for n in G.nodes if G.degree(n) == 0]
print("Fully isolated assets (no edges at all):", isolated)

# --------------------------------------------------------------------
# 3. Example queries (as requested in the brief)
# --------------------------------------------------------------------
def downstream_of(asset_id):
    """All assets reachable FROM asset_id (i.e. depend on / are affected by it)."""
    if asset_id not in G:
        return []
    return sorted(nx.descendants(G, asset_id))

def assets_connected_to(asset_id):
    return sorted(set(G.predecessors(asset_id)) | set(G.successors(asset_id))) if asset_id in G else []

def assets_under_site(site_id):
    return sorted(n for n, d in G.nodes(data=True) if d.get("site") == site_id)

def isolated_assets():
    return isolated

sample_chiller = assets[assets["asset_type"] == "Chiller"]["asset_id"].iloc[0]
sample_ahu = assets[assets["asset_type"] == "AHU"]["asset_id"].iloc[1]
sample_site = assets["site_id"].iloc[0]

query_results = {
    f"assets_connected_to({sample_chiller})": assets_connected_to(sample_chiller),
    f"downstream_impacted_by({sample_ahu})_failure": downstream_of(sample_ahu),
    f"assets_under_site({sample_site})": assets_under_site(sample_site),
    "isolated_assets": isolated_assets(),
}
print("\n" + "=" * 70)
print("EXAMPLE QUERIES")
print("=" * 70)
for q, r in query_results.items():
    print(f"\n{q}:")
    print("  ", r)

with open(f"{MODELS}/connectivity_example_queries.json", "w") as f:
    json.dump(query_results, f, indent=2)

# --------------------------------------------------------------------
# 4. Failure impact analysis for a Chiller and a Pump
# --------------------------------------------------------------------
def failure_impact_report(asset_id):
    downstream = downstream_of(asset_id)
    downstream_types = assets[assets["asset_id"].isin(downstream)]["asset_type"].value_counts().to_dict()
    return {
        "failed_asset": asset_id,
        "n_downstream_assets_impacted": len(downstream),
        "downstream_assets": downstream,
        "downstream_asset_types": downstream_types,
    }

chiller_impact = failure_impact_report(sample_chiller)
pump_id = assets[assets["asset_type"] == "Pump"]["asset_id"].iloc[0]
pump_impact = failure_impact_report(pump_id)

print("\n" + "=" * 70)
print("FAILURE IMPACT ANALYSIS")
print("=" * 70)
print(f"\nIf {sample_chiller} (Chiller) fails:")
print(json.dumps(chiller_impact, indent=2))
print(f"\nIf {pump_id} (Pump) fails:")
print(json.dumps(pump_impact, indent=2))

with open(f"{MODELS}/connectivity_failure_impact.json", "w") as f:
    json.dump({"chiller_failure": chiller_impact, "pump_failure": pump_impact}, f, indent=2)

# --------------------------------------------------------------------
# 5. Visualization: one site's hierarchy graph
# --------------------------------------------------------------------
site_nodes = [n for n, d in G.nodes(data=True) if d.get("site") == sample_site]
subG = G.subgraph(site_nodes)

type_colors = {
    "Chiller": "#c0504d", "AHU": "#4f81bd", "Pump": "#9bbb59",
    "HVAC": "#8064a2", "EnergyMeter": "#f79646", "EnvSensor": "#4bacc6",
}
node_colors = [type_colors.get(subG.nodes[n]["type"], "#888") for n in subG.nodes]
labels = {n: subG.nodes[n]["type"] + "\n" + n[-4:] for n in subG.nodes}

fig, ax = plt.subplots(figsize=(13, 9))
pos = nx.spring_layout(subG, seed=42, k=1.4)
nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=1400, ax=ax)
nx.draw_networkx_edges(subG, pos, arrows=True, arrowsize=15, ax=ax, connectionstyle="arc3,rad=0.05")
nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8, ax=ax)
ax.set_title(f"Figure 12 — Asset Connectivity Graph: {sample_site}", fontsize=14)
ax.axis("off")

import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=c, label=t) for t, c in type_colors.items()]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(f"{FIG}/fig12_connectivity_graph.png", dpi=130)
plt.close(fig)

print("\nSaved graph visualization + connectivity JSON reports.")
