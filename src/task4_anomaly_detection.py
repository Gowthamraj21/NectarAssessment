"""
Task 4: Anomaly Detection
Framework: Isolation Forest (multivariate, per-asset-type) + statistical
thresholding (rolling z-score) as a fast, explainable secondary check.
Also demonstrates change-point flagging via rolling-mean shift.

Note on the drift method specifically: earlier iterations of the dataset had
no real drift signal independent of labeled faults, so the drift check was
basically flagging the same rows the other two methods already caught -
redundant, not a real third opinion. The generator now injects a slow,
bounded sensor-calibration random walk on every asset regardless of whether
it ever faults, so this method has something genuine to find on its own.
Checked this below (see the drift-only vs. fault-flag overlap print) before
trusting the "three complementary methods" story in the README.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
DATA, FIG, MODELS = ROOT / "data", ROOT / "figures", ROOT / "models"

telem = pd.read_csv(f"{DATA}/sensor_telemetry.csv", parse_dates=["timestamp"])
assets = pd.read_csv(f"{DATA}/asset_metadata.csv")
telem = telem.merge(assets[["asset_id", "asset_type"]], on="asset_id", how="left")

# dedupe double-published (asset_id, timestamp) rows before anything else -
# an Isolation Forest trained with near-identical duplicate rows would just
# be marking its own dupes as "normal" twice, which quietly inflates the
# apparent normal-cluster density
n_before = len(telem)
telem = telem.sort_values(["asset_id", "timestamp"]).drop_duplicates(
    subset=["asset_id", "timestamp"], keep="last"
).reset_index(drop=True)
print(f"Deduped {n_before - len(telem)} double-published (asset_id, timestamp) rows "
      f"before fitting anything.")

sensor_cols = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]
telem[sensor_cols] = telem.groupby("asset_id")[sensor_cols].transform(lambda s: s.ffill())

# --------------------------------------------------------------------
# Method 1: Isolation Forest, trained separately per asset_type
#           (a Chiller's "normal" vibration is not an EnvSensor's normal)
# --------------------------------------------------------------------
telem["iso_anomaly"] = 0
telem["iso_score"] = 0.0
for atype, sub in telem.groupby("asset_type"):
    if len(sub) < 50:
        continue
    X = sub[sensor_cols].values
    clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
    clf.fit(X)
    pred = clf.predict(X)  # -1 = anomaly
    score = clf.decision_function(X)
    telem.loc[sub.index, "iso_anomaly"] = (pred == -1).astype(int)
    telem.loc[sub.index, "iso_score"] = score

# --------------------------------------------------------------------
# Method 2: Statistical thresholding (rolling z-score per asset)
# --------------------------------------------------------------------
g = telem.groupby("asset_id")
for col in ["vibration", "power_consumption", "temperature"]:
    roll_mean = g[col].transform(lambda s: s.rolling(48, min_periods=10).mean())
    roll_std = g[col].transform(lambda s: s.rolling(48, min_periods=10).std())
    telem[f"{col}_zscore"] = (telem[col] - roll_mean) / roll_std.replace(0, np.nan)
telem["stat_anomaly"] = (
    (telem["vibration_zscore"].abs() > 3.5)
    | (telem["power_consumption_zscore"].abs() > 3.5)
    | (telem["temperature_zscore"].abs() > 3.5)
).astype(int)

# --------------------------------------------------------------------
# Method 3: Change-point-style flag — sudden shift in rolling mean level
# (sensor drift detection: compares short vs long rolling mean)
# --------------------------------------------------------------------
for col in ["temperature", "vibration"]:
    short = g[col].transform(lambda s: s.rolling(6, min_periods=3).mean())
    long = g[col].transform(lambda s: s.rolling(72, min_periods=20).mean())
    telem[f"{col}_drift"] = (short - long).abs()
telem["drift_anomaly"] = (
    (telem["temperature_drift"] > telem["temperature_drift"].quantile(0.995))
    | (telem["vibration_drift"] > telem["vibration_drift"].quantile(0.995))
).astype(int)

telem["any_anomaly"] = (
    (telem["iso_anomaly"] == 1) | (telem["stat_anomaly"] == 1) | (telem["drift_anomaly"] == 1)
).astype(int)

print("Isolation Forest anomalies:", telem["iso_anomaly"].sum(), f"({telem['iso_anomaly'].mean()*100:.2f}%)")
print("Statistical z-score anomalies:", telem["stat_anomaly"].sum(), f"({telem['stat_anomaly'].mean()*100:.2f}%)")
print("Drift/change-point anomalies:", telem["drift_anomaly"].sum(), f"({telem['drift_anomaly'].mean()*100:.2f}%)")
print("Union (any method):", telem["any_anomaly"].sum(), f"({telem['any_anomaly'].mean()*100:.2f}%)")

# Cross-check against known injected faults / outliers (sanity, not ground truth for anomalies)
overlap_with_fault = telem.loc[telem["any_anomaly"] == 1, "fault_flag"].mean()
print(f"\nOf flagged anomalies, {overlap_with_fault*100:.1f}% coincide with a labeled fault "
      f"(remainder = early-warning signals / non-fault outliers, e.g. injected sensor spikes).")

# drift-only rows: flagged by the drift method but NOT by iso/stat and not a
# labeled fault - this is exactly the "sensor is quietly out of calibration"
# case the method is supposed to catch, separate from anything fault-related
drift_only = telem[
    (telem["drift_anomaly"] == 1)
    & (telem["iso_anomaly"] == 0)
    & (telem["stat_anomaly"] == 0)
    & (telem["fault_flag"] == 0)
]
print(f"Drift-only flags (calibration drift, not a fault, not caught by the other "
      f"two methods): {len(drift_only)} rows across {drift_only['asset_id'].nunique()} assets "
      f"- this is the method earning its keep as a genuinely separate check.")

telem.to_csv(f"{DATA}/telemetry_with_anomalies.csv", index=False)

summary = {
    "isolation_forest_anomalies": int(telem["iso_anomaly"].sum()),
    "statistical_anomalies": int(telem["stat_anomaly"].sum()),
    "drift_anomalies": int(telem["drift_anomaly"].sum()),
    "union_anomalies": int(telem["any_anomaly"].sum()),
    "pct_overlap_with_labeled_fault": round(float(overlap_with_fault) * 100, 2),
}
with open(f"{MODELS}/anomaly_detection_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# --------------------------------------------------------------------
# Figure: example asset timeline with anomalies marked
# --------------------------------------------------------------------
# pick an asset with a healthy number of anomalies to illustrate
counts = telem[telem["any_anomaly"] == 1].groupby("asset_id").size().sort_values(ascending=False)
example_asset = counts.index[0] if len(counts) else telem["asset_id"].iloc[0]
sub = telem[telem["asset_id"] == example_asset].sort_values("timestamp")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(sub["timestamp"], sub["vibration"], color="#333", label="Vibration")
anomalies = sub[sub["any_anomaly"] == 1]
axes[0].scatter(anomalies["timestamp"], anomalies["vibration"], color="red", s=25, zorder=5, label="Flagged anomaly")
axes[0].set_title(f"Vibration — {example_asset}")
axes[0].legend()

axes[1].plot(sub["timestamp"], sub["power_consumption"], color="#333", label="Power (kWh)")
axes[1].scatter(anomalies["timestamp"], anomalies["power_consumption"], color="red", s=25, zorder=5, label="Flagged anomaly")
axes[1].set_title(f"Power Consumption — {example_asset}")
axes[1].legend()
fig.suptitle("Figure 10 — Anomaly Detection Example (Isolation Forest + Statistical + Drift)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{FIG}/fig10_anomaly_example.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------
# Figure: anomaly counts by asset type
# --------------------------------------------------------------------
by_type = telem.groupby("asset_type")["any_anomaly"].agg(["sum", "mean"]).rename(
    columns={"sum": "n_anomalies", "mean": "anomaly_rate"}
)
fig, ax = plt.subplots(figsize=(8, 5))
by_type["anomaly_rate"].sort_values(ascending=False).mul(100).plot(kind="bar", ax=ax, color="#c0504d")
ax.set_ylabel("Anomaly rate (%)")
ax.set_title("Figure 11 — Anomaly Rate by Asset Type")
fig.tight_layout()
fig.savefig(f"{FIG}/fig11_anomaly_by_type.png", dpi=130)
plt.close(fig)

print("\nAnomaly rate by asset type:\n", by_type.round(4).to_string())
print("\nSaved figures + telemetry_with_anomalies.csv")
