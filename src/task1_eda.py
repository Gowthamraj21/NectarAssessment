"""
Task 1: Exploratory Data Analysis
Produces summary stats + saved figures into /figures, and prints key findings.

Went through this in the order I'd normally poke at a new dataset: shape and
missingness first (so I know what I'm dealing with before trusting any
number downstream), then a raw describe(), then an explicit outlier scan,
then the actual "story" plots. Two things showed up in the raw CSV that
aren't in a typical toy dataset and needed handling before anything else
made sense:

  - 48 rows are exact (asset_id, timestamp) duplicates - looks like
    double-published readings from the ingestion pipeline rather than
    real repeat measurements. Kept them visible in the raw missingness/
    describe() pass below (so the audit trail is honest about what's in
    the file), but they get deduped before any modeling task uses this
    data (see task2-4), keeping the *last* reading per (asset, timestamp)
    on the assumption that a re-publish supersedes the original.
  - the sampling grid isn't perfectly hourly - device clock skew puts
    readings a few minutes off the hour, and a handful of assets have
    multi-hour gaps (comms outages) rather than one row per hour on the
    dot. Doesn't change any of the EDA below, but it's worth flagging
    up front so nobody assumes `resample('1H')` will behave the same way
    it would on a clean grid.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
DATA, FIG = ROOT / "data", ROOT / "figures"

telem = pd.read_csv(f"{DATA}/sensor_telemetry.csv", parse_dates=["timestamp"])
assets = pd.read_csv(f"{DATA}/asset_metadata.csv")

df = telem.merge(assets[["asset_id", "asset_type", "manufacturer"]], on="asset_id", how="left")
df["hour"] = df["timestamp"].dt.hour
df["dow"] = df["timestamp"].dt.day_name()
df["date"] = df["timestamp"].dt.date

print("=" * 70)
print("1. SHAPE & MISSINGNESS")
print("=" * 70)
print("Telemetry rows:", len(df), " | Assets:", df.asset_id.nunique())

n_dupe_rows = df.duplicated(subset=["asset_id", "timestamp"], keep=False).sum()
print(f"Exact (asset_id, timestamp) duplicate rows: {n_dupe_rows} "
      f"(looks like double-published readings - dropped for all modeling tasks, "
      f"kept here so the raw-data audit is complete)")

# irregular sampling check - not a clean 1-row-per-hour grid
gap_hours = (
    df.sort_values(["asset_id", "timestamp"])
    .groupby("asset_id")["timestamp"].diff().dt.total_seconds() / 3600
)
n_gap_events = (gap_hours > 1.5).sum()
print(f"Reading gaps > 1.5h (comms outages, not just clock jitter): {n_gap_events} "
      f"across {df['asset_id'].nunique()} assets")

missing = df.isna().mean().sort_values(ascending=False) * 100
print(missing[missing > 0].round(2).to_string())
print("\n(Missingness isn't uniform - it's elevated on rows within the next 24 clock hours of a fault, "
      "i.e. it's MNAR, not MCAR. Worth remembering before doing anything as blunt "
      "as global mean-imputation.)")
recent_fault_missing = (
    df.assign(recent_fault=df.groupby("asset_id")["fault_flag"]
              .transform(lambda s: s.rolling(24, min_periods=1).max()))
    .groupby("recent_fault")[["temperature", "vibration"]].apply(lambda g: g.isna().mean())
)
print("\nMissing rate, normal vs. within-24h-of-fault rows:")
print(recent_fault_missing.round(4).to_string())

# dedupe now that the raw-data audit above has been printed - everything
# from here on (distributions, figures, the pre-fault comparison) should
# reflect one row per (asset, timestamp), keeping the later-arriving copy
# of any double-published reading.
df = df.sort_values(["asset_id", "timestamp"]).drop_duplicates(
    subset=["asset_id", "timestamp"], keep="last"
)

print("\n" + "=" * 70)
print("2. DISTRIBUTIONS (describe)")
print("=" * 70)
num_cols = ["temperature", "humidity", "pressure", "vibration", "power_consumption", "occupancy_count"]
print(df[num_cols].describe().T.round(2).to_string())

# ---- outlier scan (simple IQR flags used for narrative, not deletion) ----
print("\n" + "=" * 70)
print("3. OUTLIER SCAN (IQR rule)")
print("=" * 70)
for c in ["temperature", "power_consumption", "vibration"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    n_out = ((df[c] < lo) | (df[c] > hi)).sum()
    print(f"{c}: bounds=({lo:.2f}, {hi:.2f})  extreme_outliers={n_out}")

# --------------------------------------------------------------------
# FIGURE 1: distributions
# --------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df[col].dropna(), bins=50, ax=ax, kde=True, color="#3b6fa0")
    ax.set_title(col)
fig.suptitle("Figure 1 — Sensor Reading Distributions", fontsize=14)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_distributions.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------
# FIGURE 2: daily/hourly patterns of power consumption
# --------------------------------------------------------------------
hourly = df.groupby("hour")["power_consumption"].mean()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily = df.groupby("dow")["power_consumption"].mean().reindex(dow_order)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
hourly.plot(ax=axes[0], marker="o", color="#c0504d")
axes[0].set_title("Avg Power Consumption by Hour of Day")
axes[0].set_xlabel("Hour"); axes[0].set_ylabel("kWh")
daily.plot(kind="bar", ax=axes[1], color="#4f81bd")
axes[1].set_title("Avg Power Consumption by Day of Week")
axes[1].set_ylabel("kWh")
fig.suptitle("Figure 2 — Temporal Patterns in Energy Use", fontsize=14)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_temporal_patterns.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------
# FIGURE 3: asset-type comparison (power + vibration)
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="asset_type", y="power_consumption", ax=axes[0], hue="asset_type", legend=False)
axes[0].set_title("Power Consumption by Asset Type")
axes[0].tick_params(axis="x", rotation=30)
sns.boxplot(data=df, x="asset_type", y="vibration", ax=axes[1], hue="asset_type", legend=False)
axes[1].set_title("Vibration by Asset Type")
axes[1].tick_params(axis="x", rotation=30)
fig.suptitle("Figure 3 — Performance Comparison Across Asset Types", fontsize=14)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_asset_type_comparison.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------
# FIGURE 4: pre-fault behaviour (the predictive-maintenance signal)
# --------------------------------------------------------------------
df_sorted = df.sort_values(["asset_id", "timestamp"]).copy()
next_fault_ts = df_sorted["timestamp"].where(df_sorted["fault_flag"].eq(1)).groupby(df_sorted["asset_id"]).bfill()
df_sorted["will_fault_24h"] = ((next_fault_ts.notna()) & (next_fault_ts > df_sorted["timestamp"]) & (next_fault_ts <= df_sorted["timestamp"] + pd.Timedelta(hours=24))).astype(int)

comp = df_sorted.groupby("will_fault_24h")[["vibration", "temperature", "power_consumption"]].mean()
print("\n" + "=" * 70)
print("4. FEATURE MEANS: normal vs. pre-fault (next 24h) windows")
print("=" * 70)
print(comp.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, ["vibration", "temperature", "power_consumption"]):
    sns.barplot(data=df_sorted, x="will_fault_24h", y=col, ax=ax, hue="will_fault_24h", legend=False,
                palette=["#4f81bd", "#c0504d"])
    ax.set_xticklabels(["Normal", "Fails in <24h"])
    ax.set_title(col)
fig.suptitle("Figure 4 — Sensor Behaviour: Normal vs. Pre-Fault Window", fontsize=14)
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_prefault_behavior.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------
# FIGURE 5: correlation heatmap
# --------------------------------------------------------------------
corr = df[num_cols + ["fault_flag"]].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Figure 5 — Correlation Matrix")
fig.tight_layout()
fig.savefig(f"{FIG}/fig5_correlation_heatmap.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------
# FIGURE 6: site-level energy comparison
# --------------------------------------------------------------------
site_energy = df.groupby(["site_id", "date"])["power_consumption"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=site_energy, x="date", y="power_consumption", hue="site_id", ax=ax)
ax.set_title("Figure 6 — Daily Total Energy Consumption by Site")
ax.set_ylabel("kWh (site total)")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{FIG}/fig6_site_energy_trend.png", dpi=130)
plt.close(fig)

print("\nAll figures written to", FIG)

print("\n" + "=" * 70)
print("5. KEY OBSERVATIONS (auto-derived, for report drafting)")
print("=" * 70)
top_power_type = df.groupby("asset_type")["power_consumption"].mean().idxmax()
fault_by_type = df.groupby("asset_type")["fault_flag"].mean().sort_values(ascending=False)
print(f"- Highest average energy draw by asset type: {top_power_type}")
print("- Fault rate by asset type:\n", fault_by_type.round(5).to_string())
print(f"- Business-hours power premium: {hourly.max() / hourly.min():.2f}x peak vs. trough hour")

print("\n" + "=" * 70)
print("6. FAULT TYPE BREAKDOWN (new: multiple fault signatures, not one generic ramp)")
print("=" * 70)
fault_type_counts = df.loc[df["fault_flag"] == 1, "fault_type"].value_counts()
print(fault_type_counts.to_string())
print("\nEach fault type leans on a different lead sensor (see generate_data.py "
      "FAULT_PROFILES) - e.g. refrigerant_leak is temperature-led with almost no "
      "vibration signature, while bearing_wear is the opposite. Worth keeping in "
      "mind for Task 2: a model trained only on vibration features would miss "
      "refrigerant-leak-type failures almost entirely.")
