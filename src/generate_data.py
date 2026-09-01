"""
Nectar Intelligent Facilities Platform
Synthetic Data Generator

Generates three linked datasets that mimic a real commercial-buildings IoT
deployment:
    1. sensor_telemetry.csv   - hourly time series per asset
    2. asset_metadata.csv     - static asset registry (hierarchy via parent_asset_id)
    3. asset_connectivity.csv - explicit graph edges between assets

Design choices (documented for the README):
- 3 sites x 2-3 buildings x ~6-9 assets/building => ~48 assets total.
- Hierarchy: Site -> Building -> Chiller/HVAC -> AHU/Pump -> Sensors/Meters.
- 45 days of nominally-HOURLY telemetry (Jun 15 - Jul 30 2026), but see
  "real-world data quirks" below - the grid isn't perfectly clean, on purpose.

--------------------------------------------------------------------------
First pass at this generator just had one generic "degradation ramp" per
asset and a flat 2% missing rate sprinkled uniformly at random. That's fine
for a toy demo, but it's not what telemetry from real field devices looks
like, and it made every downstream task a little *too* easy (a single
IsolationForest basically solves the whole problem). Reworked it below to
add texture that mirrors what actually goes wrong with IoT sensor fleets:

  1. MULTIPLE, ASSET-SPECIFIC FAULT MODES - not one generic "degradation".
     A chiller failing from a refrigerant leak looks nothing like a chiller
     failing from bearing wear, and a fouled AHU filter looks nothing like
     either. Each asset type has its own small library of plausible fault
     signatures (see FAULT_PROFILES) with different lead sensors, ramp
     shapes and onset speeds. This also means "vibration always spikes
     first" is no longer a free lunch for the model - it has to actually
     learn per-fault-type behaviour.
  2. SENSOR DRIFT, independent of faults. Real sensors slowly lose
     calibration (a temperature probe reading 0.4C high after a few weeks
     is normal wear, not a fault). Implemented as a slow bounded random
     walk added on top of the "true" reading, per asset per sensor. This
     is what Task 4's drift/change-point method is actually meant to catch
     - previously there was nothing for it to catch that wasn't *also* a
     labeled fault, which made that method redundant with the others.
  3. IRREGULAR SAMPLING. Real devices don't all tick in perfect lockstep on
     the hour - clock skew means readings land a few minutes early/late,
     and devices occasionally go offline for a stretch (gateway reboot,
     Wi-Fi drop, power blip) and simply produce no rows at all for a few
     hours, rather than a convenient NaN placeholder.
  4. MNAR (missing-not-at-random) DROPOUTS layered on top of the old
     uniform-random missingness. A sensor is measurably more likely to
     report a garbled/missing reading while an asset is mid-fault-ramp
     (high vibration literally shakes connectors loose) - so missingness
     itself carries a bit of signal, which is realistic and also means
     naive listwise deletion would bias the fault-detection tasks.
  5. DUPLICATE / DOUBLE-PUBLISHED READINGS. MQTT-style at-least-once
     delivery occasionally double-publishes a reading for the same
     (asset_id, timestamp) with slightly different jitter - a classic
     real-world dedup problem that the previous version didn't have at
     all in the telemetry table (only the connectivity table had a
     seeded duplicate).
  6. CROSS-ASSET CORRELATED DEGRADATION. When a Chiller ramps toward a
     fault, the AHUs/Pumps hydraulically downstream of it feel a smaller
     correlated stress bump (shared refrigerant loop / shared load) - not
     just the failing asset in isolation. Purely independent per-asset
     noise was unrealistic for a connected mechanical system.
  7. A shared per-site "ambient" latent factor (weather-like) nudges
     temperature and cooling load in a correlated way across every asset
     on a site, on top of each asset's own noise - so correlation-heatmap
     structure in Task 1 isn't just autocorrelation, it's partly a shared
     external driver, same as in a real building portfolio.

Still true from before:
- Faults are NOT random noise; each episode ramps up over a period before
  the fault event, then resets after a simulated "maintenance" action.
- Energy consumption has daily + weekly occupancy-driven seasonality plus
  a mild upward trend.
- A handful of physically-impossible outliers (100C+ spikes etc.) are
  injected on purpose.
- The connectivity graph intentionally contains a duplicate edge, one
  orphan asset, and one invalid parent reference.

New `fault_type` column (telemetry) records which fault profile produced a
given fault_flag==1 window (NaN when not mid-fault) - this is a modeling
convenience/audit trail, not something a real deployment would necessarily
have pre-labeled, and Task 2 does not use it as a *feature* (that would be
leakage - it's only known once the fault has already happened).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent

RNG = np.random.default_rng(42)

SITES = ["SITE_A", "SITE_B", "SITE_C"]
BUILDINGS_PER_SITE = {"SITE_A": 3, "SITE_B": 2, "SITE_C": 2}
ASSET_TYPES = ["Chiller", "AHU", "Pump", "HVAC", "EnergyMeter", "EnvSensor"]
MANUFACTURERS = ["Carrier", "Trane", "Johnson Controls", "Daikin", "Siemens", "Honeywell"]

START = datetime(2026, 6, 15, 0, 0, 0)
N_HOURS = 45 * 24  # 45 days, nominal hourly grid
TIMESTAMPS = [START + timedelta(hours=h) for h in range(N_HOURS)]

# --------------------------------------------------------------------------
# Fault profiles per asset type: (fault_type, lead_sensor_weights, ramp_shape)
# lead_sensor_weights control how strongly vibration/temperature/power react,
# relative to the old one-size-fits-all ramp. ramp_exponent < 1 = fast
# initial onset that levels off; > 1 = slow creep that accelerates late.
# --------------------------------------------------------------------------
FAULT_PROFILES = {
    "Chiller": [
        # refrigerant leak: temperature climbs hard, vibration barely moves
        {"name": "refrigerant_leak", "vib_w": 0.3, "temp_w": 1.6, "pow_w": 0.6, "ramp_exp": 1.3},
        # bearing wear: vibration-dominant, slow creeping onset
        {"name": "bearing_wear", "vib_w": 1.8, "temp_w": 0.4, "pow_w": 0.5, "ramp_exp": 1.6},
        # electrical fault: fast onset, power-dominant
        {"name": "electrical_fault", "vib_w": 0.5, "temp_w": 0.5, "pow_w": 1.7, "ramp_exp": 0.6},
    ],
    "AHU": [
        # filter fouling: efficiency loss -> power creeps up, little vibration
        {"name": "filter_fouling", "vib_w": 0.2, "temp_w": 0.5, "pow_w": 1.5, "ramp_exp": 1.8},
        # belt slippage: vibration + moderate power, faster onset
        {"name": "belt_slippage", "vib_w": 1.5, "temp_w": 0.3, "pow_w": 0.8, "ramp_exp": 0.8},
    ],
    "Pump": [
        {"name": "bearing_wear", "vib_w": 1.7, "temp_w": 0.5, "pow_w": 0.6, "ramp_exp": 1.5},
        # cavitation: noisy vibration + pressure disturbance, fast onset
        {"name": "cavitation", "vib_w": 1.4, "temp_w": 0.2, "pow_w": 0.4, "ramp_exp": 0.7},
    ],
    "HVAC": [
        {"name": "refrigerant_leak", "vib_w": 0.3, "temp_w": 1.5, "pow_w": 0.6, "ramp_exp": 1.3},
        {"name": "electrical_fault", "vib_w": 0.4, "temp_w": 0.5, "pow_w": 1.6, "ramp_exp": 0.6},
    ],
    # EnergyMeter/EnvSensor: no active mechanical fault modes (passive devices)
}


# --------------------------------------------------------------------------
# 1. Asset metadata + hierarchy
# --------------------------------------------------------------------------
def build_assets():
    rows = []
    asset_counter = 1
    building_counter = 1
    asset_ids_by_building = {}

    for site in SITES:
        n_buildings = BUILDINGS_PER_SITE[site]
        for b in range(n_buildings):
            building_id = f"BLDG_{building_counter:03d}"
            building_counter += 1
            asset_ids_by_building[building_id] = {"site": site, "assets": []}

            # Each building: 1 Chiller (top), 1 HVAC, 2 AHUs under chiller,
            # 1 Pump under chiller, 1 EnergyMeter (building level, no parent),
            # 2 EnvSensors under an AHU each.
            chiller_id = f"AST_{asset_counter:04d}"; asset_counter += 1
            hvac_id = f"AST_{asset_counter:04d}"; asset_counter += 1
            ahu1_id = f"AST_{asset_counter:04d}"; asset_counter += 1
            ahu2_id = f"AST_{asset_counter:04d}"; asset_counter += 1
            pump_id = f"AST_{asset_counter:04d}"; asset_counter += 1
            meter_id = f"AST_{asset_counter:04d}"; asset_counter += 1
            sensor1_id = f"AST_{asset_counter:04d}"; asset_counter += 1
            sensor2_id = f"AST_{asset_counter:04d}"; asset_counter += 1

            defs = [
                (chiller_id, "Chiller", None),
                (hvac_id, "HVAC", None),
                (ahu1_id, "AHU", chiller_id),
                (ahu2_id, "AHU", chiller_id),
                (pump_id, "Pump", chiller_id),
                (meter_id, "EnergyMeter", None),
                (sensor1_id, "EnvSensor", ahu1_id),
                (sensor2_id, "EnvSensor", ahu2_id),
            ]

            for aid, atype, parent in defs:
                install_date = datetime(2019, 1, 1) + timedelta(
                    days=int(RNG.integers(0, 2200))
                )
                capacity = {
                    "Chiller": RNG.uniform(200, 600),
                    "AHU": RNG.uniform(20, 80),
                    "Pump": RNG.uniform(5, 25),
                    "HVAC": RNG.uniform(50, 150),
                    "EnergyMeter": np.nan,
                    "EnvSensor": np.nan,
                }[atype]
                rows.append(
                    {
                        "asset_id": aid,
                        "site_id": site,
                        "building_id": building_id,
                        "asset_name": f"{atype}-{aid[-4:]}",
                        "asset_type": atype,
                        "manufacturer": RNG.choice(MANUFACTURERS),
                        "installation_date": install_date.date().isoformat(),
                        "capacity": None if pd.isna(capacity) else round(capacity, 1),
                        "parent_asset_id": parent,
                    }
                )
                asset_ids_by_building[building_id]["assets"].append((aid, atype))

    assets_df = pd.DataFrame(rows)

    # --- inject a couple of intentional data-quality issues ---
    # 1 invalid parent reference (points to a non-existent asset)
    idx_invalid = assets_df[assets_df["asset_type"] == "AHU"].index[0]
    assets_df.loc[idx_invalid, "parent_asset_id"] = "AST_9999"
    # 1 orphan asset that should have a parent but doesn't (a Pump with no parent)
    idx_orphan = assets_df[assets_df["asset_type"] == "Pump"].index[-1]
    assets_df.loc[idx_orphan, "parent_asset_id"] = None

    return assets_df


# --------------------------------------------------------------------------
# 2. Connectivity graph
# --------------------------------------------------------------------------
def build_connectivity(assets_df):
    edges = []
    for _, row in assets_df.iterrows():
        if pd.notna(row["parent_asset_id"]) and row["parent_asset_id"] in set(
            assets_df["asset_id"]
        ):
            conn_type = {
                "AHU": "Supplies",
                "Pump": "Supplies",
                "EnvSensor": "Monitors",
            }.get(row["asset_type"], "Controls")
            edges.append(
                {
                    "source_asset_id": row["parent_asset_id"],
                    "target_asset_id": row["asset_id"],
                    "connection_type": conn_type,
                    "relationship_strength": round(RNG.uniform(0.5, 1.0), 2),
                }
            )

    # cross links: HVAC monitors the building's EnergyMeter
    for building_id, grp in assets_df.groupby("building_id"):
        hvac = grp[grp["asset_type"] == "HVAC"]["asset_id"]
        meter = grp[grp["asset_type"] == "EnergyMeter"]["asset_id"]
        if len(hvac) and len(meter):
            edges.append(
                {
                    "source_asset_id": hvac.iloc[0],
                    "target_asset_id": meter.iloc[0],
                    "connection_type": "Monitors",
                    "relationship_strength": round(RNG.uniform(0.6, 0.9), 2),
                }
            )

    conn_df = pd.DataFrame(edges)

    # --- inject intentional data-quality issues ---
    # 1 exact duplicate edge
    conn_df = pd.concat([conn_df, conn_df.iloc[[0]]], ignore_index=True)

    return conn_df


# --------------------------------------------------------------------------
# 3. Site-level shared "ambient" latent factor (weather-like)
#    Shared across every asset on a site so cross-asset correlation in
#    Task 1's heatmap isn't purely an artifact of autocorrelation.
# --------------------------------------------------------------------------
def build_site_ambient(site, seed):
    rng = np.random.default_rng(seed)
    n = N_HOURS
    day_frac = np.arange(n) / n
    hours = np.array([t.hour for t in TIMESTAMPS])
    # slow seasonal warmup across the 45 days + daily cycle + red-noise wander
    seasonal = 4.0 * np.sin(2 * np.pi * day_frac) + 3.0 * np.sin(2 * np.pi * (hours - 15) / 24)
    wander = np.cumsum(rng.normal(0, 0.15, n))
    wander -= wander.mean()
    wander = np.clip(wander, -3, 3)
    return seasonal + wander


SITE_AMBIENT = {site: build_site_ambient(site, seed=7000 + i) for i, site in enumerate(SITES)}


# --------------------------------------------------------------------------
# 4. Telemetry (the interesting part: latent degradation -> faults)
# --------------------------------------------------------------------------
def simulate_asset_series(asset_id, asset_type, site, building_id, seed,
                           parent_fault_boost=None):
    """
    parent_fault_boost: optional array (len N_HOURS) of upstream stress,
    0-1 scale, from a parent asset's own fault ramp - applied as a smaller
    correlated bump to this (downstream) asset's vibration/temperature.
    """
    rng = np.random.default_rng(seed)
    n = N_HOURS
    hours = np.array([t.hour for t in TIMESTAMPS])
    dow = np.array([t.weekday() for t in TIMESTAMPS])
    is_business_hours = ((hours >= 8) & (hours <= 19) & (dow < 5)).astype(float)

    occupancy = (
        20
        + 60 * is_business_hours
        + rng.normal(0, 4, n)
        + 10 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24)
    )
    occupancy = np.clip(occupancy, 0, None).round().astype(int)

    base_temp = {
        "Chiller": 8.0, "AHU": 18.0, "Pump": 25.0,
        "HVAC": 22.0, "EnergyMeter": 24.0, "EnvSensor": 23.0,
    }[asset_type]
    base_power = {
        "Chiller": 120, "AHU": 18, "Pump": 6,
        "HVAC": 35, "EnergyMeter": 0.5, "EnvSensor": 0.05,
    }[asset_type]
    base_vibe = {
        "Chiller": 2.5, "AHU": 1.8, "Pump": 3.0,
        "HVAC": 1.5, "EnergyMeter": 0.1, "EnvSensor": 0.05,
    }[asset_type]

    ambient = SITE_AMBIENT[site]  # shared per-site latent factor

    temperature = base_temp + 0.3 * ambient + rng.normal(0, 0.6, n)
    power = (
        base_power
        * (0.5 + 0.9 * is_business_hours + 0.15 * (occupancy / 80))
        + rng.normal(0, base_power * 0.05, n)
    )
    power = np.clip(power, 0, None)
    humidity = np.clip(45 + 8 * np.sin(2 * np.pi * (hours - 6) / 24) + rng.normal(0, 3, n), 10, 95)
    pressure = np.clip(101.3 + rng.normal(0, 0.4, n) + (0.05 if asset_type == "Pump" else 0), 95, 105)
    vibration = np.clip(base_vibe + rng.normal(0, base_vibe * 0.08, n), 0, None)

    operating_mode = np.where(
        is_business_hours == 1,
        rng.choice(["Cooling", "Heating"], size=n, p=[0.8, 0.2]),
        "Idle",
    )

    # ---- slow sensor calibration drift, independent of faults ----
    # bounded random walk per sensor - this is what Task 4's drift check
    # should catch even on assets that never actually fault.
    def drift_walk(step_std, cap):
        w = np.cumsum(rng.normal(0, step_std, n))
        return np.clip(w, -cap, cap)

    temp_drift = drift_walk(0.01, 1.2)
    vib_drift = drift_walk(0.004, base_vibe * 0.35)
    temperature = temperature + temp_drift
    vibration = np.clip(vibration + vib_drift, 0, None)

    fault_flag = np.zeros(n, dtype=int)
    fault_type = np.array([None] * n, dtype=object)
    stress_signal = np.zeros(n)  # 0-1, exported for downstream correlated-fault use

    profiles = FAULT_PROFILES.get(asset_type)
    if profiles:
        n_episodes = rng.integers(1, 4) if asset_type in ("Chiller", "Pump", "AHU") else rng.integers(0, 2)
        for _ in range(n_episodes):
            profile = profiles[rng.integers(0, len(profiles))]
            ramp_len = int(rng.integers(12, 48))
            fault_start = int(rng.integers(ramp_len + 1, n - 5))
            ramp_idx = np.arange(fault_start - ramp_len, fault_start)
            ramp_progress = np.linspace(0, 1, ramp_len) ** profile["ramp_exp"]

            vibration[ramp_idx] += ramp_progress * base_vibe * profile["vib_w"] * rng.uniform(1.3, 2.6)
            temperature[ramp_idx] += ramp_progress * profile["temp_w"] * rng.uniform(3, 8)
            power[ramp_idx] += ramp_progress * base_power * profile["pow_w"] * rng.uniform(0.25, 0.7)
            stress_signal[ramp_idx] = np.maximum(stress_signal[ramp_idx], ramp_progress)

            fault_window = min(fault_start + int(rng.integers(1, 4)), n)
            fault_flag[fault_start:fault_window] = 1
            fault_type[fault_start:fault_window] = profile["name"]
            stress_signal[fault_start:fault_window] = 1.0

            # post-fault "maintenance reset" - brief cooldown then back to normal
            recover_end = min(fault_window + 6, n)
            if recover_end > fault_window:
                vibration[fault_window:recover_end] *= np.linspace(1.0, 0.4, recover_end - fault_window)

    # ---- correlated stress bump from an upstream (parent) asset's fault ----
    if parent_fault_boost is not None:
        bump = 0.35 * parent_fault_boost  # damped relative to the source asset
        vibration = np.clip(vibration + bump * base_vibe * 0.6, 0, None)
        temperature = temperature + bump * 1.2
        stress_signal = np.maximum(stress_signal, 0.4 * parent_fault_boost)

    df = pd.DataFrame(
        {
            "timestamp": TIMESTAMPS,
            "site_id": site,
            "building_id": building_id,
            "asset_id": asset_id,
            "temperature": np.round(temperature, 2),
            "humidity": np.round(humidity, 2),
            "pressure": np.round(pressure, 2),
            "vibration": np.round(vibration, 3),
            "power_consumption": np.round(power, 3),
            "occupancy_count": occupancy,
            "operating_mode": operating_mode,
            "fault_flag": fault_flag,
            "fault_type": fault_type,
        }
    )
    return df, stress_signal


def build_telemetry(assets_df):
    frames = []
    stress_by_asset = {}  # asset_id -> stress_signal array, for downstream correlation

    # pass 1: simulate every asset with no upstream boost yet
    for i, row in assets_df.reset_index(drop=True).iterrows():
        df, stress = simulate_asset_series(
            row["asset_id"], row["asset_type"], row["site_id"],
            row["building_id"], seed=1000 + i,
        )
        frames.append(df)
        stress_by_asset[row["asset_id"]] = stress

    # pass 2: re-simulate children of a Chiller/AHU with a correlated bump
    # from their parent's stress signal (hydraulically/thermally coupled
    # equipment doesn't fail in total isolation from what it's plumbed into)
    parent_map = dict(zip(assets_df["asset_id"], assets_df["parent_asset_id"]))
    frames_final = []
    for i, row in assets_df.reset_index(drop=True).iterrows():
        parent = parent_map.get(row["asset_id"])
        if parent in stress_by_asset and row["asset_type"] in ("AHU", "Pump"):
            df, _ = simulate_asset_series(
                row["asset_id"], row["asset_type"], row["site_id"],
                row["building_id"], seed=1000 + i,
                parent_fault_boost=stress_by_asset[parent],
            )
            frames_final.append(df)
        else:
            frames_final.append(frames[i])

    telem = pd.concat(frames_final, ignore_index=True)

    # ======================================================================
    # Real-world data quirks layered on top of the "clean" simulated signal
    # ======================================================================

    # --- (a) MNAR missingness: baseline random + elevated during fault ramps
    # (high vibration literally shakes connectors loose / saturates ADCs) ---
    telem["_recent_fault"] = (
        telem.groupby("asset_id")["fault_flag"]
        .transform(lambda s: s.rolling(24, min_periods=1).max())
        .fillna(0)
    )
    base_missing_p = 0.012
    elevated_missing_p = 0.06
    for col in ["temperature", "humidity", "vibration", "power_consumption"]:
        p = np.where(telem["_recent_fault"] == 1, elevated_missing_p, base_missing_p)
        mask = RNG.random(len(telem)) < p
        telem.loc[mask, col] = np.nan
    telem = telem.drop(columns=["_recent_fault"])

    # --- (b) physically-impossible outliers on purpose ---
    n_outliers = int(0.0015 * len(telem))
    outlier_idx = RNG.choice(telem.index, size=n_outliers, replace=False)
    for idx in outlier_idx:
        choice = RNG.integers(0, 3)
        if choice == 0:
            telem.loc[idx, "temperature"] = RNG.uniform(80, 120)   # sensor spike
        elif choice == 1:
            telem.loc[idx, "power_consumption"] *= RNG.uniform(8, 15)  # power spike
        else:
            telem.loc[idx, "vibration"] = RNG.uniform(15, 25)      # vibration spike

    # --- (c) irregular sampling: per-reading clock jitter (device clock skew) ---
    jitter_minutes = RNG.integers(-4, 5, size=len(telem))
    telem["timestamp"] = pd.to_datetime(telem["timestamp"]) + pd.to_timedelta(
        jitter_minutes, unit="m"
    )

    # --- (d) comms outages: a handful of assets go fully offline for a
    # few consecutive hours a few times over the 45 days (rows dropped
    # entirely, not NaN'd - the device never phoned home at all) ---
    telem = telem.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    outage_assets = RNG.choice(
        telem["asset_id"].unique(), size=max(1, len(telem["asset_id"].unique()) // 4), replace=False
    )
    drop_idx = []
    for aid in outage_assets:
        asset_rows = telem.index[telem["asset_id"] == aid].to_numpy()
        n_outage_events = RNG.integers(1, 3)
        for _ in range(n_outage_events):
            if len(asset_rows) < 10:
                continue
            outage_len = int(RNG.integers(2, 7))  # 2-6 consecutive hourly rows
            start_pos = int(RNG.integers(0, max(1, len(asset_rows) - outage_len)))
            drop_idx.extend(asset_rows[start_pos:start_pos + outage_len].tolist())
    telem = telem.drop(index=drop_idx).reset_index(drop=True)

    # --- (e) duplicate / double-published readings (at-least-once delivery) ---
    n_dupes = max(1, int(0.0008 * len(telem)))
    dupe_idx = RNG.choice(telem.index, size=n_dupes, replace=False)
    dupes = telem.loc[dupe_idx].copy()
    # the re-published copy has slightly different jitter/noise, same asset+timestamp
    for col in ["temperature", "vibration", "power_consumption"]:
        dupes[col] = dupes[col] + RNG.normal(0, 0.02, len(dupes)) * dupes[col].abs().clip(lower=0.1)
    telem = pd.concat([telem, dupes], ignore_index=True)

    telem = telem.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    return telem


if __name__ == "__main__":
    assets_df = build_assets()
    conn_df = build_connectivity(assets_df)
    telem_df = build_telemetry(assets_df)

    assets_df.to_csv(ROOT / "data" / "asset_metadata.csv", index=False)
    conn_df.to_csv(ROOT / "data" / "asset_connectivity.csv", index=False)
    telem_df.to_csv(ROOT / "data" / "sensor_telemetry.csv", index=False)

    print("assets:", assets_df.shape)
    print("connectivity:", conn_df.shape)
    print("telemetry:", telem_df.shape)
    print("fault rate:", telem_df["fault_flag"].mean())
    print("fault types:", telem_df["fault_type"].value_counts(dropna=True).to_dict())
    print("exact duplicate (asset_id, timestamp) rows:",
          telem_df.duplicated(subset=["asset_id", "timestamp"]).sum())
    print("missing rate by column:\n", telem_df.isna().mean().round(4).to_string())
