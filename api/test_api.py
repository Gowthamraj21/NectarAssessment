
"""
Nectar API Prediction Test

Reads telemetry CSV + asset metadata CSV, builds JSON requests,
sends them to the FastAPI /predict_failure endpoint, and saves
the prediction results to a CSV file.

Run:
    python api/test_api.py
"""

import pandas as pd
import requests
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TELEMETRY_FILE = ROOT / "data" / "sensor_telemetry.csv"

# CHANGE THIS if your asset metadata file has a different name
ASSET_FILE = ROOT / "data" / "asset_metadata.csv"

API_URL = "http://127.0.0.1:8000/predict_failure"

OUTPUT_FILE = ROOT / "api" / "prediction_results.csv"

# Number of previous readings to send as history
HISTORY_POINTS = 10


# ============================================================
# LOAD DATA
# ============================================================

print("Loading telemetry data...")

telemetry = pd.read_csv(TELEMETRY_FILE)

print(f"Telemetry rows: {len(telemetry)}")

print("Loading asset metadata...")

assets = pd.read_csv(ASSET_FILE)

print(f"Asset records: {len(assets)}")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

telemetry.columns = telemetry.columns.str.strip()
assets.columns = assets.columns.str.strip()


# ============================================================
# MERGE TELEMETRY + ASSET METADATA
# ============================================================

print("Merging telemetry with asset metadata...")

df = telemetry.merge(
    assets[
        [
            "asset_id",
            "asset_type",
            "capacity"
        ]
    ],
    on="asset_id",
    how="left"
)

print(f"Merged rows: {len(df)}")


# ============================================================
# CHECK FOR MISSING ASSET INFORMATION
# ============================================================

missing_assets = df["asset_type"].isna().sum()

if missing_assets > 0:
    print(
        f"WARNING: {missing_assets} telemetry rows "
        f"have no matching asset metadata."
    )


# ============================================================
# DATE/TIME CONVERSION
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(subset=["timestamp"])

df = df.sort_values(
    ["asset_id", "timestamp"]
).reset_index(drop=True)


# ============================================================
# BUILD API TELEMETRY OBJECT
# ============================================================

def build_reading(row):
    """Convert one dataframe row into the API telemetry format."""

    return {
        "timestamp": row["timestamp"].isoformat(),
        "asset_type": str(row["asset_type"]),
        "operating_mode": str(row["operating_mode"]),
        "temperature": float(row["temperature"]),
        "humidity": float(row["humidity"]),
        "pressure": float(row["pressure"]),
        "vibration": float(row["vibration"]),
        "power_consumption": float(row["power_consumption"]),
        "occupancy_count": int(row["occupancy_count"]),
        "capacity": (
            float(row["capacity"])
            if pd.notna(row["capacity"])
            else None
        )
    }


# ============================================================
# SEND PREDICTION REQUEST
# ============================================================

def predict(row, history):

    current = build_reading(row)

    history_payload = [
        build_reading(h)
        for h in history
    ]

    payload = {
        "current": current,
        "history": history_payload
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# PROCESS DATA
# ============================================================

results = []

print("\nStarting predictions...")
print(f"API: {API_URL}")
print("-" * 70)


for asset_id, asset_df in df.groupby("asset_id"):

    asset_df = asset_df.sort_values("timestamp").reset_index(drop=True)

    print(f"\nProcessing {asset_id} ({len(asset_df)} readings)")

    for i, row in asset_df.iterrows():

        # Previous readings only
        start_index = max(0, i - HISTORY_POINTS)

        history_df = asset_df.iloc[start_index:i]

        try:

            prediction = predict(
                row,
                history_df.to_dict("records")
            )

            result = {
                "asset_id": asset_id,
                "timestamp": row["timestamp"],
                "failure_probability_24h":
                    prediction["failure_probability_24h"],
                "risk_tier":
                    prediction["risk_tier"],
                "decision_threshold":
                    prediction["decision_threshold"],
                "model_used":
                    prediction["model_used"],
                "history_points_used":
                    prediction["history_points_used"]
            }

            results.append(result)

            print(
                f"  {row['timestamp']} -> "
                f"probability={prediction['failure_probability_24h']:.4f}, "
                f"risk={prediction['risk_tier']}"
            )

        except requests.exceptions.RequestException as e:

            print(
                f"  ERROR for {asset_id} "
                f"at {row['timestamp']}: {e}"
            )

        except Exception as e:

            print(
                f"  ERROR processing {asset_id}: {e}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

if results:

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETED")
    print("=" * 70)

    print(f"Predictions generated: {len(results_df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nFirst 10 predictions:")
    print(
        results_df.head(10).to_string(index=False)
    )

else:

    print("\nNo predictions were generated.")


print("\nDone.")

