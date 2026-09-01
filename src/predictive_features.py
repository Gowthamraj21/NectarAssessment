"""Shared causal feature engineering for predictive-maintenance training/serving."""
import numpy as np
import pandas as pd

SENSOR_COLS = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]
SIGNAL_COLS = ["vibration", "temperature", "power_consumption"]

def _time_roll(df, col, window, statistic):
    out = pd.Series(index=df.index, dtype=float)
    for _, idx in df.groupby("asset_id", sort=False).groups.items():
        part = df.loc[idx].sort_values("timestamp")
        rolled = part.set_index("timestamp")[col].rolling(window, min_periods=2)
        out.loc[part.index] = getattr(rolled, statistic)().to_numpy()
    return out

def engineer_predictive_features(df, medians=None):
    x=df.copy().sort_values(["asset_id","timestamp"]).reset_index(drop=True)
    for c in SENSOR_COLS:
        x[c]=x.groupby("asset_id",sort=False)[c].ffill()
    x["hour"]=x.timestamp.dt.hour; x["dow"]=x.timestamp.dt.dayofweek; x["is_weekend"]=(x.dow>=5).astype(int)
    for col in SIGNAL_COLS:
        x[f"{col}_roll_mean_6"]=_time_roll(x,col,"6h","mean")
        x[f"{col}_roll_std_6"]=_time_roll(x,col,"6h","std").fillna(0)
        x[f"{col}_roll_mean_24"]=_time_roll(x,col,"24h","mean")
        x[f"{col}_delta_1"]=x.groupby("asset_id")[col].diff()
        x[f"{col}_slope_6"]=x[f"{col}_roll_mean_6"]-x[f"{col}_roll_mean_24"]
    x=pd.get_dummies(x,columns=["asset_type","operating_mode"],prefix=["type","mode"],dtype=int)
    if medians is not None:
        for c,v in medians.items():
            if c in x: x[c]=x[c].fillna(v)
    return x

def get_feature_columns(df):
    return SENSOR_COLS+["occupancy_count","hour","dow","is_weekend","capacity"]+[c for c in df.columns if any(k in c for k in ["_roll_","_delta_","_slope_"])]+[c for c in df.columns if c.startswith("type_") or c.startswith("mode_")]
