"""Task 3: Direct multi-horizon (1–24h) building energy forecasting.

The model predicts each of the next 24 hourly values directly from information
available at the forecast origin. Test origins are the final 7 days, so the
reported metrics measure a genuine 24-hour forecast horizon rather than a
one-step-ahead evaluation that accidentally uses observations inside the test
window.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_XGB = False
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
DATA, FIG, MODELS = ROOT / "data", ROOT / "figures", ROOT / "models"
for d in (FIG, MODELS): d.mkdir(parents=True, exist_ok=True)

telem = pd.read_csv(DATA / "sensor_telemetry.csv", parse_dates=["timestamp"])
telem = telem.sort_values(["asset_id", "timestamp"]).drop_duplicates(["asset_id", "timestamp"], keep="last")
telem["ts_hour"] = telem["timestamp"].dt.round("h")
per_asset_hour = telem.groupby(["building_id", "asset_id", "ts_hour"], as_index=False)["power_consumption"].mean()
bld = per_asset_hour.groupby(["building_id", "ts_hour"], as_index=False)["power_consumption"].sum().rename(columns={"ts_hour":"timestamp"})

# Put every building on a common hourly grid. Missing target hours remain NaN;
# they are not filled with future information. Training samples requiring a
# missing target are skipped, while lag features are built only from history.
frames=[]
for bid, g in bld.groupby("building_id"):
    idx=pd.date_range(g.timestamp.min().floor("h"), g.timestamp.max().ceil("h"), freq="h")
    x=g.set_index("timestamp")["power_consumption"].reindex(idx)
    frames.append(pd.DataFrame({"building_id":bid,"timestamp":idx,"power_consumption":x.values}))
bld=pd.concat(frames, ignore_index=True).sort_values(["building_id","timestamp"]).reset_index(drop=True)

# Causal features at forecast origin.
bld["hour"] = bld.timestamp.dt.hour
bld["dow"] = bld.timestamp.dt.dayofweek
bld["is_weekend"] = (bld.dow >= 5).astype(int)
bld["hour_sin"] = np.sin(2*np.pi*bld.hour/24)
bld["hour_cos"] = np.cos(2*np.pi*bld.hour/24)
g=bld.groupby("building_id")["power_consumption"]
for lag in [1,2,3,24,48,168]: bld[f"lag_{lag}"]=g.shift(lag)
bld["roll_mean_24"]=g.transform(lambda s:s.shift(1).rolling(24,min_periods=12).mean())
bld["roll_mean_168"]=g.transform(lambda s:s.shift(1).rolling(168,min_periods=48).mean())
bld["roll_std_24"]=g.transform(lambda s:s.shift(1).rolling(24,min_periods=12).std())

# Direct multi-horizon samples: each origin predicts t+1 ... t+24 using only data at t or earlier.
max_ts=bld.timestamp.max()
test_origin_start=max_ts-pd.Timedelta(days=7)+pd.Timedelta(hours=1)
base=bld.copy()
base=base[base.lag_168.notna()].copy()
# Calendar features of the target time are known at forecast origin and therefore valid.
samples=[]
for h in range(1,25):
    x=base.copy()
    g_target=x.groupby("building_id")["power_consumption"]
    x["target"]=g_target.shift(-h)
    x["baseline"]=g_target.shift(24-h)
    target_ts=x.timestamp+pd.Timedelta(hours=h)
    x["horizon"]=h
    x["target_hour"]=target_ts.dt.hour
    x["target_dow"]=target_ts.dt.dayofweek
    x["target_is_weekend"]=(x.target_dow>=5).astype(int)
    x["target_hour_sin"]=np.sin(2*np.pi*x.target_hour/24)
    x["target_hour_cos"]=np.cos(2*np.pi*x.target_hour/24)
    x["target_timestamp"]=target_ts
    x=x[x.timestamp < max_ts-pd.Timedelta(hours=h)]
    x=x[x.target.notna() & x.baseline.notna()]
    samples.append(x)
samples_df=pd.concat(samples,ignore_index=True)
# Ensure train/test separation by forecast origin, not target timestamp.
train=samples_df[samples_df.timestamp < test_origin_start]
# Hold out exactly the final 7 days of forecast origins.
test=samples_df[samples_df.timestamp >= test_origin_start]
feature_cols=[c for c in samples_df.columns if c not in ("timestamp","target_timestamp","target","baseline","power_consumption")]
# One-hot building ID, fitted consistently across both partitions.
all_x=pd.get_dummies(samples_df[feature_cols], columns=["building_id"], prefix="bld", dtype=int)
train_x=all_x.loc[train.index]; test_x=all_x.loc[test.index]

if HAVE_XGB:
    model=XGBRegressor(n_estimators=250,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective="reg:squarederror",random_state=42,n_jobs=4)
else:
    model=HistGradientBoostingRegressor(max_iter=300,max_depth=5,learning_rate=.05,random_state=42)
model.fit(train_x, train.target)
pred=np.clip(model.predict(test_x),0,None)
mae=mean_absolute_error(test.target,pred); rmse=mean_squared_error(test.target,pred)**.5
mape=np.mean(np.abs((test.target-pred)/np.maximum(np.abs(test.target),1e-3)))*100
base_mae=mean_absolute_error(test.target,test.baseline); base_rmse=mean_squared_error(test.target,test.baseline)**.5
improvement=100*(1-mae/base_mae)
metrics={"evaluation":"direct multi-horizon 1-24h forecasts from a single origin; test origins are final 7 days","MAE":float(mae),"RMSE":float(rmse),"MAPE_pct":float(mape),"naive_baseline_MAE":float(base_mae),"naive_baseline_RMSE":float(base_rmse),"MAE_improvement_pct":float(improvement),"test_origin_start":str(test_origin_start)}
print(json.dumps(metrics,indent=2))
with open(MODELS/"forecasting_metrics.json","w") as f: json.dump(metrics,f,indent=2)
joblib.dump(model,MODELS/"forecasting_model.joblib")
joblib.dump(list(all_x.columns),MODELS/"forecasting_features.joblib")

out=test[["building_id","timestamp"]].copy(); out["target_hour"]=np.tile(np.arange(1,25),len(test)//24) if len(test)%24==0 else np.nan
out["actual"]=test.target.to_numpy(); out["forecast"]=pred; out["baseline"]=test.baseline.to_numpy()
# Recover building IDs as strings for plotting.
fig,axes=plt.subplots(2,2,figsize=(15,8)); buildings=sorted(test.building_id.unique())[:4]
for ax,bid in zip(axes.flat,buildings):
    sub=test[test.building_id==bid].copy(); pr=pd.Series(pred,index=test.index)[sub.index]
    ax.plot(sub.target_timestamp,sub.target,label="Actual"); ax.plot(sub.target_timestamp,pr,label="24h Forecast",linestyle="--"); ax.set_title(f"{bid} — direct 1–24h held-out forecasts"); ax.tick_params(axis="x",rotation=30); ax.legend(fontsize=8)
fig.suptitle("Figure 8 — Direct Multi-Horizon Energy Forecasts",fontsize=14); fig.tight_layout(); fig.savefig(FIG/"fig8_forecast_actual_vs_pred.png",dpi=130); plt.close(fig)

if hasattr(model,"feature_importances_"): imp=pd.Series(model.feature_importances_,index=all_x.columns).sort_values(ascending=False).head(12)
else:
    imp=pd.Series(np.zeros(len(all_x.columns)),index=all_x.columns).sort_values().tail(12)
fig,ax=plt.subplots(figsize=(8,5)); ax.barh(imp.index,imp.values); ax.set_title("Figure 9 — Forecasting Model: Top Feature Importances"); fig.tight_layout(); fig.savefig(FIG/"fig9_forecast_feature_importance.png",dpi=130); plt.close(fig)
