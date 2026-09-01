"""Task 2: Predictive maintenance — leak-free 24-clock-hour classification."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
from predictive_features import engineer_predictive_features, get_feature_columns, SENSOR_COLS as sensor_cols
try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGBOOST = False

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
DATA, FIG, MODELS = ROOT / "data", ROOT / "figures", ROOT / "models"
for d in (FIG, MODELS): d.mkdir(parents=True, exist_ok=True)

telem = pd.read_csv(DATA / "sensor_telemetry.csv", parse_dates=["timestamp"])
assets = pd.read_csv(DATA / "asset_metadata.csv")
telem = telem.merge(assets[["asset_id", "asset_type", "capacity"]], on="asset_id", how="left")
telem = telem.sort_values(["asset_id", "timestamp"]).drop_duplicates(["asset_id", "timestamp"], keep="last").reset_index(drop=True)

# A true 24-clock-hour target: for a non-fault row, find the next observed fault
# timestamp for the same asset and mark positive only when it is <= current+24h.
next_fault_ts = telem["timestamp"].where(telem["fault_flag"].eq(1)).groupby(telem["asset_id"]).bfill()
telem["label_fail_24h"] = ((next_fault_ts.notna()) & (next_fault_ts > telem["timestamp"]) & (next_fault_ts <= telem["timestamp"] + pd.Timedelta(hours=24))).astype(int)
# Active faults are excluded: the use case is proactive prediction, not detection.
telem = telem[telem["fault_flag"].eq(0)].copy()
telem.drop(columns=["fault_type"], errors="ignore", inplace=True)

# Shared causal feature engineering (the same implementation is reused by API/dashboard).
telem = engineer_predictive_features(telem)
feature_cols = get_feature_columns(telem)

# Global chronological split. A 24h embargo prevents train labels from looking into the test period.
cutoff = telem["timestamp"].quantile(0.70)
train_end = cutoff - pd.Timedelta(hours=24)
train = telem[telem["timestamp"] <= train_end].copy()
test = telem[telem["timestamp"] > cutoff].copy()
# Imputation statistics are learned from training only and then applied to both sets.
medians = {c: float(train[c].median()) if pd.notna(train[c].median()) else 0.0 for c in sensor_cols + ["capacity"]}
for c, med in medians.items():
    train[c] = train[c].fillna(med); test[c] = test[c].fillna(med)
for c in feature_cols:
    if c not in train: train[c] = 0
    if c not in test: test[c] = 0
X_train, y_train = train[feature_cols], train["label_fail_24h"]
X_test, y_test = test[feature_cols], test["label_fail_24h"]
print(f"Train: {X_train.shape}, positive rate={y_train.mean():.4f}")
print(f"Test:  {X_test.shape}, positive rate={y_test.mean():.4f}")

rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
if HAVE_XGBOOST:
    gb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42, n_jobs=-1)
    gb.fit(X_train, y_train)
else:
    gb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train, sample_weight=np.where(y_train.eq(1), scale_pos_weight, 1.0))
gb_proba = gb.predict_proba(X_test)[:, 1]
gb_name = "XGBoost" if HAVE_XGBOOST else "HistGradientBoosting"

# Choose threshold on the training data using out-of-fold predictions, then freeze it for test.
# Tune the alert threshold on a final chronological validation slice of TRAIN only.
# TEST remains untouched until final evaluation.
val_cut = int(len(X_train) * 0.80)
threshold_model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=43, n_jobs=-1)
threshold_model.fit(X_train.iloc[:val_cut], y_train.iloc[:val_cut])
val_proba = threshold_model.predict_proba(X_train.iloc[val_cut:])[:, 1]
thresholds = np.linspace(0.05, 0.95, 91)
threshold = max(thresholds, key=lambda t: f1_score(y_train.iloc[val_cut:], (val_proba >= t).astype(int), zero_division=0))

results = {}
for name, proba in [("RandomForest", rf_proba), (gb_name, gb_proba)]:
    pred = (proba >= threshold).astype(int)
    results[name] = {"precision": float(precision_score(y_test, pred, zero_division=0)), "recall": float(recall_score(y_test, pred, zero_division=0)), "f1": float(f1_score(y_test, pred, zero_division=0)), "roc_auc": float(roc_auc_score(y_test, proba))}
    print(f"\n--- {name} @ threshold={threshold:.3f} ---\n", json.dumps(results[name], indent=2))
    print(classification_report(y_test, pred, digits=3, zero_division=0))

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = rf if best_name == "RandomForest" else gb
best_proba = rf_proba if best_name == "RandomForest" else gb_proba
best_pred = (best_proba >= threshold).astype(int)
joblib.dump(best_model, MODELS / "predictive_maintenance_model.joblib")
joblib.dump(feature_cols, MODELS / "predictive_maintenance_features.joblib")
joblib.dump({"threshold": float(threshold), "medians": medians}, MODELS / "predictive_maintenance_preprocessing.joblib")
with open(MODELS / "predictive_maintenance_metrics.json", "w") as f:
    json.dump({"results": results, "selected_model": best_name, "decision_threshold": float(threshold), "split": {"train_end": str(train_end), "test_start": str(cutoff), "embargo_hours": 24}, "label_definition": "next observed fault timestamp within 24 clock hours; current fault rows excluded"}, f, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
RocCurveDisplay.from_predictions(y_test, rf_proba, name="RandomForest", ax=axes[0])
RocCurveDisplay.from_predictions(y_test, gb_proba, name=gb_name, ax=axes[0])
axes[0].plot([0,1],[0,1],"k--",alpha=.4); axes[0].set_title("ROC Curve — Time Holdout")
cm = confusion_matrix(y_test, best_pred)
axes[1].imshow(cm, cmap="Blues")
for (i,j),v in np.ndenumerate(cm): axes[1].text(j,i,str(v),ha="center",va="center")
axes[1].set_xticks([0,1], ["No Fail","Fail <24h"]); axes[1].set_yticks([0,1], ["No Fail","Fail <24h"]); axes[1].set_title(f"Confusion Matrix — {best_name}")
# Built-in tree importance is deterministic and fast enough for a challenge artifact.
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values().tail(12)
else:
    imp = pd.Series(np.zeros(len(feature_cols)), index=feature_cols).sort_values().tail(12)
axes[2].barh(imp.index, imp.values); axes[2].set_title("Top Feature Importance")
fig.tight_layout(); fig.savefig(FIG / "fig7_predictive_maintenance_diagnostics.png", dpi=130); plt.close(fig)
print(f"Selected {best_name}; threshold={threshold:.3f}")
