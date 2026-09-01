# Nectar Intelligent Facilities Platform — Data Scientist Challenge

End-to-end solution covering EDA, predictive maintenance, energy forecasting,
anomaly detection, and multi-asset connectivity analysis for a synthetic
commercial-buildings IoT deployment, plus a bonus API deployment, dashboard,
and a Kafka streaming-ingestion demo.

## Note: A folder named Basic Analysis is added in which I've manually workedout everything.

## ⚠️ About the data

**No dataset was provided with the challenge brief** — only column schemas
for `sensor_telemetry`, `asset_metadata`, and `asset_connectivity`. A
realistic **synthetic** dataset was generated to match that schema
(`src/generate_data.py`). This is flagged clearly rather than pretending it's
real customer data. Key design choices (documented as comments in the script
itself):

- **3 sites, 7 buildings, 56 assets** (Chillers, AHUs, Pumps, HVAC units,
  Energy Meters, Environmental Sensors), hierarchically related via
  `parent_asset_id` and mirrored in `asset_connectivity`.
- **45 days of nominally-hourly telemetry** (~60,000 rows) — long enough to
  show weekly/seasonal structure, small enough to run the whole pipeline in
  minutes on a laptop. "Nominally" because the sampling grid isn't perfectly
  clean — see below.
- Faults are **not random noise, and not one generic ramp**. Each asset type
  has a small library of distinct fault signatures (e.g. a chiller
  refrigerant leak is temperature-led with almost no vibration signature;
  bearing wear on a pump is the opposite) — see `FAULT_PROFILES` in
  `generate_data.py`. Each episode ramps up for 12–48h before the fault
  event, then resets after a simulated "maintenance" action. This gives
  Task 2 a genuine, physically-plausible signal to learn, and makes it a
  real multi-class-flavored problem under the hood rather than one easy
  pattern.
- **Sensor calibration drift**, independent of any fault — a slow, bounded
  random walk on top of each sensor's "true" reading, present on every
  asset whether it ever faults or not. This is what Task 4's drift/
  change-point method is actually meant to catch (verified in that script's
  output: it now flags genuine drift-only anomalies that the other two
  methods miss, rather than just duplicating them).
- **Irregular sampling**: each reading has a few minutes of independent
  clock jitter, and a subset of assets have multi-hour comms-outage gaps
  (rows missing entirely, not NaN'd) a couple of times over the 45 days.
  Task 3's building-level aggregation had to be fixed to bucket readings to
  the nearest hour before summing across assets, because it previously
  assumed every asset reported at the exact same timestamp.
- **MNAR (missing-not-at-random) missingness**: baseline ~1.2% missing rate,
  elevated to ~6% on readings within 24h of a fault (stressed equipment
  drops readings more often — shaken connectors, saturated ADCs). A small
  number of deliberately implausible outliers (e.g. 100°C+ temperature
  spikes) are also injected on top, so Task 1's "identify anomalies/missing
  values" step and Task 4's anomaly detector have real cases to catch.
- **Duplicate / double-published readings**: a handful of exact
  `(asset_id, timestamp)` duplicates with slightly different values, mimicking
  at-least-once delivery from a real MQTT/Kafka ingestion pipeline. Every
  downstream task script dedupes these explicitly (keeping the
  later-arriving copy) before doing anything else with the data — the raw
  duplicate count is printed first so the audit trail stays honest about
  what's actually in the CSV.
- **Cross-asset correlated degradation**: when a Chiller ramps toward a
  fault, the AHUs/Pumps plumbed into it feel a smaller, correlated stress
  bump — not just the failing asset in total isolation. A shared per-site
  "ambient" latent factor also nudges temperature/load across every asset
  on a site, so correlation structure in Task 1's heatmap isn't purely
  autocorrelation.
- The connectivity graph deliberately contains **one duplicate edge, one
  invalid parent reference, and one orphaned asset**, so Task 5's data-quality
  audit isn't vacuous.

Regenerate the data anytime with:
```bash
python src/generate_data.py
```
(seeded with `numpy`'s `default_rng(42)` — fully reproducible.)

## Project structure

```
nectar_project/
├── data/                          # synthetic CSVs (generated, not hand-authored)
│   ├── sensor_telemetry.csv
│   ├── asset_metadata.csv
│   ├── asset_connectivity.csv
│   └── telemetry_with_anomalies.csv   # written by Task 4
├── src/
│   ├── generate_data.py           # synthetic data generator (documented assumptions)
│   ├── task1_eda.py                # Task 1: EDA
│   ├── task2_predictive_maintenance.py   # Task 2: failure prediction (RF + XGBoost)
│   ├── task3_forecasting.py       # Task 3: energy forecasting (XGBoost)
│   ├── task4_anomaly_detection.py # Task 4: IsolationForest + stats + drift
│   ├── task5_connectivity_analysis.py    # Task 5: graph analysis
│   └── build_notebook.py          # programmatically assembles + executes the notebook
├── notebooks/
│   └── Nectar_Facilities_Analysis.ipynb  # single narrated notebook, all tasks
├── figures/                       # all charts (also embedded in the notebook)
├── models/                        # trained models (.joblib) + metric/report JSONs
├── api/
│   └── main.py                    # FastAPI deployment: POST /predict_failure
├── dashboard/
│   └── app.py                     # Streamlit ops dashboard (bonus)
├── reports/
│   └── Nectar_Challenge_Report.docx     # written report (max 5 pages)
├── streaming/                      # bonus: Kafka ingestion demo
│   ├── producer.py                 # replays telemetry onto a Kafka topic
│   ├── consumer.py                 # subscribes, logs, persists to SQLite
│   ├── docker-compose.yml          # local single-broker Kafka (KRaft mode)
│   └── README.md                   # setup + run instructions
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

Run everything from scratch, in order:
```bash
python src/generate_data.py
python src/task1_eda.py
python src/task2_predictive_maintenance.py
python src/task3_forecasting.py
python src/task4_anomaly_detection.py
python src/task5_connectivity_analysis.py
```
Each script writes its figures to `figures/`, and Tasks 2–5 also write
models/metrics/reports to `models/`.

Or open the single narrated notebook, which runs the same code with markdown
commentary in between; execute it to refresh outputs:
```bash
jupyter notebook notebooks/Nectar_Facilities_Analysis.ipynb
```

### Run the API (bonus)
```bash
cd api
uvicorn main:app --reload --port 8000
# then open http://localhost:8000/docs for interactive Swagger UI
```
Example request body for `POST /predict_failure` is in the Swagger docs;
`history` is optional; when supplied, it lets the API reconstruct the same causal
rolling features used in training. In production, history should come from a
time-series feature store.

### Run the dashboard (bonus)
```bash
cd dashboard
streamlit run app.py
```

### Run the streaming ingestion demo (bonus)
```bash
cd streaming
docker compose up -d          # local Kafka broker
pip install kafka-python      # already in requirements.txt
python consumer.py            # terminal 1
python producer.py            # terminal 2
```
See `streaming/README.md` for the full walkthrough, options, and an
explanation of why Kafka (partition-per-asset ordering, multiple independent
consumer groups, replay buffer) fits this data's shape.

## Architecture overview

- **Data layer**: 3 CSVs matching the challenge schema, one generator script,
  fully reproducible.
- **Modeling layer**: independent, single-responsibility scripts per task
  (`task1`…`task5`), each idempotent — safe to re-run, each writes its own
  artifacts and doesn't depend on interactive state from another script.
- **Feature engineering** is **leak-free by construction**: all rolling
  statistics use `.rolling()` on past+current values only (no
  `center=True`, no future peeking), and Task 2 additionally uses a
  **time-based train/test split** (not random shuffling) since this is
  sequential sensor data — random splits would leak information from a
  fault's aftermath into the training set for predicting that same fault.
- **Serving layer**: FastAPI wraps the saved `joblib` model + feature list;
  Streamlit dashboard reads directly from the CSV/model artifacts (no
  separate database needed for this exercise).
- **Ingestion layer (bonus)**: `streaming/` demonstrates how this telemetry
  would actually arrive in production — a Kafka producer/consumer pair
  instead of a static CSV read. Kept as a clearly-scoped architecture demo
  (log + persist to SQLite) rather than wiring live scoring into the
  consumer, to avoid duplicating `api/main.py`'s feature-construction logic
  in two places — see `streaming/README.md`.

## Key assumptions

1. Telemetry is nominally hourly, but the generator intentionally adds clock
   jitter and multi-hour communication gaps. Task 2 therefore defines the
   target by timestamp: the next observed fault must occur within 24 clock
   hours, not merely within the next 24 rows.
2. `fault_flag` is treated as ground truth for *supervised* Task 2 only;
   Task 4 (anomaly detection) is deliberately **unsupervised** and evaluated
   by inspection + partial overlap with `fault_flag`, since real anomaly
   detection must also catch *novel* failure modes with no historical label.
3. Rows where `fault_flag==1` (an asset actively mid-fault) are excluded from
   Task 2 training/evaluation — the goal is predicting an *upcoming* failure
   for proactive scheduling, not detecting one already happening.
4. Energy forecasting is done at the **building** level (sum of all
   assets' `power_consumption` in that building), matching "predict energy
   consumption... for each building" in the brief.
5. Asset hierarchy is inferred from `parent_asset_id` in `asset_metadata`
   *and* the explicit `asset_connectivity` edges; both are merged into one
   directed graph for Task 5, since the brief describes both a tree-like
   hierarchy and a general connectivity graph.

## Design decisions & trade-offs

- **RandomForest vs. XGBoost for predictive maintenance**: both are trained
  and compared; the script auto-selects by F1 on the held-out time period.
  The current seeded run has a ~1.9% positive rate and six fault signatures,
  producing a deliberately non-trivial classification problem. The alert
  threshold is tuned on a chronological validation slice inside training. If `xgboost`
  isn't installed, both task2 and task3 fall back to sklearn's
  `HistGradientBoosting{Classifier,Regressor}` automatically (same
  histogram-based gradient-boosting family) so the pipeline still runs
  end-to-end on a minimal environment.
- **XGBoost (not Prophet/ARIMA/LSTM) for forecasting**: a single unified
  gradient-boosted model across all buildings, using lag + calendar
  features, was chosen for simplicity, speed, and because it naturally
  captures the business-hours × day-of-week interaction that drives most of
  the variance here. Prophet/ARIMA typically need one model per building and
  more manual seasonality specification; they're valid alternatives and
  called out as such in the script docstring, but weren't worth the added
  complexity given the size and character of this dataset. A naive
  "same-hour-yesterday" baseline is included so the improvement is
  quantified, not just asserted.
- **Three complementary anomaly methods, not one**: Isolation Forest catches
  multivariate outliers, rolling z-scores catch fast point spikes cheaply and
  explainably, and a short-vs-long rolling-mean "drift" check catches slow
  sensor drift that the other two miss. Recommending one fused pipeline over
  a single black-box model was a deliberate choice for explainability, which
  the brief calls out as a priority.
- **Graph over pure tree for connectivity**: `networkx.DiGraph` was used
  instead of a strict tree because a few assets (e.g. HVAC → EnergyMeter)
  have "Monitors" relationships that cut across the parent/child hierarchy —
  a tree can't represent that, a directed graph can.

## Known limitations

- Data is synthetic; absolute metric values (AUC, MAPE, etc.) reflect the
  synthetic signal-to-noise ratio designed into the generator, not a claim
  about real building performance.
- The API accepts optional recent history so the same causal rolling features can be reconstructed at serving time; in production this history should come from a feature store rather than the request body.
- Isolation Forest contamination (1%) and the anomaly z-score threshold
  (3.5σ) are fixed hyperparameters chosen by inspection; a production system
  would tune these per asset type/site using labeled anomaly feedback over
  time.
- The streaming demo (`streaming/`) is an ingestion-pattern demo, not a
  deployed real-time scoring service — see the scope note in
  `streaming/README.md`.
- `fault_type` (new column) is populated only for rows where `fault_flag==1`
  and is explicitly dropped before feature engineering in Task 2 — it's
  only knowable after a fault has already occurred, so using it as a
  feature would be label leakage. It's there for EDA/audit purposes (see
  Task 1's fault-type breakdown) and to make the streaming consumer's
  `[FAULT]` log lines more informative.


## Current regenerated metrics

The checked-in artifacts are generated from the current code. Do not copy metrics from an older report.

- Predictive maintenance: XGBoost selected; precision **68.9%**, recall **70.8%**, F1 **0.699**, ROC-AUC **0.952** at the training-tuned threshold **0.65**.
- Forecasting: direct 1–24h MAE **11.57 kWh**, RMSE **21.00 kWh**, MAPE **9.34%**; naive MAE **41.68 kWh**; MAE improvement **72.2%**.
- Anomaly detection: Isolation Forest **607**, z-score **1,210**, drift **588**, union **2,172** readings.
- Connectivity: **1** duplicate edge, **1** invalid parent reference, **1** orphan asset, **1** missing relationship.
