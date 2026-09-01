# Streaming Ingestion Demo (Kafka)

Every task script in `src/` reads `sensor_telemetry.csv` directly, which is
the right call for offline EDA/model development but isn't how telemetry
would actually arrive from a real IoT fleet. This folder demonstrates the
ingestion side of that architecture: a producer that publishes readings onto
a Kafka topic, and a consumer that subscribes and logs/persists them.

**Scope note:** this is an architecture demo, not a wired-up real-time
scoring pipeline. The consumer logs each reading and writes it to a local
SQLite table so you can see messages actually flowing end to end; it doesn't
call the trained `models/*.joblib` files (that's what `api/main.py` is for,
on a request/response basis). Extending the consumer to also score each
message through the predictive-maintenance model would mean re-implementing
`api/main.py`'s feature-construction logic inside a long-running consumer
loop - worth doing before this ever became a real service, but redundant for
what's meant to be a bonus ingestion-pattern demo.

## Architecture

```
sensor_telemetry.csv --> producer.py --> Kafka topic --> consumer.py --> SQLite
                          (replays          "nectar.                    (ingested_
                           historical        telemetry.                  telemetry.
                           readings in       raw",                       db)
                           timestamp         keyed by
                           order)            asset_id
```

Messages are keyed by `asset_id`, so Kafka's default partitioning keeps all
of a given asset's readings in the same partition and therefore in order -
important for anything downstream that computes rolling/lag features per
asset, the same leak-free-by-construction principle the task scripts use.

## Run it

1. **Start a local Kafka broker:**
   ```bash
   cd streaming
   docker compose up -d
   ```
   This starts a single-broker Kafka (KRaft mode, no separate Zookeeper
   container) on `localhost:9092`, plus an optional web UI at
   `http://localhost:8080` for browsing the topic while the demo runs.

2. **Install the Python client:**
   ```bash
   pip install kafka-python
   ```
   (already listed in the project's `requirements.txt`)

3. **Start the consumer first** (so it doesn't miss the producer's messages
   on this first run - or pass `--from-beginning` to the consumer to replay
   everything retained on the topic regardless of when you start it):
   ```bash
   python consumer.py
   ```

4. **In a second terminal, start the producer:**
   ```bash
   python producer.py
   ```
   By default this replays all ~60k rows of `../data/sensor_telemetry.csv`
   in timestamp order at a fast but visible pace. Useful variants:
   ```bash
   python producer.py --asset AST_0001 --interval 0.2   # one asset, easy to watch
   python producer.py --interval 0 --loop               # fire-hose, loops forever
   ```

5. Watch the consumer terminal - it logs a running count, and prints every
   reading where `fault_flag == 1` immediately as it arrives. Check
   `streaming/ingested_telemetry.db` afterward (e.g. with `sqlite3` or
   `pandas.read_sql`) to see everything that was persisted.

## Tear down

```bash
docker compose down -v   # -v also removes the Kafka data volume
```

## Why Kafka (vs. a simpler queue) for this

The brief's schema is fundamentally a multi-producer (56 assets, each an
independent publisher), ordered-per-key stream, which is exactly Kafka's
sweet spot - partitioning by `asset_id` gives per-asset ordering for free,
multiple independent consumer groups could subscribe to the same topic
(one for feature-store writes, one for real-time anomaly scoring, one for
a live dashboard) without the producer knowing or caring, and the topic
naturally acts as a short-term replay buffer if a downstream consumer falls
behind or needs to reprocess - none of which a plain point-to-point queue
gives you as a default property.
