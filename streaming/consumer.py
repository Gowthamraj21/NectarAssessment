"""
Nectar Streaming Ingestion — Consumer

Subscribes to the telemetry topic the producer publishes to, logs each
message to the console, and persists it into a local SQLite table
(streaming/ingested_telemetry.db) as a lightweight stand-in for the
time-series feature store a real deployment would write into.

This is deliberately kept as an architecture demo (log + store), not wired
into Task 2/Task 4's actual trained models - see README.md "Known
limitations" for why that's a reasonable bonus-scope boundary here rather
than a real gap: scoring live traffic through a joblib model loaded in a
long-running consumer process is a fairly small extension of this same
script, but it duplicates the feature-engineering logic in api/main.py
rather than reusing it cleanly, which felt like scope creep for a bonus demo.

Usage:
    python consumer.py
    python consumer.py --bootstrap-servers localhost:9092 --topic nectar.telemetry.raw
    python consumer.py --from-beginning     # replay everything retained on the topic

Requires a running Kafka broker - see streaming/README.md for
`docker compose up`. Requires `pip install kafka-python`.
"""
import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


try:
    from kafka import KafkaConsumer
except ImportError as e:
    raise SystemExit(
        "kafka-python isn't installed. Run `python -m pip install kafka-python` "
        "(see streaming/README.md) before running the consumer."
    ) from e



DEFAULT_TOPIC = "nectar.telemetry.raw"
DB_PATH = str(Path(__file__).resolve().parent / "ingested_telemetry.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry_stream (
    asset_id TEXT,
    timestamp TEXT,
    site_id TEXT,
    building_id TEXT,
    temperature REAL,
    humidity REAL,
    pressure REAL,
    vibration REAL,
    power_consumption REAL,
    occupancy_count INTEGER,
    operating_mode TEXT,
    fault_flag INTEGER,
    fault_type TEXT,
    produced_at TEXT,
    consumed_at TEXT
)
"""

def init_db(path):
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_ts ON telemetry_stream(asset_id, timestamp)")
    conn.commit()
    return conn


def store(conn, msg):
    conn.execute(
        """INSERT INTO telemetry_stream
           (asset_id, timestamp, site_id, building_id, temperature, humidity,
            pressure, vibration, power_consumption, occupancy_count,
            operating_mode, fault_flag, fault_type, produced_at, consumed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            msg.get("asset_id"), msg.get("timestamp"), msg.get("site_id"), msg.get("building_id"),
            msg.get("temperature"), msg.get("humidity"), msg.get("pressure"), msg.get("vibration"),
            msg.get("power_consumption"), msg.get("occupancy_count"), msg.get("operating_mode"),
            msg.get("fault_flag"), msg.get("fault_type"), msg.get("produced_at"),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def run(args):
    try:
        consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=args.bootstrap_servers.split(","),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest" if args.from_beginning else "latest",
            enable_auto_commit=True,
            group_id=args.group_id,
            consumer_timeout_ms=args.idle_timeout_ms,
        )
    except Exception as e:
        raise SystemExit(
            f"Couldn't connect to Kafka at {args.bootstrap_servers}.\n"
            f"Error: {e}\n"
            f"Start Kafka first - see streaming/README.md (`docker compose up`)."
        )

    conn = init_db(args.db)
    print(f"Subscribed to '{args.topic}' on {args.bootstrap_servers} "
          f"(group_id={args.group_id}, from_beginning={args.from_beginning})")
    print(f"Storing readings into {args.db} (table: telemetry_stream)")
    print("Waiting for messages... (Ctrl+C to stop)\n")

    seen_per_asset = defaultdict(int)
    n_msgs = 0
    n_faults = 0
    commit_every = 50

    try:
        for record in consumer:
            msg = record.value
            n_msgs += 1
            seen_per_asset[msg.get("asset_id")] += 1
            store(conn, msg)

            flag = msg.get("fault_flag", 0)
            if flag:
                n_faults += 1
                print(f"[FAULT] {msg.get('asset_id')} @ {msg.get('timestamp')} "
                      f"type={msg.get('fault_type')} vibration={msg.get('vibration')}")
            elif n_msgs % 100 == 0:
                print(f"[ok] {n_msgs} messages consumed so far "
                      f"({len(seen_per_asset)} distinct assets, {n_faults} fault readings)")

            if n_msgs % commit_every == 0:
                conn.commit()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        conn.commit()
        conn.close()
        print(f"\nConsumer closed. {n_msgs} messages consumed, "
              f"{len(seen_per_asset)} distinct assets, {n_faults} fault readings. "
              f"Data persisted to {args.db}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume Nectar telemetry from Kafka and log/store it.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--group-id", default="nectar-ingestion-demo")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--from-beginning", action="store_true",
                         help="Replay everything retained on the topic instead of only new messages")
    parser.add_argument("--idle-timeout-ms", type=int, default=30000,
                         help="Stop after this many ms with no new messages (0 = never stop)")
    run(parser.parse_args())
