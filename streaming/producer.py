"""
Nectar Streaming Ingestion — Producer

Replays sensor_telemetry.csv onto a Kafka topic, one message per reading, in
timestamp order, as a stand-in for how real field devices would publish
readings via an MQTT-to-Kafka bridge or a gateway's Kafka client.

Why this exists: every task script in src/ reads a static CSV, which is the
right call for offline model development but isn't how telemetry actually
arrives. This demonstrates the ingestion side of the architecture the rest
of the project assumes - a topic that Task 2's rolling-feature pipeline and
Task 4's anomaly scoring would, in a real deployment, consume from instead
of a file.

Usage:
    python producer.py                          # replay at 200 msgs/sec
    python producer.py --interval 0.5            # slower, easier to watch
    python producer.py --interval 0 --loop       # fire-hose, loop forever
    python producer.py --asset AST_0001           # only replay one asset
    python producer.py --bootstrap-servers localhost:9092 --topic nectar.telemetry.raw

Requires a running Kafka broker - see streaming/README.md for
`docker compose up`. Requires `pip install kafka-python`.
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from kafka import KafkaProducer
except ImportError as e:
    raise SystemExit(
        f"Kafka import failed: {e}\n"
        "Install kafka-python with: python -m pip install kafka-python"
    ) from e

DEFAULT_TOPIC = "nectar.telemetry.raw"
DEFAULT_CSV = str(Path(__file__).resolve().parents[1] / "data" / "sensor_telemetry.csv")


def load_readings(csv_path, asset_filter=None):
    """Load telemetry rows, sorted by timestamp, as plain dicts (JSON-safe)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if asset_filter:
        rows = [r for r in rows if r["asset_id"] == asset_filter]
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def to_message(row):
    """Coerce the CSV's string fields back to numeric types for a clean
    JSON payload - the CSV reader gives us strings for everything."""
    msg = dict(row)
    for col in ["temperature", "humidity", "pressure", "vibration", "power_consumption"]:
        v = msg.get(col, "")
        msg[col] = float(v) if v not in ("", None) else None
    msg["occupancy_count"] = int(msg["occupancy_count"]) if msg.get("occupancy_count") not in ("", None) else None
    msg["fault_flag"] = int(msg["fault_flag"]) if msg.get("fault_flag") not in ("", None) else 0
    msg["produced_at"] = datetime.now(timezone.utc).isoformat()
    return msg


def run(args):
    try:
        producer = KafkaProducer(
            bootstrap_servers=args.bootstrap_servers.split(","),
            key_serializer=lambda k: k.encode("utf-8"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            linger_ms=5,
        )
    except Exception as e:
        raise SystemExit(
            f"Couldn't connect to Kafka at {args.bootstrap_servers}.\n"
            f"Error: {e}\n"
            f"Start Kafka first - see streaming/README.md (`docker compose up`)."
        )

    readings = load_readings(args.csv, asset_filter=args.asset)
    print(f"Loaded {len(readings)} readings"
          f"{' for ' + args.asset if args.asset else ''} from {args.csv}")
    print(f"Publishing to topic '{args.topic}' on {args.bootstrap_servers} "
          f"(key = asset_id, so per-asset ordering is preserved within a partition)")

    sent = 0
    try:
        while True:
            for row in readings:
                msg = to_message(row)
                producer.send(args.topic, key=row["asset_id"], value=msg)
                sent += 1
                if sent % 500 == 0:
                    producer.flush()
                    print(f"  ...{sent} messages sent (latest: {row['asset_id']} @ {row['timestamp']})")
                if args.interval > 0:
                    time.sleep(args.interval)
            producer.flush()
            print(f"Finished one pass: {sent} messages sent total.")
            if not args.loop:
                break
            print("--loop set, replaying from the start...")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        producer.flush()
        producer.close()
        print(f"Producer closed. {sent} messages sent this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay Nectar telemetry onto a Kafka topic.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to sensor_telemetry.csv")
    parser.add_argument("--asset", default=None, help="Only replay one asset_id (for a focused demo)")
    parser.add_argument("--interval", type=float, default=0.005,
                         help="Seconds to sleep between messages (0 = as fast as possible)")
    parser.add_argument("--loop", action="store_true", help="Replay the file on a loop instead of stopping")
    run(parser.parse_args())
