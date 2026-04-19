"""
producer.py  –  Continuous Kafka producer
------------------------------------------
Streams records one-by-one (not in chunk-blobs) at a controlled
rate.  Loops through batch CSVs in order (batch_1, batch_2, …)
and cycles back to batch_2 once the dataset is exhausted, so the
pipeline never stops unless the stop_event is set externally or
Ctrl-C is pressed.

Key differences from the old producer
  • Sends each row individually  → consumer can predict per-record
  • Injects schema / concept drift via data_prep (same as before)
  • Rate-limited to PipelineState.STREAM_RATE  records/sec (default 500)
  • Publishes a lightweight  {"BATCH_START": batch_id}  sentinel at the
    start of every CSV so the consumer knows which batch is in-flight
  • On stop_event or KeyboardInterrupt it sends {"STREAM_END": True}
"""

import json
import time
import os
import threading

import pandas as pd
from kafka import KafkaProducer

from src.data_pipeline.data_prep import create_batch_if_missing
from src.streaming.pipeline_state import PipelineState


def _build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=["127.0.0.1:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        api_version=(7, 4, 0),
        # Small batching for low-latency per-record delivery
        linger_ms=5,
        batch_size=16_384,
    )


def stream_continuous(state: PipelineState, max_batches: int = 0):
    """
    Stream records continuously until state.stop_event is set.

    Parameters
    ----------
    state       : shared PipelineState object
    max_batches : stop after this many batch files (0 = run forever)
    """
    producer   = _build_producer()
    rate        = state.STREAM_RATE          # records / second
    interval    = 1.0 / rate                 # seconds / record

    batch_id    = 1
    batches_run = 0

    print(f"📡 [Producer] Starting continuous stream at {rate} records/sec …")

    try:
        while not state.stop_event.is_set():
            # ── 1. Prepare the batch CSV (creates / caches on disk) ──
            batch_path = create_batch_if_missing(batch_id)

            print(f"📡 [Producer] Streaming batch {batch_id} → {batch_path}")

            # Notify consumer which batch is starting
            producer.send("fraud-transactions", {"BATCH_START": batch_id})
            producer.flush()

            # ── 2. Stream row-by-row ─────────────────────────────────
            chunk_iter = pd.read_csv(batch_path, chunksize=1_000)
            for chunk in chunk_iter:
                if state.stop_event.is_set():
                    break
                for _, row in chunk.iterrows():
                    if state.stop_event.is_set():
                        break

                    t0 = time.perf_counter()
                    producer.send("fraud-transactions", row.to_dict())

                    # Rate limiting: sleep for remainder of interval
                    elapsed = time.perf_counter() - t0
                    sleep   = interval - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

            producer.flush()

            # ── 3. Advance to next batch (cycle: 1 → 2 → 3 → 2 → 3 …)
            batches_run += 1
            if max_batches and batches_run >= max_batches:
                print("📡 [Producer] Reached max_batches limit — stopping.")
                state.stop_event.set()
                break

            # Cycle: after batch 1 (baseline), alternate batch 2 & 3
            # so we always have schema drift (even) and concept drift (odd≥3)
            if batch_id == 1:
                batch_id = 2
            elif batch_id == 2:
                batch_id = 3
            else:
                batch_id = 2          # cycle between 2 and 3 indefinitely

    except KeyboardInterrupt:
        print("📡 [Producer] Interrupted by user.")
    finally:
        producer.send("fraud-transactions", {"STREAM_END": True})
        producer.flush()
        producer.close()
        print("📡 [Producer] Shut down cleanly.")