"""
pipeline_state.py
-----------------
Thread-safe shared state between the inference loop and the
background retraining loop. Uses a filelock-free design:
  - active_model_path  : path to the .pkl the inference thread loads
  - model_version      : integer bumped on every hot-swap
  - rolling_buffer     : list of raw dicts (last N records)
  - batch_counter      : how many virtual batches have completed
  - total_processed    : total records predicted since start
  - stats              : per-virtual-batch metrics written by bg thread
  - stop_event         : threading.Event to signal clean shutdown
"""

import threading
import time


class PipelineState:
    BUFFER_SIZE = 10_000          # records per virtual batch
    STREAM_RATE  = 500            # target records/sec from producer

    def __init__(self):
        self._lock = threading.Lock()

        # ── Model hot-swap ──────────────────────────────────────────
        self.active_model_path = "models/baselines/baseline_xgb.pkl"
        self.model_version     = 0          # bumped when bg thread saves new model

        # ── Rolling buffer  ─────────────────────────────────────────
        self.rolling_buffer: list[dict] = []

        # ── Counters ────────────────────────────────────────────────
        self.batch_counter    = 0           # completed virtual batches
        self.total_processed  = 0           # records predicted so far

        # ── Per-batch results written by bg thread ──────────────────
        # Each entry:  { batch_id, ts, drift, action, reward,
        #                f1_pre, f1_post, precision, recall,
        #                schema_healed, n_records }
        self.batch_results: list[dict] = []

        # ── Live inference stats (updated every record) ─────────────
        self.live_fraud_count   = 0
        self.live_legit_count   = 0
        self.live_preds_per_sec = 0.0       # EMA of throughput

        # ── Control ─────────────────────────────────────────────────
        self.stop_event        = threading.Event()
        self.bg_busy           = False      # True while retraining

    # ── Thread-safe helpers ─────────────────────────────────────────

    def add_record(self, record: dict):
        with self._lock:
            self.rolling_buffer.append(record)
            self.total_processed += 1

    def flush_buffer(self) -> list[dict]:
        """Return and clear the rolling buffer atomically."""
        with self._lock:
            batch = list(self.rolling_buffer)
            self.rolling_buffer.clear()
            self.batch_counter += 1
            return batch

    def swap_model(self, new_path: str):
        with self._lock:
            self.active_model_path = new_path
            self.model_version    += 1

    def get_model_path(self) -> str:
        with self._lock:
            return self.active_model_path

    def append_result(self, result: dict):
        with self._lock:
            self.batch_results.append(result)

    def get_results(self) -> list[dict]:
        with self._lock:
            return list(self.batch_results)

    def increment_live(self, prediction: int):
        with self._lock:
            if prediction == 1:
                self.live_fraud_count += 1
            else:
                self.live_legit_count += 1

    def snapshot(self) -> dict:
        """Lightweight read-only snapshot for the dashboard."""
        with self._lock:
            return {
                "total_processed"  : self.total_processed,
                "batch_counter"    : self.batch_counter,
                "live_fraud"       : self.live_fraud_count,
                "live_legit"       : self.live_legit_count,
                "model_version"    : self.model_version,
                "bg_busy"          : self.bg_busy,
                "buffer_fill"      : len(self.rolling_buffer),
                "buffer_size"      : self.BUFFER_SIZE,
            }