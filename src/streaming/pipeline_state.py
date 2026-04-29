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

Resume behaviour
----------------
On __init__, _resume() scans reports/benchmarks/ for existing
batch_N_report.txt files and sets batch_counter to the highest
N found. This means a pipeline restarted after processing 10 batches
will automatically number new batches from 11 onward, and the
dashboard will show the full history seamlessly.

It also checks whether a previously trained adaptive model exists
(models/baselines/active_rl_model.pkl) and uses it as the starting
active model instead of the baseline XGBoost, so the RL agent's
previous retraining work is not discarded on restart.
"""

import os
import threading


class PipelineState:
    BUFFER_SIZE = 10_000      # records per virtual batch
    STREAM_RATE = 500         # target records/sec from producer

    # Model paths in priority order for resume
    _ADAPTIVE_MODEL = "models/baselines/active_rl_model.pkl"
    _BASELINE_MODEL = "models/baselines/baseline_xgb.pkl"
    _REPORTS_DIR    = "reports/benchmarks"

    def __init__(self):
        self._lock = threading.Lock()

        # ── Model hot-swap ──────────────────────────────────────────
        self.active_model_path = self._resume_model()
        self.model_version     = 0

        # ── Rolling buffer  ─────────────────────────────────────────
        self.rolling_buffer: list[dict] = []

        # ── Counters ────────────────────────────────────────────────
        self.batch_counter   = self._resume_batch_counter()
        self.total_processed = 0

        # ── Per-batch results (in-memory, for live dashboard) ────────
        self.batch_results: list[dict] = []

        # ── Live inference stats ────────────────────────────────────
        self.live_fraud_count   = 0
        self.live_legit_count   = 0
        self.live_preds_per_sec = 0.0

        # ── Control ─────────────────────────────────────────────────
        self.stop_event = threading.Event()
        self.bg_busy    = False

    # ── Resume helpers ──────────────────────────────────────────────

    def _resume_batch_counter(self) -> int:
        """
        Scan reports/benchmarks/ for batch_N_report.txt files and
        return the highest N found, so new batches are numbered N+1.
        Returns 0 if no reports exist (fresh start).
        """
        if not os.path.isdir(self._REPORTS_DIR):
            return 0

        highest = 0
        for fname in os.listdir(self._REPORTS_DIR):
            if fname.startswith("batch_") and fname.endswith("_report.txt"):
                try:
                    n = int(fname.split("_")[1])
                    if n > highest:
                        highest = n
                except (IndexError, ValueError):
                    continue

        if highest > 0:
            print(f"♻️  [State] Resuming from batch {highest} → "
                  f"next batch will be {highest + 1}")
        else:
            print("🆕 [State] No existing reports found — starting fresh.")

        return highest

    def _resume_model(self) -> str:
        """
        Return the best available model path:
          1. active_rl_model.pkl  — most recently retrained adaptive model
          2. baseline_xgb.pkl     — original baseline (fallback)
        Logs which model is being resumed so the user knows.
        """
        if os.path.exists(self._ADAPTIVE_MODEL):
            print(f"♻️  [State] Resuming with last adaptive model: "
                  f"{self._ADAPTIVE_MODEL}")
            return self._ADAPTIVE_MODEL

        print(f"🆕 [State] No adaptive model found — "
              f"starting with baseline: {self._BASELINE_MODEL}")
        return self._BASELINE_MODEL

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
                "total_processed": self.total_processed,
                "batch_counter"  : self.batch_counter,
                "live_fraud"     : self.live_fraud_count,
                "live_legit"     : self.live_legit_count,
                "model_version"  : self.model_version,
                "bg_busy"        : self.bg_busy,
                "buffer_fill"    : len(self.rolling_buffer),
                "buffer_size"    : self.BUFFER_SIZE,
            }