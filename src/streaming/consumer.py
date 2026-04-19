"""
consumer.py  –  Real-time inference + background retraining
------------------------------------------------------------
Two concurrent threads share a PipelineState object:

  Thread A  –  inference_loop()
    • Reads every Kafka message as it arrives
    • Pre-processes the record (LLM schema healer, feature engineering)
    • Predicts fraud / legit with the current active model  (<1 ms)
    • Appends the record to the rolling buffer
    • When buffer reaches BUFFER_SIZE → hands it to Thread B and clears

  Thread B  –  retraining_loop()  (daemon, started by Thread A)
    • Receives a completed buffer via a Queue (non-blocking for Thread A)
    • Runs KS drift test
    • Asks RL agent which action to take
    • Retrains (if needed) on the mini-batch  (~300 MB RAM, ~20 s)
    • Hot-swaps the model path in PipelineState
    • Writes a report file + updates state.batch_results for dashboard
    • Releases the "bg_busy" flag so Thread A can hand off next batch

Why a Queue and not a direct call?
  Thread A never blocks.  If Thread B is still retraining when the next
  buffer is ready, Thread A keeps predicting with the old model and the
  new buffer accumulates until Thread B is free (queue depth = 1 kept
  intentionally so we don't stack up stale data).
"""

import os
import json
import time
import queue
import threading
import traceback

import joblib
import pandas as pd
from kafka import KafkaConsumer
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)

from src.llm_mapping.schema_adapter import LLMSchemaMapper
from src.drift_detection.ks_test import DriftDetector
from src.rl_agent.training_agent import FraudRLAgent
from src.streaming.pipeline_state import PipelineState


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    fallback = "models/baselines/baseline_xgb.pkl"
    print(f"⚠️  Model {path} not found — using baseline.")
    return joblib.load(fallback)


def _evaluate(model, X, y_true) -> dict:
    try:
        y_pred = model.predict(X)
        return {
            "accuracy" : round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall"   : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1"       : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }
    except Exception as e:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "error": str(e)}


def _write_report(batch_id: int, result: dict):
    os.makedirs("reports/benchmarks", exist_ok=True)
    path = f"reports/benchmarks/batch_{batch_id}_report.txt"
    with open(path, "w") as f:
        f.write(f"BATCH {batch_id} REPORT\n")
        f.write(f"Timestamp: {result['ts']}\n")
        f.write(f"Records: {result['n_records']}\n")
        f.write(f"Drift: {result['drift']}\n")
        f.write(f"Schema_Healed: {result['schema_healed']}\n")
        f.write(f"Action: {result['action']}\n")
        f.write(f"Reward: {result['reward']}\n")
        f.write(f"Old: {{'F1 Score': {result['f1_pre']}}}\n")
        f.write(f"New: {{'F1 Score': {result['f1_post']}, "
                f"'Precision': {result['precision']}, "
                f"'Recall': {result['recall']}}}\n")
    print(f"📄 Report saved → {path}")


# ── Background retraining thread ─────────────────────────────────────────────

def _retraining_loop(work_queue: queue.Queue, state: PipelineState):
    """Daemon thread: picks up completed buffers and retrains."""
    mapper   = LLMSchemaMapper()
    detector = DriftDetector()
    rl_agent = FraudRLAgent()

    while not state.stop_event.is_set():
        try:
            # Block for up to 2 s, then re-check stop_event
            try:
                batch_id, raw_records, current_model_path = work_queue.get(timeout=2)
            except queue.Empty:
                continue

            state.bg_busy = True
            print(f"\n🔄 [BG] Processing virtual batch {batch_id} "
                  f"({len(raw_records)} records) …")

            # ── 1. Build DataFrame ───────────────────────────────────
            df = pd.DataFrame(raw_records)
            if df.empty:
                state.bg_busy = False
                work_queue.task_done()
                continue

            y_col    = "isFraud"
            y_true   = df[y_col].values if y_col in df.columns else [0] * len(df)
            drop_cols = [c for c in ["isFraud", "isFlaggedFraud"] if c in df.columns]

            # ── 2. Schema healing ────────────────────────────────────
            raw_cols    = set(df.columns)
            X_raw       = df.drop(columns=drop_cols, errors="ignore")
            X_clean     = mapper.heal_schema(X_raw)
            healed_cols = set(X_clean.columns)
            schema_healed = bool(raw_cols != healed_cols)

            # ── 3. Evaluate current model BEFORE any action ──────────
            current_model  = _load_model(current_model_path)
            pre_metrics    = _evaluate(current_model, X_clean, y_true)

            # ── 4. Drift detection ───────────────────────────────────
            os.makedirs("data/stream_buffer", exist_ok=True)
            buf_path = "data/stream_buffer/current_batch.csv"
            X_clean.to_csv(buf_path, index=False)
            drift_detected, drift_report = detector.run_ks_test(buf_path)

            # ── 5. RL agent decides action ───────────────────────────
            rl_state = rl_agent.get_state(drift_detected)
            action   = rl_agent.choose_action(rl_state)
            new_model_path = rl_agent.execute_action(
                action, X_clean, y_true, current_model_path
            )

            # ── 6. Evaluate NEW model ────────────────────────────────
            new_model    = _load_model(new_model_path)
            post_metrics = _evaluate(new_model, X_clean, y_true)

            # ── 7. Reward shaping ────────────────────────────────────
            reward = post_metrics["f1"] - pre_metrics["f1"]
            if action == 0 and drift_detected and post_metrics["f1"] < 0.5:
                reward -= 2.0
            elif post_metrics["f1"] > 0.8:
                reward += 1.0
            elif action != 0 and reward <= 0.001:
                reward -= 0.2

            next_rl_state = "Drift_0" if post_metrics["f1"] > 0.8 else "Drift_1"
            rl_agent.learn(rl_state, action, reward, next_rl_state)

            # ── 8. Hot-swap model in inference thread ─────────────────
            state.swap_model(new_model_path)
            print(f"✅ [BG] Model hot-swapped → {new_model_path}  "
                  f"(F1: {pre_metrics['f1']:.4f} → {post_metrics['f1']:.4f})")

            # ── 9. Record result ─────────────────────────────────────
            result = {
                "batch_id"     : batch_id,
                "ts"           : time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_records"    : len(raw_records),
                "drift"        : drift_detected,
                "schema_healed": schema_healed,
                "action"       : action,
                "reward"       : round(reward, 4),
                "f1_pre"       : pre_metrics["f1"],
                "f1_post"      : post_metrics["f1"],
                "precision"    : post_metrics["precision"],
                "recall"       : post_metrics["recall"],
            }
            state.append_result(result)
            _write_report(batch_id, result)

            state.bg_busy = False
            work_queue.task_done()

        except Exception:
            print("❌ [BG] Error in retraining loop:")
            traceback.print_exc()
            state.bg_busy = False
            try:
                work_queue.task_done()
            except Exception:
                pass


# ── Inference loop (main consumer thread) ────────────────────────────────────

def run_pipeline(state: PipelineState):
    """
    Main entry point.  Call this in a dedicated process/thread.
    Blocks until state.stop_event is set or STREAM_END arrives.
    """
    # ── 1. Shared objects ────────────────────────────────────────────
    mapper     = LLMSchemaMapper()
    work_queue : queue.Queue = queue.Queue(maxsize=1)

    # ── 2. Start background retraining daemon ────────────────────────
    bg_thread = threading.Thread(
        target=_retraining_loop,
        args=(work_queue, state),
        daemon=True,
        name="BG-Retrain",
    )
    bg_thread.start()
    print("🧵 [Consumer] Background retraining thread started.")

    # ── 3. Kafka consumer ────────────────────────────────────────────
    consumer = KafkaConsumer(
        "fraud-transactions",
        bootstrap_servers=["127.0.0.1:9092"],
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        api_version=(7, 4, 0),
        consumer_timeout_ms=5_000,       # 5 s idle before loop re-checks
        fetch_min_bytes=1,
        fetch_max_wait_ms=100,           # low latency
    )

    # ── 4. Per-record inference state ────────────────────────────────
    loaded_model_version = -1
    model                = None
    current_batch_id     = 0
    _t_last              = time.perf_counter()
    _preds_window        = 0             # for EMA throughput

    print("👂 [Consumer] Listening for transactions …")

    try:
        for message in consumer:
            if state.stop_event.is_set():
                break

            msg = message.value

            # ── Sentinel: stream ended ───────────────────────────────
            if "STREAM_END" in msg:
                print("🛑 [Consumer] Received STREAM_END.")
                state.stop_event.set()
                break

            # ── Sentinel: new batch starting ─────────────────────────
            if "BATCH_START" in msg:
                current_batch_id = msg["BATCH_START"]
                print(f"🔔 [Consumer] Batch {current_batch_id} starting.")
                continue

            # ── Hot-reload model if bg thread swapped it ─────────────
            current_ver = state.model_version
            if current_ver != loaded_model_version:
                path  = state.get_model_path()
                model = _load_model(path)
                loaded_model_version = current_ver
                print(f"🔄 [Consumer] Loaded model v{current_ver} ← {path}")

            if model is None:
                # Haven't loaded yet — skip
                state.add_record(msg)
                continue

            # ── Per-record preprocessing (lightweight) ───────────────
            try:
                row_df  = pd.DataFrame([msg])
                drop_c  = [c for c in ["isFraud", "isFlaggedFraud"] if c in row_df]
                X_row   = mapper.heal_schema(row_df.drop(columns=drop_c, errors="ignore"))
                pred    = int(model.predict(X_row)[0])
            except Exception:
                pred = 0        # default to legit on parse errors

            state.increment_live(pred)

            # ── Throughput EMA ───────────────────────────────────────
            _preds_window += 1
            now = time.perf_counter()
            if now - _t_last >= 1.0:
                state.live_preds_per_sec = _preds_window / (now - _t_last)
                _preds_window = 0
                _t_last       = now

            # ── Accumulate into rolling buffer ───────────────────────
            state.add_record(msg)

            # ── Virtual batch boundary ───────────────────────────────
            snap = state.snapshot()
            if snap["buffer_fill"] >= state.BUFFER_SIZE:
                if not work_queue.full():
                    raw_batch = state.flush_buffer()
                    model_at_dispatch = state.get_model_path()
                    bid = state.batch_counter      # incremented by flush
                    print(f"📦 [Consumer] Virtual batch {bid} → BG thread "
                          f"({len(raw_batch)} records)")
                    work_queue.put_nowait((bid, raw_batch, model_at_dispatch))
                else:
                    # BG thread still busy — extend the current buffer
                    # (records keep accumulating; we'll flush next time)
                    pass

    except KeyboardInterrupt:
        print("👂 [Consumer] Interrupted by user.")
    finally:
        consumer.close()
        state.stop_event.set()
        # Give BG thread a moment to finish its current retrain
        bg_thread.join(timeout=60)
        print("👂 [Consumer] Shut down cleanly.")