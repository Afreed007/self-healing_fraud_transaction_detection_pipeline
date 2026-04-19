"""
cly.py  –  Master CLI
----------------------
Commands
  train-base          Train baseline XGBoost on Batch 1 (unchanged)
  run-pipeline        Launch continuous producer + consumer pipeline
                      (runs until you press Ctrl-C)

Usage
  python cly.py train-base
  python cly.py run-pipeline
"""

import argparse
import os
import time
import threading
import pandas as pd
import joblib
from xgboost import XGBClassifier

from src.data_pipeline.data_prep import create_batch_if_missing
from src.llm_mapping.schema_adapter import LLMSchemaMapper
from src.streaming.pipeline_state import PipelineState
from src.streaming.producer import stream_continuous
from src.streaming.consumer import run_pipeline


def cmd_train_base():
    print("🧹 Preparing Batch 1 (1 M records) for base training …")
    batch_1_path = create_batch_if_missing(1)

    print("🏋️  Training Baseline XGBoost …")
    df     = pd.read_csv(batch_1_path)
    mapper = LLMSchemaMapper()
    X      = mapper.heal_schema(df.drop(columns=["isFraud", "isFlaggedFraud"],
                                        errors="ignore"))
    y      = df["isFraud"]

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X, y)

    os.makedirs("models/baselines", exist_ok=True)
    joblib.dump(model, "models/baselines/baseline_xgb.pkl")
    print("✅ Baseline trained → models/baselines/baseline_xgb.pkl")


def cmd_run_pipeline():
    """
    Launch producer and consumer as two threads sharing a PipelineState.
    Both run until Ctrl-C sets stop_event.
    """
    print("🚀 Starting continuous pipeline  (Ctrl-C to stop) …")

    state = PipelineState()

    # ── Consumer thread (inference + background retraining) ──────────
    consumer_thread = threading.Thread(
        target=run_pipeline,
        args=(state,),
        name="Consumer",
        daemon=False,
    )

    # ── Producer thread (continuous Kafka streaming) ──────────────────
    producer_thread = threading.Thread(
        target=stream_continuous,
        args=(state,),
        name="Producer",
        daemon=False,
    )

    consumer_thread.start()
    time.sleep(2)           # give consumer a moment to connect to Kafka
    producer_thread.start()

    # ── Live status printer ──────────────────────────────────────────
    def _status():
        while not state.stop_event.is_set():
            time.sleep(10)
            snap = state.snapshot()
            print(
                f"📊 Status | processed: {snap['total_processed']:,}  "
                f"fraud: {snap['live_fraud']:,}  "
                f"legit: {snap['live_legit']:,}  "
                f"batches: {snap['batch_counter']}  "
                f"model_v: {snap['model_version']}  "
                f"bg_busy: {snap['bg_busy']}  "
                f"buffer: {snap['buffer_fill']}/{snap['buffer_size']}"
            )

    status_thread = threading.Thread(target=_status, daemon=True)
    status_thread.start()

    try:
        producer_thread.join()
        consumer_thread.join()
    except KeyboardInterrupt:
        print("\n⛔  Ctrl-C — signalling shutdown …")
        state.stop_event.set()
        producer_thread.join(timeout=10)
        consumer_thread.join(timeout=60)

    print("✅ Pipeline stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Fraud Detection Pipeline")
    parser.add_argument(
        "command",
        choices=["train-base", "run-pipeline"],
        help="Command to execute",
    )
    args = parser.parse_args()

    if args.command == "train-base":
        cmd_train_base()
    elif args.command == "run-pipeline":
        cmd_run_pipeline()