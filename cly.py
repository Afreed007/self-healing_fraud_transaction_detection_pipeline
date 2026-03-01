import argparse
import time
import pandas as pd
import os
import joblib
from multiprocessing import Process
from xgboost import XGBClassifier

from src.data_pipeline.data_prep import create_batch_if_missing
from src.streaming.producer import stream_batch
from src.streaming.consumer import process_batch
from src.llm_mapping.schema_adapter import LLMSchemaMapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Adaptive Pipeline")
    parser.add_argument("command", choices=["train-base", "run-batch"], help="Command to execute")
    parser.add_argument("--id", type=int, help="Batch ID to process (e.g., 2, 3)")
    
    args = parser.parse_args()

    if args.command == "train-base":
        print("🧹 Preparing Batch 1 (1M records) for Base Training...")
        batch_1_path = create_batch_if_missing(1)
        
        print("🏋️ Training Baseline XGBoost Model...")
        df = pd.read_csv(batch_1_path)
        
        mapper = LLMSchemaMapper()
        X = mapper.heal_schema(df.drop(columns=['isFraud', 'isFlaggedFraud'], errors='ignore'))
        y = df['isFraud']
        
        # XGBoost Baseline Setup
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1, eval_metric='logloss')
        model.fit(X, y)
        
        os.makedirs("models/baselines", exist_ok=True)
        joblib.dump(model, "models/baselines/baseline_xgb.pkl")
        print("✅ Baseline trained and saved successfully!")

    elif args.command == "run-batch":
        if not args.id:
            print("❌ Please provide a batch ID: --id 2")
            exit()
            
        batch_path = create_batch_if_missing(args.id)

        consumer_process = Process(target=process_batch, args=(args.id,))
        producer_process = Process(target=stream_batch, args=(batch_path,))
        
        consumer_process.start()
        time.sleep(3) 
        producer_process.start()
        
        producer_process.join()
        consumer_process.join()