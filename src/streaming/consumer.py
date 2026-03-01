import os
import json
import pandas as pd
import joblib
from kafka import KafkaConsumer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.llm_mapping.schema_adapter import LLMSchemaMapper
from src.drift_detection.ks_test import DriftDetector
from src.rl_agent.training_agent import FraudRLAgent

def evaluate_model(model_path, X, y_true):
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        return {
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 4)
        }
    except Exception as e:
        return {"Error": f"Model error: {e}"}

def process_batch(batch_id):
    print("👂 [Consumer] Listening for incoming transactions...")
    consumer = KafkaConsumer(
        'fraud-transactions', bootstrap_servers=['127.0.0.1:9092'],
        auto_offset_reset='latest', value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        api_version=(7, 4, 0), consumer_timeout_ms=15000
    )
    
    mapper = LLMSchemaMapper()
    detector = DriftDetector()
    rl_agent = FraudRLAgent()
    raw_data = []
    
    for message in consumer:
        if "END_OF_BATCH" in message.value: break
        raw_data.append(message.value)

    print("\n🛑 Stream ended. Triggering ML Pipeline...")
    df = pd.DataFrame(raw_data)
    
    print("🧠 Step 1: Schema Healing (LLM -> Preprocessing)...")
    # This invokes the two-step shield!
    X_clean = mapper.heal_schema(df.drop(columns=['isFraud', 'isFlaggedFraud'], errors='ignore'))
    y_true = df['isFraud'] if 'isFraud' in df.columns else [0]*len(df)

    current_model = "models/baselines/active_rl_model.pkl"
    if not os.path.exists(current_model): current_model = "models/baselines/baseline_xgb.pkl"

    print(f"📉 Step 2: Evaluating Current Model...")
    old_metrics = evaluate_model(current_model, X_clean, y_true)

    print("📊 Step 3: Checking for Concept Drift...")
    os.makedirs("data/stream_buffer", exist_ok=True)
    buffer_path = "data/stream_buffer/current_batch.csv"
    X_clean.to_csv(buffer_path, index=False)
    drift_detected, drift_report = detector.run_ks_test(buffer_path)

    print("🤖 Step 4: Reinforcement Learning Decision...")
    state = rl_agent.get_state(drift_detected)
    action = rl_agent.choose_action(state)
    new_model_path = rl_agent.execute_action(action, X_clean, y_true, current_model)
    
    new_metrics = evaluate_model(new_model_path, X_clean, y_true)
    
    old_f1 = old_metrics.get("F1 Score", 0)
    new_f1 = new_metrics.get("F1 Score", 0)
    reward = new_f1 - old_f1 
    
    if action == 0 and drift_detected and new_f1 < 0.5: reward -= 2.0 
    elif new_f1 > 0.8: reward += 1.0
    elif action != 0 and reward <= 0.001: reward -= 0.2 

    next_state = "Drift_0" if new_f1 > 0.8 else "Drift_1"
    rl_agent.learn(state, action, reward, next_state)

    print(f"\n📄 BATCH {batch_id} FINAL REPORT Generated.")
    os.makedirs("reports/benchmarks", exist_ok=True)
    with open(f"reports/benchmarks/batch_{batch_id}_report.txt", "w") as f:
        f.write(f"BATCH {batch_id} REPORT\nDrift: {drift_detected}\nAction: {action}\nReward: {reward}\nOld: {old_metrics}\nNew: {new_metrics}")