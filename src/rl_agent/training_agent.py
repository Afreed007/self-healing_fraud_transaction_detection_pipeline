import numpy as np
import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

class FraudRLAgent:
    def __init__(self, q_table_path="models/rl_q_table.json"):
        print("🧠 Initializing Q-Learning RL Agent...")
        self.q_table_path = q_table_path
        self.epsilon = 0.2  # 20% chance to explore a random action
        self.alpha = 0.5    # Learning Rate (How fast it learns)
        self.gamma = 0.9    # Discount Factor
        
        # Actions: 0=Ignore, 1=Retrain New, 2=Retrain Mix
        self.actions = [0, 1, 2] 
        self.q_table = self._load_q_table()

    def _load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'r') as f:
                return json.load(f)
        
        # OPTIMISTIC INITIALIZATION: 
        # If no drift, favor Action 0 (Ignore).
        # If Drift IS detected, heavily favor Actions 1 & 2 (Retrain).
        return {
            "Drift_0": [1.0, 0.0, 0.0], 
            "Drift_1": [-1.0, 1.0, 1.0]
        }

    def _save_q_table(self):
        os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
        with open(self.q_table_path, 'w') as f:
            json.dump(self.q_table, f, indent=4)

    def get_state(self, drift_detected):
        return "Drift_1" if drift_detected else "Drift_0"

    def choose_action(self, state):
        # Epsilon-Greedy Strategy
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
            print(f"🎲 RL Agent is EXPLORING: Chose Action {action}")
        else:
            action = np.argmax(self.q_table[state])
            print(f"🎯 RL Agent is EXPLOITING its knowledge: Chose Action {action}")
        return int(action)

    def execute_action(self, action, current_X, current_y, model_path="models/baselines/baseline_xgb.pkl"):
        action_names = {0: "Ignore Drift", 1: "Retrain on New Data", 2: "Retrain on Mixed Data"}
        print(f"🚀 Executing Action {action}: {action_names[action]}")

        if action == 0:
            # Do nothing, return the existing model
            return model_path
            
        elif action == 1:
            # Retrain on JUST the new streaming batch
            model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
            model.fit(current_X, current_y)
            
            new_model_path = "models/baselines/active_rl_model.pkl"
            joblib.dump(model, new_model_path)
            return new_model_path
            
        elif action == 2:
            print("📦 Loading baseline data to mix with current batch...")
            base_df = pd.read_csv("data/raw/batch_1.csv").sample(
                n=len(current_X), replace=True, random_state=42
            )
    
            from src.llm_mapping.schema_adapter import LLMSchemaMapper
            mapper = LLMSchemaMapper()
            base_X = mapper.heal_schema(
            base_df.drop(columns=['isFraud', 'isFlaggedFraud'], errors='ignore')
            )
            base_y = base_df['isFraud']

            mixed_X = pd.concat([base_X, current_X], ignore_index=True)
            mixed_y = pd.concat([base_y, pd.Series(current_y)], ignore_index=True)

            model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
            model.fit(mixed_X, mixed_y)
    
            new_model_path = "models/baselines/active_rl_model.pkl"
            joblib.dump(model, new_model_path)
            return new_model_path

    def learn(self, state, action, reward, next_state):
        # The Q-Learning Bellman Equation
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = predict + self.alpha * (target - predict)
        self._save_q_table()
        print(f"📈 Q-Table Updated! Reward: {reward:+.4f}. New Q-Value for Action {action}: {self.q_table[state][action]:.4f}")