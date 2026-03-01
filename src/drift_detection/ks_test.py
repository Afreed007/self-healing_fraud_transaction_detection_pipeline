import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os
import json

class DriftDetector:
    def __init__(self, baseline_path="data/processed/X_baseline.csv"):
        if not os.path.exists(baseline_path):
            raise FileNotFoundError("Baseline data not found. Run Phase 1 & 2 first.")
        
        print("📊 Loading Baseline Data for Drift Analysis...")
        self.baseline_df = pd.read_csv(baseline_path)
        # Use only a sample of baseline for speed if it's massive
        if len(self.baseline_df) > 100000:
            self.baseline_df = self.baseline_df.sample(100000, random_state=42)

    def run_ks_test(self, current_batch_path="data/stream_buffer/current_batch.csv"):
        if not os.path.exists(current_batch_path):
            return {"error": "No current batch found."}

        current_df = pd.read_csv(current_batch_path)
        drift_report = {}
        drift_detected = False

        print(f"🔬 Comparing {len(current_df)} new records against baseline...")

        # We only test numerical columns that affect model performance
        numerical_cols = self.baseline_df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if col in current_df.columns:
                # SAFEGUARD: Ensure both columns are numeric before testing
                if pd.api.types.is_numeric_dtype(self.baseline_df[col]) and pd.api.types.is_numeric_dtype(current_df[col]):
                    # KS Test: Returns (statistic, p-value)
                    stat, p_val = ks_2samp(self.baseline_df[col].dropna(), current_df[col].dropna())
                    
                    is_drifted = p_val < 0.05
                    drift_report[col] = {
                        "p_value": round(float(p_val), 6),
                        "drift_detected": bool(is_drifted)
                    }
                    
                    if is_drifted:
                        print(f"🚩 DRIFT DETECTED in feature: {col} (p={p_val:.6f})")
                        drift_detected = True

        return drift_detected, drift_report

if __name__ == "__main__":
    detector = DriftDetector()
    detected, report = detector.run_ks_test()
    
    if detected:
        print("\n🚨 SYSTEM ALERT: Data Drift detected. Retraining highly recommended.")
    else:
        print("\n✅ System Stable: Data distribution matches baseline.")