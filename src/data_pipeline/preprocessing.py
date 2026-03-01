import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

class AdaptiveDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def split_baseline_and_stream(self, baseline_size=1000000):
        print(f"✂️ Splitting dataset: {baseline_size} for Baseline, rest for Streaming...")
        # Load only the baseline part for now to save memory
        df_baseline = pd.read_csv(self.file_path, nrows=baseline_size)
        
        # Save the rest of the CSV indices for the Kafka Producer later
        # (This avoids loading the whole 6M into RAM)
        df_baseline.to_csv("data/raw/baseline_raw.csv", index=False)
        print("✅ baseline_raw.csv created.")
        return df_baseline

    def engineer_features(self, df):
        print("🛠 Engineering features...")
        df['is_all_out'] = (df['amount'] == df['oldbalanceOrg']).astype(int)
        df['error_balance_orig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['type'] = pd.get_dummies(df['type'], drop_first=True).iloc[:, 0] # Simplify for Baseline
        
        cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
        return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    def prepare_baseline(self, df):
        print("⚖️ Applying SMOTE to Baseline...")
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        
        # Use float32 to save memory
        X = X.astype('float32')
        
        smote = SMOTE(sampling_strategy=0.1, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

if __name__ == "__main__":
    RAW_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"
    proc = AdaptiveDataProcessor(RAW_PATH)
    
    # 1. Create the small baseline dataset
    base_df = proc.split_baseline_and_stream(baseline_size=1000000)
    
    # 2. Process and save for Training
    base_df = proc.engineer_features(base_df)
    X, y = proc.prepare_baseline(base_df)
    
    X.to_csv("data/processed/X_baseline.csv", index=False)
    y.to_csv("data/processed/y_baseline.csv", index=False)
    print("🚀 Baseline data ready at data/processed/X_baseline.csv")