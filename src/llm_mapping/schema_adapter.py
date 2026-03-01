import pandas as pd
import json
import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

class LLMSchemaMapper:
    def __init__(self, baseline_path="data/processed/X_baseline.csv", cache_path="data/column_vocabulary.json"):
        self.cache_path = cache_path
        
        # Load baseline columns
        ref_df = pd.read_csv(baseline_path, nrows=1)
        self.expected_features = ref_df.columns.tolist()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Load the "Vocabulary" Cache
        self.vocab_cache = self._load_cache()

    def _load_cache(self):
        # Initialize with perfect matches (e.g., 'amount' maps to 'amount')
        cache = {col: col for col in self.expected_features}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                cache.update(json.load(f))
        return cache

    def _save_cache(self):
        with open(self.cache_path, 'w') as f:
            json.dump(self.vocab_cache, f, indent=4)

    def _engineer_features(self, df):
        # 1. ENCODE STRINGS TO NUMBERS (Fixes the 'PAYMENT' crash)
        type_mapping = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
        if 'type' in df.columns:
            # If the LLM misses a type or a new one appears, it defaults to -1
            df['type'] = df['type'].map(type_mapping).fillna(-1).astype(int)

        # 2. Safely engineer features only if the required base columns exist
        if 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
            df['is_all_out'] = ((df['amount'] == df['oldbalanceOrg']) & (df['amount'] != 0)).astype(int)
        else: 
            df['is_all_out'] = 0

        if all(col in df.columns for col in ['newbalanceOrig', 'oldbalanceOrg', 'amount']):
            df['error_balance_orig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        else: 
            df['error_balance_orig'] = 0
                
        return df

    def heal_schema(self, df):
        incoming_cols = df.columns.tolist()
        unknown_cols = [col for col in incoming_cols if col not in self.vocab_cache and col not in ['is_all_out', 'error_balance_orig']]

        if unknown_cols:
            print(f"⚠️ UNKNOWN COLUMNS DETECTED: {unknown_cols}. Asking Gemini...")
            
            prompt = f"""
            My machine learning model expects these exact features: {self.expected_features}.
            Incoming data has these unknown columns: {unknown_cols}.
            
            Return a flat JSON dictionary mapping the UNKNOWN columns to the EXPECTED features. 
            ONLY map columns with a clear semantic match or obvious typo (e.g., 'amout_val' -> 'amount').
            
            CRITICAL INSTRUCTIONS:
            1. 'step' is a time metric. 'nameOrig' and 'nameDest' are string IDs.
            2. You MUST NOT map strings to numeric features.
            3. If a column is 'step', 'nameOrig', 'nameDest', or 'isFraud', map it exactly to the word "IGNORE".
            """
            
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config={'response_mime_type': 'application/json'}
                )
                
                raw_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                new_mappings = json.loads(raw_text)
                
                if len(new_mappings) == 1 and isinstance(list(new_mappings.values())[0], dict):
                    new_mappings = list(new_mappings.values())[0]

                valid_mappings = {k: v for k, v in new_mappings.items() if v != "IGNORE" and v in self.expected_features}
                
                if valid_mappings:
                    self.vocab_cache.update(valid_mappings)
                    self._save_cache()
                    print(f"📚 LLM successfully mapped typos: {valid_mappings}")
                else:
                    print("💡 LLM correctly identified remaining columns as irrelevant.")
                
                ignored_mappings = {k: "IGNORE" for k, v in new_mappings.items() if v == "IGNORE"}
                self.vocab_cache.update(ignored_mappings)
                self._save_cache()
                
            except Exception as e:
                print(f"❌ Gemini API Error: {e}")

        safe_mapping = {k: v for k, v in self.vocab_cache.items() if v != "IGNORE"}
        df = df.rename(columns=safe_mapping)
        df = df.loc[:, ~df.columns.duplicated()]
        
        df = self._engineer_features(df)
        
        # 🛡️ THE FIX 1: Strip any duplicate columns that formed in the DataFrame
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 🛡️ THE FIX 2: Ensure our expected list doesn't accidentally have duplicates
        final_expected = self.expected_features + ['is_all_out', 'error_balance_orig']
        final_expected = list(dict.fromkeys(final_expected)) # Removes duplicate strings
        
        for col in final_expected:
            if col not in df.columns: df[col] = 0
                
        return df[final_expected]