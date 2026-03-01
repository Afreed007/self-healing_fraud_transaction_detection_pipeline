import os
import pandas as pd

def create_batch_if_missing(batch_id):
    source_file = "data/raw/PS_20174392719_1491204439457_log.csv" 
    batch_file = f"data/raw/batch_{batch_id}.csv"
    
    if os.path.exists(batch_file) and os.path.getsize(batch_file) > 1000:
        return batch_file

    print(f"⏳ Slicing Batch {batch_id} efficiently (Low RAM mode)...")
    if not os.path.exists(source_file):
        print(f"❌ ERROR: Cannot find master dataset at {source_file}")
        exit()

    rows_to_read = 500000
    skip_rows_count = 0 if batch_id == 1 else 1000000 + ((batch_id - 2) * 500000)
    if batch_id == 1: rows_to_read = 1000000

    with open(source_file, 'r') as infile, open(batch_file, 'w') as outfile:
        header = infile.readline()
        outfile.write(header)
        for _ in range(skip_rows_count): infile.readline()
        for _ in range(rows_to_read):
            line = infile.readline()
            if not line: break 
            outfile.write(line)

    df = pd.read_csv(batch_file)
    if batch_id % 2 == 0:
        print("🌪️ Injecting Schema Drift (Typos)...")
        df = df.rename(columns={'amount': 'amout_val', 'oldbalanceOrg': 'old_bal_org'})
    if batch_id >= 3:
        print("🌋 Injecting HUGE Concept Drift...")
        if 'isFraud' in df.columns and 'amount' in df.columns:
            df.loc[df['isFraud'] == 1, 'amount'] = df.loc[df['isFraud'] == 1, 'amount'] * 10
            if 'type' in df.columns: df.loc[df['isFraud'] == 1, 'type'] = 'CASH_OUT'

    df.to_csv(batch_file, index=False)
    print(f"✅ Created {batch_file} successfully!")
    return batch_file