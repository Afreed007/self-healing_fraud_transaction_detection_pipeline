import time
import json
import pandas as pd
from kafka import KafkaProducer

def stream_batch(batch_file):
    print(f"📡 [Producer] Starting stream for {batch_file} at 5,000 txns/sec...")
    producer = KafkaProducer(
        bootstrap_servers=['127.0.0.1:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        api_version=(7, 4, 0)
    )
    
    for chunk in pd.read_csv(batch_file, chunksize=5000):
        records = chunk.to_dict(orient='records')
        start_time = time.time()
        
        for record in records:
            producer.send('fraud-transactions', record)
        producer.flush()
        
        elapsed = time.time() - start_time
        if elapsed < 1.0: time.sleep(1.0 - elapsed)
            
    producer.send('fraud-transactions', {"END_OF_BATCH": True})
    producer.flush()
    print("📡 [Producer] Finished streaming all records.")