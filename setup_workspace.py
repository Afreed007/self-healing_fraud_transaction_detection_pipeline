import os

# Define the directory structure
folders = [
    "data/raw",              # Original PaySim CSV
    "data/processed",        # X_baseline, y_baseline
    "data/stream_buffer",    # Temporary storage for Kafka batches
    "models/baselines",      # XGBoost .pkl files
    "models/rl_agents",      # DQN / StableBaselines models
    "notebooks",             # EDA.ipynb
    "config",                # settings.yaml (Centralized paths)
    "src/data_pipeline",     # __init__.py, preprocessing.py, kafka_producer.py
    "src/drift_detection",   # ks_test.py
    "src/llm_mapping",       # schema_adapter.py (GPT-4 logic)
    "src/rl_agent",          # training_agent.py
    "src/streaming",         #  producer.py,
    "src/utils",             # logger.py, helpers.py
    "airflow/dags",          # Orchestration
    "reports/benchmarks",    # Performance plots
    "logs"                   # System logs
]

def create_structure():
    print("🏗️ Building Professional Project Structure...")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        # Create __init__.py in all 'src' subfolders to make them importable packages
        if "src" in folder or "airflow" in folder:
            with open(os.path.join(folder, "__init__.py"), "w") as f:
                pass
        # Standard .gitkeep for empty data folders
        else:
            with open(os.path.join(folder, ".gitkeep"), "w") as f:
                pass
    
    # Create a basic config file to avoid hardcoding paths later
    config_content = """
paths:
  raw_data: "data/raw/PS_20174392719_1491204439457_log.csv"
  processed_dir: "data/processed/"
  model_dir: "models/baselines/"
kafka:
  topic: "fraud-transactions"
  bootstrap_servers: ["localhost:9092"]
"""
    with open("config/settings.yaml", "w") as f:
        f.write(config_content.strip())
        
    print("✅ Project directory structure and config/settings.yaml created.")

requirements = """
# Data & ML
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
scipy
pyyaml

# Drift & LLM
evidently
openai
python-dotenv

# Reinforcement Learning
gymnasium
stable-baselines3
shimmy>=2.0

# Orchestration & Streaming
apache-airflow
mlflow
kafka-python

# Visualization
matplotlib
seaborn
jupyter
"""

def create_requirements():
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    print("✅ requirements.txt updated with hybrid pipeline dependencies.")

if __name__ == "__main__":
    create_structure()
    create_requirements()