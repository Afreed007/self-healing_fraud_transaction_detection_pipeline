# 🛡️ Autonomous MLOps Pipeline: Adaptive Fraud Detection
**Self-Healing Schema and Concept Drift via LLM, XGBoost, & Reinforcement Learning**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Kafka](https://img.shields.io/badge/Apache_Kafka-Event_Streaming-red.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-XGBoost-orange.svg)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement_Learning-Q--Learning-20B2AA.svg)
![LLM-API](https://img.shields.io/badge/GenAI-Google_Gemini-8E75B2.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B.svg)

## 📖 Project Overview
Financial fraud detection systems typically fail in production due to two critical issues: **Schema Drift** (upstream data formats changing unexpectedly) and **Concept Drift** (fraudsters adapting their behaviors). 

This project is a high-throughput, autonomous MLOps pipeline designed to survive both. Streaming at 5,000 transactions per second via Apache Kafka, the system utilizes a **Generative AI Schema Adapter** to map unknown incoming data structures in real-time. When statistical Concept Drift is detected via Kolmogorov-Smirnov (KS) testing, a **Q-Learning Reinforcement Learning (RL) Agent** autonomously determines the optimal retraining strategy to recover model accuracy without human intervention.

Developed by **Mohammed Afreed Pasha** as a Master of Computer Applications (MCA) capstone project focusing on advanced Data Science, modular pipeline design, and scalable ML architectures.

---

## ✨ Core Architecture & Features

1. **High-Speed Data Streaming (Apache Kafka):**
   * Simulates real-world financial ecosystems by chunking and streaming massive datasets at 5,000 txns/sec via decoupled Producer and Consumer processes.
2. **The "Two-Step Shield" (LLM + Preprocessing):**
   * **Step 1 (LLM Healing):** Catches typos and renamed columns (Schema Drift) using a Gemini LLM fallback, caching vocabulary for $0 API costs and sub-millisecond latency.
   * **Step 2 (Feature Engineering):** Safely applies deterministic math and strips duplicate features to satisfy strict algorithmic requirements without causing LLM hallucinations.
3. **Advanced Baseline Modeling (XGBoost):**
   * Utilizes sequential gradient boosting (`XGBClassifier`) optimized with log-loss metrics to handle extreme class imbalance. 
4. **Statistical Concept Drift Detection:**
   * Runs KS-Tests on rolling data distributions to detect when the baseline machine learning model is no longer mathematically valid.
5. **Autonomous Self-Healing (Reinforcement Learning):**
   * A Q-Learning RL agent actively chooses whether to ignore drift, retrain on new data, or retrain on a mixed memory buffer, learning dynamically via reward shaping.
6. **Live Presentation Dashboard (Streamlit):**
   * Real-time web UI to visualize model decay, LLM schema healing, RL agent actions, and pipeline recovery.

---

## 📂 Enterprise-Grade Modular Structure

The pipeline is strictly decoupled by responsibility, routed through a central CLI.

```text
autonomous-fraud-pipeline/
│
├── data/
│   ├── processed/            # Engineered baseline features (X_baseline, y_baseline)
│   ├── raw/                  # Sliced batch CSVs & original log
│   └── stream_buffer/        # Temporary storage for streaming chunks
├── models/
│   ├── baselines/            # Saved active & baseline XGBoost models
│   └── rl_q_table.json       # The RL Agent's persistent memory matrix
├── reports/
│   └── benchmarks/           # Auto-generated text metrics for dashboarding
│
├── src/
│   ├── data_pipeline/
│   │   ├── data_prep.py      # Low-RAM dataset slicing & drift injection
│   │   └── preprocessing.py  # Cleans raw data and generates baseline CSVs
│   ├── streaming/
│   │   ├── producer.py       # Kafka message publishing
│   │   └── consumer.py       # Kafka ingestion & ML Pipeline execution
│   ├── llm_mapping/
│   │   └── schema_adapter.py # Multi-tier LLM & Fuzzy mapping logic
│   ├── drift_detection/
│   │   └── ks_test.py        # Kolmogorov-Smirnov statistical testing
│   └── rl_agent/
│       └── training_agent.py # Q-Learning agent and reward shaping
│
├── cly.py                    # Master Command Line Interface Router
├── app.py                    # Live Streamlit Dashboard
├── requirements.txt
└── README.md
```

## 🚀 Complete Setup Guide (From Scratch)
Because this repository contains strict .gitignore rules to prevent uploading 6M+ row CSVs and private API keys, follow these steps to initialize the environment locally.

## 1. Clone & Setup Virtual Environment
Clone the repository and create an isolated Python environment.

```
git clone [https://github.com/yourusername/automation-pipline.git](https://github.com/yourusername/automation-pipline.git)
cd automation-pipline
```

# Create and activate a conda/venv environment (example using conda)
```
conda create -n adaptive-pipeline python=3.10
conda activate adaptive-pipeline
```
# Install required dependencies
```
pip install -r requirements.txt
```
2. Download the Dataset
Due to size limits, the raw dataset is not tracked in version control.

Download the PaySim Mobile Money Fraud Dataset (PS_20174392719_1491204439457_log.csv) from Kaggle.

Place the downloaded CSV directly into the data/raw/ directory.

3. Configure Environment Variables
Create a .env file in the root directory and add your Google Gemini API key (required for the Schema Healing Agent):

Plaintext
GEMINI_API_KEY=your_api_key_here

4. Boot Up Kafka Infrastructure
Ensure Docker Desktop is running, then spin up the Apache Kafka and Zookeeper containers:
```
docker-compose up -d
(Optional: Run python setup_workspace.py if you need to initialize empty folder structures not tracked by Git).
```

🎮 Executing the Pipeline
Once the setup is complete, run the system in the following order:

Step 1: Data Preprocessing
Generate the foundational feature matrices (X_baseline.csv, y_baseline.csv) from the raw data.
```
python src/data_pipeline/preprocessing.py
```

Step 2: Train the Baseline XGBoost Model
Initializes the Two-Step Shield and trains the foundational XGBoost model.
```
python cly.py train-base
```

Step 3: Stream Batches & Trigger AI Agents
Stream subsequent batches (500k records each). The system will automatically inject schema drift (even batches) and massive concept drift (Batch 3+) to demonstrate the RL agent's autonomous recovery.
```
python cly.py run-batch --id 2
python cly.py run-batch --id 3
```

Step 4: Monitor Live via Dashboard
Open a separate terminal and launch the Streamlit UI to watch the AI decisions and F1 score metrics update in real-time as the batches process:
```
streamlit run app.py
```
