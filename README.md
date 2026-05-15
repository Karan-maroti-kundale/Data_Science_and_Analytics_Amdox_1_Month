# NeuralRetail — AI Sales Intelligence Platform

**Project Type:** Amdox Technologies — First Month, First Project  
**Domain:** Data Science & Analytics  
**Project Focus:** Retail Intelligence, Predictive Analytics, and MLOps

NeuralRetail is an end-to-end retail analytics and machine learning platform built on the **Online Retail II** dataset.  
It transforms raw transaction data into a clean **Feature Master**, trains predictive models for **churn** and **demand forecasting**, serves predictions through **FastAPI**, and visualizes business insights in a **Streamlit dashboard**.

---

## Project Goals

This project is designed to simulate a production-grade retail intelligence system with:

- advanced feature engineering
- customer churn prediction
- demand forecasting
- executive KPI dashboards
- API-based model serving
- prediction logging for drift analysis
- Docker-based deployment

---

## Key Features

### 1. Advanced Feature Engineering
- RFM-style customer metrics
- Rolling 7-day and 30-day sales averages
- Lag features (`t-1`, `t-7`)
- Seasonality indicators
- Average days between purchases
- Product diversity score
- Inventory logic using burn rate and current stock

### 2. Machine Learning Models
- **XGBoost** for churn prediction
- **Prophet** for demand forecasting

### 3. FastAPI Serving Layer
- `/predict/churn`
- `/predict/demand`
- `/health`

### 4. Streamlit Executive Dashboard
- Executive Overview
- Customer Hub with churn prediction
- Data Drift Monitor

### 5. MLOps-Friendly Design
- model artifact saving
- prediction logging
- drift monitoring foundation
- container-ready structure

---

## Dataset

This project uses the **Online Retail II** dataset.

Recommended files:
- `OnlineRetailII.csv`
- `feature_master_online_retail_ii.csv`

The cleaned `feature_master_online_retail_ii.csv` is used for model training and dashboarding.

---

## Repository Structure

```text
NeuralRetail/
│
├── app/
│   ├── main.py
│   ├── utils.py
│   └── __init__.py
│
├── dashboard.py
├── feature_engineering_pipeline.py
├── train_churn_model.py
├── train_demand_model.py
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── churn_xgb.pkl
│   ├── demand_prophet.pkl
│   ├── demand_training_metadata.json
│   └── demand_training_series.csv
│
├── logs/
│   └── predictions.json
│
├── OnlineRetailII.csv
└── feature_master_online_retail_ii.csv

## Installation

1. Clone or open the project folder.

   Make sure all files are in one root directory.

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - Windows:
     ```powershell
     venv\Scripts\activate
     ```
   - Linux / macOS:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project

### Step 1: Train the demand model

If you already have `feature_master_online_retail_ii.csv`, use that as input.
Else you can download from `kaggle`.

```bash
python train_demand_model.py --input feature_master_online_retail_ii.csv
```

This saves:

- `models/demand_prophet.pkl`
- `models/demand_training_series.csv`
- `models/demand_training_metadata.json`

### Step 2: Start the FastAPI backend

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check the API health:

```text
http://127.0.0.1:8000/health
```

### Step 3: Start the Streamlit dashboard

```bash
streamlit run dashboard.py
```

The dashboard will connect to:

```text
http://127.0.0.1:8000/predict/churn
```