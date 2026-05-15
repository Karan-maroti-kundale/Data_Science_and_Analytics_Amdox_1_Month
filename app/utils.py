import joblib
import json
import pandas as pd
from datetime import datetime

def load_retail_model(file_path):
    """Loads your trained AI model safely."""
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"Error loading model at {file_path}: {e}")
        return None

def log_prediction(input_data, prediction, model_type):
    """Saves every prediction to a file so we can check for 'Data Drift' later."""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_type,
        "input": input_data,
        "result": prediction
    }
    with open("logs/prediction_audit.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")