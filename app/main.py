from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .utils import load_retail_model, log_prediction

app = FastAPI(title="NeuralRetail Advanced API")

# Load your models when the app starts
churn_model = load_retail_model("models/churn_xgb.pkl")

# Define what the input data looks like (Humanized for your project)
class CustomerData(BaseModel):
    recency: int
    frequency: int
    monetary: float
    diversity_score: float

@app.get("/")
def home():
    return {"message": "NeuralRetail API is Online"}

@app.post("/predict/churn")
def predict_churn(data: CustomerData):
    if not churn_model:
        raise HTTPException(status_code=500, detail="Churn model not found in /models folder")
    
    # Convert input to the format the model expects
    input_features = [[data.recency, data.frequency, data.monetary, data.diversity_score]]
    
    # Get the prediction
    prediction = churn_model.predict(input_features)[0]
    probability = churn_model.predict_proba(input_features)[0][1]
    
    # Log it for MLOps tracking
    log_prediction(data.dict(), float(probability), "XGBoost_Churn")
    
    return {
        "will_churn": "Yes" if prediction == 1 else "No",
        "churn_probability": f"{round(probability * 100, 2)}%"
    }