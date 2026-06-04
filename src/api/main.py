import os
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException

# Align project layout structures to import from sister directories safely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.api.pydantic_models import TransactionRequest, PredictionResponse

app = FastAPI(
    title="Xente Alternative Credit Risk Scoring API",
    description="Containerized real-time predictive serving engine for evaluating default risk.",
    version="1.0.0"
)

# Global memory placeholder for the model object
model = None

@app.on_event("startup")
def load_champion_model():
    """Locates and loads the local tracked MLflow random forest model into runtime memory."""
    global model
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_dir = os.path.join(BASE_DIR, "mlruns", "1")
    
    try:
        # Pull subfolders while ignoring configuration files
        runs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d != "meta.yaml"]
        if not runs:
            raise FileNotFoundError("No active model training runs found inside the tracking repository store.")
        
        # Pick the most recent run artifact
        latest_run = max(runs, key=lambda d: os.path.getmtime(os.path.join(model_dir, d)))
        active_model_path = os.path.join(model_dir, latest_run, "artifacts", "model")
        
        model = mlflow.sklearn.load_model(active_model_path)
        print("🎉 Champion Random Forest model parsed and cached into API memory successfully!")
    except Exception as e:
        print("⚠️ Path auto-loading warning: " + str(e))
        print("Starting in safe structural fallback mode.")

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Xente Credit API Container"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(payload: TransactionRequest):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model engine is uninitialized or currently unavailable.")
    
    try:
        # Convert incoming data vector to standard features matching the model training structure
        timestamp = pd.to_datetime(payload.transactionstarttime)
        hour = timestamp.hour
        day = timestamp.day
        month = timestamp.month
        year = timestamp.year
        
        # Build vector array [amount, value, hour, day, month, year]
        feature_vector = np.array([[payload.amount, payload.value, hour, day, month, year]])
        
        # Predict class and target risk probabilities
        prediction = int(model.predict(feature_vector))
        probability = float(model.predict_proba(feature_vector))
        
        decision = "Deny Credit (High Risk)" if prediction == 1 else "Approve Credit (Low Risk)"
        
        return PredictionResponse(
            transaction_id=payload.transactionid,
            risk_score_probability=round(probability, 4),
            prediction_class=prediction,
            credit_decision=decision
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error compiling metrics for real-time scoring: " + str(e))