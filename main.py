import os
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Symptom Disease Prediction API")

# CORS: Allow Flutter/web frontend calls (development phase can use *)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and feature names
model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Request body model
class SymptomRequest(BaseModel):
    symptoms: List[str]  # e.g. ["fever", "cough", "fatigue"]

# Production log file (for drift detection)
LOG_FILE = "production_log.csv"

def log_production_data(vector_row: dict):
    """Append production request data to CSV for drift detection."""
    df = pd.DataFrame([vector_row])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

@app.post("/predict")
def predict(request: SymptomRequest):
    # 1. Convert symptom name list to 132-dimensional binary vector
    # Normalize: lowercase and remove spaces, to match column names
    input_vector = {col: 0 for col in feature_columns}
    matched = []
    for s in request.symptoms:
        s_clean = s.strip().lower().replace(" ", "_")
        if s_clean in input_vector:
            input_vector[s_clean] = 1
            matched.append(s_clean)

    if not matched:
        return {"error": "No valid symptoms matched. Check symptom names.", "available_example": feature_columns[:5]}

    # 2. Predict
    X_input = pd.DataFrame([input_vector])
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    confidence = float(max(proba))

    # 3. Log production data (for subsequent drift detection)
    log_production_data(input_vector)

    return {
        "disease": pred,
        "confidence": round(confidence, 4),
        "matched_symptoms": matched
    }