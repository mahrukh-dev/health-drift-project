# Symptom Triage Service with Drift-Aware Retraining

A production-style ML service that predicts diseases from structured symptom 
input, designed to be consumed by a Flutter mobile frontend. Includes 
production request logging and data drift monitoring.

## Architecture

Flutter app → FastAPI /predict → RandomForest model
                          ↓
                  production_log.csv → Evidently drift report

## Tech Stack
- Scikit-learn (RandomForest, class-weighted)
- FastAPI + Uvicorn
- Evidently (DataDriftPreset)
- Pandas / joblib

## Model Performance
- Accuracy: The model achieves a 100% F1-score on the test set. However, this is a reflection of the dataset's synthetic nature rather than real-world clinical performance. The dataset contains perfectly structured symptom combinations with zero noise. In a real-world scenario, patient data would be sparse, messy, and have overlapping symptoms, which would drastically reduce accuracy. The focus of this project is not the model's accuracy, but the MLOps architecture (API serving, logging, and drift detection) built around it.
- Class imbalance handled via class_weight="balanced"

## API Contract (for Flutter integration)
POST /predict
Request:  {"symptoms": ["fever", "cough", "fatigue"]}
Response: {"disease": "Influenza", "confidence": 0.91, "matched_symptoms": [...]}

CORS enabled for cross-origin frontend calls.

## Drift Monitoring
Every API call is logged to production_log.csv. `check_drift.py` compares 
this log against the training distribution (reference_data.csv) using 
Evidently's DataDriftPreset and outputs an HTML report.

![Drift report screenshot](drift_report.png)

## How to Run
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
# in another terminal:
python generate_traffic.py
python check_drift.py

## Limitations
- Dataset is a public Kaggle educational dataset, not clinical-grade.
- Drift is currently detected on demand, not scheduled.
- Retraining is manual; automation via GitHub Actions is planned.
- Free-tier deployment cold-start latency not yet benchmarked.