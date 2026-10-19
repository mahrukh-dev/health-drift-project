# Symptom Triage Service with Drift-Aware Retraining

A production-style Machine Learning service that predicts diseases from structured symptom input, designed to be consumed by a Flutter mobile frontend. It includes production request logging, data drift monitoring, and an automated CI/CD pipeline.

##  Live Demo
**[Test the Live API here](https://health-drift-project.onrender.com/docs)** 
*(Note: Hosted on Render's free tier. The server spins down after 15 minutes of inactivity, so the first request may take ~30 seconds to cold-start.)*

##  Architecture

1. **Client (Flutter/Web)** sends structured symptom arrays to the FastAPI `/predict` endpoint.
2. **FastAPI** processes the input, predicts the disease using a trained RandomForest model, and returns the result.
3. **Logging**: Every request is logged to `production_log.csv` to simulate real production data streaming.
4. **CI/CD Pipeline (GitHub Actions)**: A daily workflow automatically trains the model and runs an Evidently drift report comparing the production log against the training distribution.

##  Tech Stack
- **Machine Learning:** Scikit-learn (RandomForestClassifier, class-weighted)
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **MLOps / Monitoring:** Evidently, GitHub Actions, Render
- **Data Processing:** Pandas, Joblib

##  Model Performance & Limitations
The model achieves a **100% F1-score** on the test set. 
*Crucial Note:* This is a reflection of the dataset's synthetic nature (perfectly structured symptom combinations) rather than real-world clinical performance. In a real-world scenario, patient data would be sparse and noisy, drastically reducing accuracy. **The focus of this project is the MLOps architecture** (API serving, logging, and drift detection) built around the model, not just the model score.

##  Drift Monitoring
![Drift report screenshot](drift_report.png)
In a simulated test run of random traffic, 46 out of 132 features (34.8%) showed statistical drift, which was correctly flagged by Evidently as being below the 50% dataset drift threshold. This demonstrates the system's ability to distinguish between minor variance and significant dataset drift that would trigger a retraining pipeline.

##  How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload