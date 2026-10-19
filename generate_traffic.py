import requests
import random

symptom_pool = ["fever", "cough", "fatigue", "headache", "nausea",
                "vomiting", "skin_rash", "itching", "chills", "joint_pain"]

for i in range(30):
    symptoms = random.sample(symptom_pool, k=random.randint(2, 5))
    r = requests.post("http://127.0.0.1:8000/predict", json={"symptoms": symptoms})
    print(i, r.status_code, r.json().get("disease"))