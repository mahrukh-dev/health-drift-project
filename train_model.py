import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv("data/Training.csv")

# 2. Separate features and labels
# Assume last column is prognosis, all others are symptoms
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Save feature column names, needed later for API conversion
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Save model and reference data (for drift detection)
joblib.dump(model, "model.pkl")
X_train.to_csv("reference_data.csv", index=False)
print("Model and reference_data.csv saved.")