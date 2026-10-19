import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Load reference data (training distribution) and production data (API requests)
reference = pd.read_csv("reference_data.csv")
current = pd.read_csv("production_log.csv")

# Align columns (production log may have missing columns or order differences)
current = current.reindex(columns=reference.columns, fill_value=0)

# Generate Data Drift report
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(current_data=current, reference_data=reference)
snapshot.save_html("drift_report.html")

print("Drift report saved to drift_report.html")