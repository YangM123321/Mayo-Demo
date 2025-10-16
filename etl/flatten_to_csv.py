import pandas as pd
import pathlib

# Minimal placeholder data
data = [
    {"patient_id": "p1", "timestamp": "2025-09-18T10:00", "code": "HR", "value": 85, "unit": "bpm"},
    {"patient_id": "p2", "timestamp": "2025-09-18T10:05", "code": "BP_SYS", "value": 120, "unit": "mmHg"},
]

df = pd.DataFrame(data)

pathlib.Path("data/interim").mkdir(parents=True, exist_ok=True)
df.to_csv("data/interim/observations.csv", index=False)
print("Created data/interim/observations.csv")
