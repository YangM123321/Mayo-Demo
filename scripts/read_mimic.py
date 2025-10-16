

from pathlib import Path
import pandas as pd

base_path = Path(r"C:\Users\yangm\Desktop\Mayo-Demo\data\physionet.org\files\mimic-iv-ed-demo\2.2\ed")

vitals = pd.read_csv(base_path / "vitalsign.csv.gz", compression="gzip")
triage = pd.read_csv(base_path / "triage.csv.gz",    compression="gzip")

print(vitals.head())
print(triage.head())

