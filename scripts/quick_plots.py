from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the vitals and triage tables
base_path = Path(r"C:\Users\yangm\Desktop\Mayo-Demo\data\physionet.org\files\mimic-iv-ed-demo\2.2\ed")
vitals = pd.read_csv(base_path / "vitalsign.csv.gz", compression="gzip")
triage = pd.read_csv(base_path / "triage.csv.gz", compression="gzip")

# --- First plot: Heart Rate Distribution ---
plt.figure(figsize=(8,6))
plt.hist(vitals['heartrate'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title("Heart Rate Distribution")
plt.xlabel("Heart Rate (bpm)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("heart_rate_distribution.png")

# --- Second plot: Top 10 Chief Complaints ---
plt.figure(figsize=(8,6))
triage['chiefcomplaint'].value_counts().head(10).plot(kind='barh', color='coral', edgecolor='black')
plt.title("Top 10 Chief Complaints")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig("top10_chief_complaints.png")

# --- Third plot: Heart Rate Boxplot ---
plt.figure(figsize=(8,6))
plt.boxplot(vitals['heartrate'].dropna())
plt.title("Heart Rate Boxplot")
plt.ylabel("Heart Rate (bpm)")
plt.tight_layout()
plt.savefig("heart_rate_boxplot.png")

# ✅ Show all figures at once
plt.show()

print("✅ Saved plots: heart_rate_distribution.png, top10_chief_complaints.png, and heart_rate_boxplot.png")
