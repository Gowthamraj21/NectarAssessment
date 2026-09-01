"""Run the complete reproducible analysis pipeline from the repository root."""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
# steps=["generate_data.py","task1_eda.py","task2_predictive_maintenance.py","task3_forecasting.py","task4_anomaly_detection.py","task5_connectivity_analysis.py","build_report.py","build_notebook.py"]
steps=["generate_data.py","task1_eda.py","task2_predictive_maintenance.py","task3_forecasting.py","task4_anomaly_detection.py","task5_connectivity_analysis.py","build_notebook.py"]

for step in steps:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable,str(ROOT/"src"/step)],cwd=ROOT,check=True)
print("\nPipeline completed successfully.")
