"""
Evaluate baseline vs prompted outputs using BERTScore-F1
for clinical interpretation generation.

Inputs:
- synthetic_lab_reports.csv
- baseline_gemini_pdf_outputs.csv
- prompted_gemini_pdf_outputs.csv

Output:
- Prints mean BERTScore-F1
- Prints improvement (delta)
"""

import csv
from bert_score import score
import numpy as np

# =========================
# FILE PATHS
# =========================

GROUND_TRUTH_FILE = "synthetic_lab_reports.csv"
BASELINE_FILE = "baseline_gemini_pdf_outputs.csv"
PROMPTED_FILE = "prompted_gemini_pdf_outputs.csv"

# =========================
# LOAD CSVs
# =========================

def load_csv_to_dict(file_path, key_col, value_col):
    data = {}
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row[key_col]] = row[value_col]
    return data

ground_truth = load_csv_to_dict(
    GROUND_TRUTH_FILE,
    key_col="report_id",
    value_col="ground_truth_result"
)

baseline_outputs = load_csv_to_dict(
    BASELINE_FILE,
    key_col="report_id",
    value_col="baseline_output"
)

prompted_outputs = load_csv_to_dict(
    PROMPTED_FILE,
    key_col="report_id",
    value_col="prompted_output"
)

# =========================
# ALIGN DATA
# =========================

common_ids = set(ground_truth.keys()) & \
             set(baseline_outputs.keys()) & \
             set(prompted_outputs.keys())

print(f"Loaded {len(ground_truth)} ground truth entries")
print(f"Loaded {len(baseline_outputs)} baseline outputs")
print(f"Loaded {len(prompted_outputs)} prompted outputs")
print(f"Evaluating {len(common_ids)} aligned samples")

if len(common_ids) == 0:
    print("\nERROR: No matching report_ids found across all three files.")
    print("Check if the report_ids in your output files match those in synthetic_lab_reports.csv.")
    print("Example ground truth ID:", list(ground_truth.keys())[0] if ground_truth else "N/A")
    print("Example baseline ID:", list(baseline_outputs.keys())[0] if baseline_outputs else "N/A")
    import sys
    sys.exit(1)

refs = []
baseline_preds = []
prompted_preds = []

for rid in sorted(common_ids):
    refs.append(ground_truth[rid])
    baseline_preds.append(baseline_outputs[rid])
    prompted_preds.append(prompted_outputs[rid])

# =========================
# COMPUTE BERTSCORE
# =========================

print("\nComputing BERTScore for BASELINE...")
_, _, f1_baseline = score(
    baseline_preds,
    refs,
    lang="en",
    rescale_with_baseline=False
)

print("Computing BERTScore for PROMPTED...")
_, _, f1_prompted = score(
    prompted_preds,
    refs,
    lang="en",
    rescale_with_baseline=False
)

baseline_mean = f1_baseline.mean().item()
prompted_mean = f1_prompted.mean().item()
delta = prompted_mean - baseline_mean

# =========================
# RESULTS
# =========================

print("\n===== RESULTS =====")
print(f"Baseline BERTScore-F1 : {baseline_mean:.4f}")
print(f"Prompted BERTScore-F1 : {prompted_mean:.4f}")
print(f"Improvement (Δ)       : {delta:.4f}")

# =========================
# OPTIONAL: CONFIDENCE INTERVAL
# =========================

diffs = (f1_prompted - f1_baseline).cpu().numpy()
ci_low = np.percentile(diffs, 2.5)
ci_high = np.percentile(diffs, 97.5)

print("\n95% Confidence Interval for Δ:")
print(f"[{ci_low:.4f}, {ci_high:.4f}]")
