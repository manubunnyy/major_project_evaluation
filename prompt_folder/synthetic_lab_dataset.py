import random
import csv
import uuid

def generate_cbc_viral():
    return (
        """Complete Blood Count:
Hemoglobin: {:.1f} g/dL (Normal)
Total WBC Count: {} /mm3 (High)
Neutrophils: {}% (Normal)
Lymphocytes: {}% (High)
Platelet Count: {} /mm3 (Normal)
""".format(
            random.uniform(13.0, 15.0),
            random.randint(11000, 15000),
            random.randint(40, 55),
            random.randint(45, 55),
            random.randint(180000, 260000)
        ),
        "Findings suggest viral fever with lymphocytosis.",
        "cbc"
    )

def generate_lipid():
    return (
        """Lipid Profile:
Total Cholesterol: {} mg/dL (High)
LDL Cholesterol: {} mg/dL (High)
HDL Cholesterol: {} mg/dL (Low)
Triglycerides: {} mg/dL (High)
""".format(
            random.randint(240, 300),
            random.randint(160, 220),
            random.randint(25, 39),
            random.randint(180, 260)
        ),
        "Results indicate hyperlipidemia with elevated cardiovascular risk.",
        "lipid"
    )

def generate_diabetes():
    return (
        """Blood Glucose Analysis:
Fasting Plasma Glucose: {} mg/dL (High)
HbA1c: {:.1f}% (High)
Random Blood Sugar: {} mg/dL (High)
""".format(
            random.randint(140, 200),
            random.uniform(7.0, 10.0),
            random.randint(200, 320)
        ),
        "Results are consistent with uncontrolled diabetes mellitus.",
        "glucose"
    )

def generate_ckd():
    return (
        """Renal Function Test:
Serum Creatinine: {:.1f} mg/dL (High)
Blood Urea Nitrogen: {} mg/dL (High)
eGFR: {} mL/min/1.73m2 (Low)
""".format(
            random.uniform(1.8, 3.2),
            random.randint(40, 70),
            random.randint(15, 45)
        ),
        "Findings suggest chronic kidney disease with reduced renal function.",
        "renal"
    )

def generate_hypothyroid():
    return (
        """Thyroid Function Test:
TSH: {:.1f} μIU/mL (High)
Free T4: {:.1f} ng/dL (Low)
Free T3: {:.1f} pg/mL (Low)
""".format(
            random.uniform(8.0, 18.0),
            random.uniform(0.4, 0.8),
            random.uniform(1.8, 2.3)
        ),
        "Findings are consistent with primary hypothyroidism.",
        "thyroid"
    )

GENERATORS = [
    generate_cbc_viral,
    generate_lipid,
    generate_diabetes,
    generate_ckd,
    generate_hypothyroid
]

def generate_dataset(n_samples=500, output_file="synthetic_lab_reports.csv"):
    rows = []
    for _ in range(n_samples):
        gen = random.choice(GENERATORS)
        report, diagnosis, category = gen()
        rows.append({
            "report_id": str(uuid.uuid4()),
            "dataset_report": report.strip(),
            "ground_truth_result": diagnosis,
            "category": category
        })

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["report_id", "dataset_report", "ground_truth_result", "category"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {n_samples} samples → {output_file}")

if __name__ == "__main__":
    generate_dataset(n_samples=1000)
