
import csv
import os

# The ID we want to match from synthetic_lab_reports.csv
TARGET_ID = "3d990803-6fc8-4cd4-81fc-91e59a3abdc0"

def update_csv(file_path, id_col):
    if not os.path.exists(file_path):
        return
    
    rows = []
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row[id_col] == "report":
                row[id_col] = TARGET_ID
            rows.append(row)
            
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {file_path}")

update_csv("baseline_gemini_pdf_outputs.csv", "report_id")
update_csv("prompted_gemini_pdf_outputs.csv", "report_id")
