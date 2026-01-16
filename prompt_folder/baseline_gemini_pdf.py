"""
Baseline inference using Gemini API with PDF input.
NO PROMPT. NO AGENTS. RAW MODEL BEHAVIOR ONLY.

Input:
- PDFs placed inside ./pdf_reports/

Output:
- baseline_gemini_pdf_outputs.csv
"""

import os
import csv
import time
import uuid
import google.generativeai as genai
from PyPDF2 import PdfReader

# =========================
# CONFIG
# =========================

GEMINI_API_KEY = "AIzaSyAXfH5JuX4TlxhvNcnp25b_PWTl_YE69Wo"
MODEL_NAMES = [
    "models/gemini-2.0-flash",
    "models/gemini-flash-latest",
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash-exp",
    "models/gemini-pro-latest",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro"
]


PDF_FOLDER = "pdf_reports"
OUTPUT_CSV = "baseline_gemini_pdf_outputs.csv"

TEMPERATURE = 0
MAX_TOKENS = 2048
REQUEST_DELAY = 1.2  # seconds

# =========================
# INIT GEMINI
# =========================

genai.configure(api_key=GEMINI_API_KEY)
# We will initialize the model dynamically in the fallback loop

# =========================
# PDF TEXT EXTRACTION
# =========================

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text).strip()

# =========================
# BASELINE GEMINI CALL
# =========================

def gemini_baseline(text: str) -> str:
    """
    Baseline inference:
    - ONLY extracted PDF text
    - NO INSTRUCTIONS
    - Tries multiple models as fallback
    """
    errors = []
    for model_name in MODEL_NAMES:
        try:
            print(f"  Trying model: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                text,
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_TOKENS
                }
            )
            
            # Check if text is available (handles block/empty responses)
            if response.candidates and response.candidates[0].content.parts:
                return response.text.strip()
            else:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "unknown"
                raise Exception(f"Empty response candidate or blocked. Finish reason: {finish_reason}")
                
        except Exception as e:
            error_msg = str(e)
            print(f"    --> Failed: {error_msg[:100]}...")
            errors.append(f"{model_name}: {error_msg}")
    
    return f"ERROR: All models failed. Details: {'; '.join(errors)}"

# =========================
# MAIN PIPELINE
# =========================

def run_baseline_pdf():
    results = []

    pdf_files = [
        f for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    print(f"Found {len(pdf_files)} PDF files")

    for idx, pdf_name in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(PDF_FOLDER, pdf_name)

        extracted_text = extract_text_from_pdf(pdf_path)

        if not extracted_text:
            output = "ERROR: No text extracted"
        else:
            output = gemini_baseline(extracted_text)

        # Use filename without extension as report_id to allow matching with ground truth
        report_id = os.path.splitext(pdf_name)[0]

        results.append({
            "report_id": report_id,
            "pdf_file": pdf_name,
            "baseline_output": output
        })

        print(f"[{idx}/{len(pdf_files)}] Processed {pdf_name}")
        time.sleep(REQUEST_DELAY)

    # Save results
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["report_id", "pdf_file", "baseline_output"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBaseline PDF outputs saved to: {OUTPUT_CSV}")

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_baseline_pdf()

