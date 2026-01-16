import asyncio
import os
import csv
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from core.health_analyzer import HealthReportAnalyzer
from core.utils import extract_text_from_pdf
from bert_score import score

# Load environment variables
load_dotenv()

# Configuration
REPORTS_DIR = "data/reports"
GROUND_TRUTH_FILE = "data/ground_truth.csv"
OUTPUT_DIR = "outputs"
FINAL_CSV = "outputs/evaluation_results.csv"

# Models to evaluate
MODELS = [
    "gemini-2.5-flash",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b"
]

def map_model_name(model_name: str):
    """Correct model names for API calls if needed."""
    # Based on listing, these are correct as is on Groq
    return model_name

async def get_baseline_summary(model: str, text: str) -> str:
    """Run baseline: No special prompt, no agents."""
    try:
        if "gemini" in model:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
        else:
            llm = ChatGroq(model_name=map_model_name(model), groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0)
        
        # Truncate text to stay within request limits for strict models (like Qwen)
        safe_text = text[:4000] if len(text) > 4000 else text
        
        # Minimal prompt to ensure response exists
        prompt = f"Summarize the health findings: \n\n{safe_text}"
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Baseline Error: {str(e)}"

async def get_agentic_findings(model: str, text: str) -> str:
    """Run agentic: Full prompt + agent architecture."""
    try:
        analyzer = HealthReportAnalyzer(model_name=map_model_name(model))
        results = await analyzer.analyze_report_async(text)
        # Taking "findings only" - mapping 'summary_agent' as the findings summary
        return results.get("summary", "No findings found.")
    except Exception as e:
        return f"Agentic Error: {str(e)}"

async def evaluate():
    print("\n🚀 Starting Unified Health Report Evaluation...")
    
    # Ensure folders
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Ground Truth
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"❌ Error: {GROUND_TRUTH_FILE} not found.")
        return
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    gt_map = dict(zip(gt_df['report_id'], gt_df['ground_truth_result']))
    
    # Get PDF files
    reports = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.pdf')]
    if not reports:
        print(f"❌ No PDF reports found in {REPORTS_DIR}")
        return

    results_data = []

    for report_file in reports:
        report_id = os.path.splitext(report_file)[0]
        if report_id not in gt_map:
            print(f"⏩ Skipping {report_file}: No ground truth.")
            continue
            
        print(f"\n📄 Processing: {report_id}")
        text = extract_text_from_pdf(os.path.join(REPORTS_DIR, report_file))
        
        for model in MODELS:
            print(f"  🤖 Model: {model}")
            
            # 1. Baseline
            print("    - Running Baseline...")
            b_findings = await get_baseline_summary(model, text)
            
            # 2. Agentic
            print("    - Running Agentic...")
            # Truncate input to avoid 413 errors for sensitive models
            safe_text = text[:4000] if len(text) > 4000 else text
            a_findings = await get_agentic_findings(model, safe_text)
            
            results_data.append({
                "report_id": report_id,
                "model": model,
                "baseline_response": b_findings,
                "agentic_response": a_findings,
                "ground_truth": gt_map[report_id]
            })

    if not results_data:
        print("❌ No data processed.")
        return

    # 3. BERTScore Calculation
    print("\n📊 Calculating BERTScore Metrics...")
    
    df = pd.DataFrame(results_data)
    
    # Filter out errors for scoring
    valid_mask = (~df['baseline_response'].str.contains("Error")) & (~df['agentic_response'].str.contains("Error"))
    scored_df = df[valid_mask].copy()
    
    if scored_df.empty or len(scored_df) < len(df):
        print("\n⚠️ Note: Some models were excluded due to errors.")
        excluded = df[~valid_mask]
        for _, row in excluded.iterrows():
            print(f"  ❌ Model {row['model']} failed:")
            print(f"     Baseline: {row['baseline_response'][:100]}...")
            print(f"     Agentic:  {row['agentic_response'][:100]}...")
    
    if scored_df.empty:
        print("❌ No valid responses to score.")
        return

    # Baseline scores
    P_b, R_b, F1_b = score(scored_df['baseline_response'].tolist(), scored_df['ground_truth'].tolist(), lang='en', rescale_with_baseline=False)
    # Agentic scores
    P_a, R_a, F1_a = score(scored_df['agentic_response'].tolist(), scored_df['ground_truth'].tolist(), lang='en', rescale_with_baseline=False)
    
    scored_df['baseline_f1'] = F1_b.tolist()
    scored_df['agentic_f1'] = F1_a.tolist()
    scored_df['change'] = scored_df['agentic_f1'] - scored_df['baseline_f1']

    # 4. Final Output to Terminal and File
    summary_txt = "outputs/performance_summary.txt"
    with open(summary_txt, "w") as f:
        header = f"{'MODEL':<40} | {'BASE':<8} | {'AGENT':<8} | {'CHANGE':<8}"
        separator = "-" * 80
        
        print("\n" + "="*80)
        print(header)
        print(separator)
        
        f.write("="*80 + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        
        for _, row in scored_df.iterrows():
            color = "\033[92m" if row['change'] > 0 else "\033[91m"
            reset = "\033[0m"
            row_str = f"{row['model']:<40} | {row['baseline_f1']:.4f} | {row['agentic_f1']:.4f} | {row['change']:+.4f}"
            
            # Print to terminal with color
            print(f"{row['model']:<40} | {row['baseline_f1']:.4f} | {row['agentic_f1']:.4f} | {color}{row['change']:+.4f}{reset}")
            
            # Write to file without color codes
            f.write(row_str + "\n")
        
        print("=" * 80)
        f.write("=" * 80 + "\n")
    
    # Save to CSV
    scored_df.to_csv(FINAL_CSV, index=False)
    print(f"\n✅ Results saved to {FINAL_CSV}")
    print(f"✅ Summary saved to {summary_txt}\n")

if __name__ == "__main__":
    asyncio.run(evaluate())
