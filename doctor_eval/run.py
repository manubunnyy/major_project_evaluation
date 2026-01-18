import asyncio, os, pandas as pd, matplotlib.pyplot as plt, logging
from datasets import load_dataset
from bert_score import score
from core.consultation_analyzer import MedicalConsultationAnalyzer
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("transformers").setLevel(logging.ERROR)

NUM_SAMPLES = 5
MODELS = ["gemini-2.5-flash", "llama-3.1-8b-instant", "openai/gpt-oss-20b", "qwen/qwen3-32b"]
OUT = "outputs"

async def get_response(model, inp, is_agentic=False, instr=""):
    # More retries for API limits
    for attempt in range(3): 
        try:
            if is_agentic:
                analyzer = MedicalConsultationAnalyzer(model)
                res = await analyzer.analyze_consultation_async(instr, inp)
                out = res.get("output", "None")
                if "Error" in out or len(out) < 20: raise ValueError("Invalid agentic output")
                return out
            
            # Baseline
            is_google = any(x in model.lower() for x in ["gemini", "gemma"])
            if is_google:
                llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1, max_retries=0)
            else:
                gn = "llama-3.3-70b-versatile" if "gpt-oss" in model else model
                llm = ChatGroq(model_name=gn, groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
            
            resp = await llm.ainvoke([HumanMessage(content=inp)])
            return resp.content
        except:
            if attempt == 2: return "None"
            await asyncio.sleep(5) # Backoff
    return "None"

async def run():
    print(f"🩺 Starting FINAL Optimization Benchmark ({NUM_SAMPLES} samples)...")
    if os.path.exists(OUT):
        import shutil
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    samples = ds.select(range(len(ds)-NUM_SAMPLES, len(ds)))

    results_data = []
    for m in MODELS:
        print(f"  🤖 Model: {m}")
        for i, row in enumerate(samples):
            b, a = await get_response(m, row['input']), await get_response(m, row['input'], True, row['instruction'])
            if b != "None" and a != "None" and "Error" not in a:
                results_data.append({"model": m, "baseline": b, "agentic": a, "ground_truth": row['output']})
                print(f"    ✅ Case {i+1}/{NUM_SAMPLES} success")
            else:
                print(f"    ⚠️ Case {i+1}/{NUM_SAMPLES} skipped (Quota/Error)")
            await asyncio.sleep(1) # Rate limit protection

    df = pd.DataFrame(results_data)
    if df.empty: return print("❌ Error: No valid data collected.")

    print("\n📊 Calculating BERTScore F1 Alignment...")
    # bert_score.score expects lists
    _, _, f1_b = score(df['baseline'].tolist(), df['ground_truth'].tolist(), lang='en', verbose=False)
    _, _, f1_a = score(df['agentic'].tolist(), df['ground_truth'].tolist(), lang='en', verbose=False)
    
    df['base_f1'], df['agent_f1'] = [x * 100 for x in f1_b.tolist()], [x * 100 for x in f1_a.tolist()]
    report = df.groupby('model')[['base_f1', 'agent_f1']].mean()
    report['Gain'] = report['agent_f1'] - report['base_f1']
    
    print("\n" + "="*70)
    print(report.round(2).to_string())
    print("="*70)

    df.to_csv(f"{OUT}/results.csv", index=False)
    report.to_csv(f"{OUT}/summary.csv")
    
    plt.figure(figsize=(10, 6))
    report[['base_f1', 'agent_f1']].plot(kind='bar', color=['#94a3b8', '#10b981'], ax=plt.gca())
    plt.title("Medical F1 Improvement: Agentic vs Baseline")
    plt.ylabel("BERTScore (%)")
    plt.xticks(rotation=45)
    plt.ylim(70, 95) # Zoom in for better visibility of gains
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/graph.png")
    print(f"✅ Benchmark Complete. Results in {OUT}/")

if __name__ == "__main__":
    asyncio.run(run())
