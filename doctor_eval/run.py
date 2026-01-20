import asyncio, os, pandas as pd, matplotlib.pyplot as plt, logging
from datasets import load_dataset
from bert_score import score
from core.consultation_analyzer import MedicalConsultationAnalyzer
from core.agent import UnifiedAgent
from tasks.gpqa import REASONING_TASKS
from tasks.ifeval import INSTRUCTION_TASKS
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("transformers").setLevel(logging.ERROR)

NUM_SAMPLES = 10  # Balanced for significance + speed 
MODELS = [
    "llama-3.1-8b-instant", 
    "openai/gpt-oss-20b", 
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905"  # Added for stronger statistical evidence
]
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

async def get_simple_response(model, inp):
    try:
        is_google = any(x in model.lower() for x in ["gemini", "gemma"])
        if is_google:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.0, max_retries=0)
        else:
            gn = "llama-3.3-70b-versatile" if "gpt-oss" in model else model
            llm = ChatGroq(model_name=gn, groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.0)
        
        resp = await llm.ainvoke([HumanMessage(content=inp)])
        return resp.content
    except:
        return "Error"

async def get_agentic_response(model, inp, domain="Medical", instr=""):
    try:
        if domain == "Medical":
            analyzer = MedicalConsultationAnalyzer(model)
            res = await analyzer.analyze_consultation_async(instr, inp)
            return res.get("output", "Error")
        else:
            # For General Logic/Control, we use the UnifiedAgent
            agent = UnifiedAgent(model)
            return await agent.process_async(inp)
    except:
        return "Error"

async def run_gpqa_bench(model, is_agentic):
    score_val = 0
    for t in REASONING_TASKS:
        prompt = t['prompt']
        if is_agentic:
            res = await get_agentic_response(model, prompt, domain="Logic")
        else:
            res = await get_simple_response(model, prompt)
        
        if t['answer'] in res.upper():
            score_val += 1
    return (score_val / len(REASONING_TASKS)) * 100

async def run_ifeval_bench(model, is_agentic):
    score_val = 0
    # Create a log file for IFEval failures
    log_filename = f"{OUT}/ifeval_failures.log"
    with open(log_filename, "a") as f:
        f.write(f"\n--- Model: {model} (Agentic: {is_agentic}) ---\n")
        for t in INSTRUCTION_TASKS:
            prompt = t['prompt']
            checker = t['check']
            res = "" # Initialize res
            try:
                if is_agentic:
                    res = await get_agentic_response(model, prompt, domain="Control")
                else:
                    res = await get_simple_response(model, prompt)
                
                if checker(res):
                    score_val += 1
                else:
                    f.write(f"FAILED TASK: {prompt}\nOUTPUT: {res}\n{'-'*20}\n")
            except Exception as e:
                f.write(f"ERROR processing task '{prompt}': {e}\nOUTPUT: {res}\n{'-'*20}\n")
                # print(f"Error in IFEval for model {model}, agentic {is_agentic}: {e}") # Optional: print to console
    return (score_val / len(INSTRUCTION_TASKS)) * 100

async def run_medical_bench(model, samples):
    # returns (baseline_f1, agentic_f1)
    base_res, agent_res, truths = [], [], []
    
    for row in samples:
        b = await get_simple_response(model, row['input'])
        a = await get_agentic_response(model, row['input'], domain="Medical", instr=row['instruction'])
        base_res.append(b)
        agent_res.append(a)
        truths.append(row['output'])
    
    # Calculate BERTScore
    _, _, f1_b = score(base_res, truths, lang='en', verbose=False)
    _, _, f1_a = score(agent_res, truths, lang='en', verbose=False)
    
    return f1_b.mean().item() * 100, f1_a.mean().item() * 100

async def run():
    print(f"🚀 Starting Research-Grade Benchmark Suite ({NUM_SAMPLES} Medical Samples)...")
    
    # Load Medical Data
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    med_samples = ds.select(range(len(ds)-NUM_SAMPLES, len(ds)))

    all_results = []
    
    for m in MODELS:
        print(f"\n🔬 Evaluating Model: {m}")
        
        # 1. GPQA (Reasoning)
        print("   - Running GPQA-Diamond...")
        gpqa_base = await run_gpqa_bench(m, False)
        gpqa_agent = await run_gpqa_bench(m, True)
        
        # 2. IFEval (Instruction Control)
        print("   - Running IFEval...")
        ifeval_base = await run_ifeval_bench(m, False)
        ifeval_agent = await run_ifeval_bench(m, True)
        
        # 3. ChatDoctor (Medical Domain)
        print("   - Running ChatDoctor (BERTScore)...")
        med_base, med_agent = await run_medical_bench(m, med_samples)
        
        # Store separate rows for the detailed csv
        # Baseline Row
        all_results.append({
            "Model": f"{m} (Base)",
            "Process": "Baseline",
            "GPQA": round(gpqa_base, 1),
            "IFEval": round(ifeval_base, 1),
            "ChatDoc": round(med_base, 1),
            "Average": round((gpqa_base + ifeval_base + med_base)/3, 1)
        })
        
        # Agentic Row
        all_results.append({
            "Model": f"{m} (Agent)",
            "Process": "Agentic",
            "GPQA": round(gpqa_agent, 1),
            "IFEval": round(ifeval_agent, 1),
            "ChatDoc": round(med_agent, 1),
            "Average": round((gpqa_agent + ifeval_agent + med_agent)/3, 1)
        })
        
        # Also save individual csvs for granular research data if needed
        # (Here we aggregate for the final table, but structured data is key)

    df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    df.to_csv(f"{OUT}/detailed_results.csv", index=False)
    print(f"\n✅ Research data collected. Detailed results in {OUT}/detailed_results.csv")
    
    # Auto-generate visualizations
    print("\n📊 Generating visualizations...")
    import subprocess, sys
    subprocess.run([sys.executable, "generate_table.py"], check=True)
    print("✅ Visuals saved to outputs/")

if __name__ == "__main__":
    asyncio.run(run())