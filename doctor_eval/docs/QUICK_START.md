# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Prerequisites

- Python 3.9+
- Groq API key ([get free key](https://console.groq.com))
- 2GB disk space
- Internet connection

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/doctor_eval
cd doctor_eval

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
export GROQ_API_KEY="your_groq_api_key_here"
```

### Run Evaluation

```bash
cd doctor_eval
python3 run.py
```

**Expected Output**:

```
🚀 Starting Research-Grade Benchmark Suite (10 Medical Samples)...

🔬 Evaluating Model: llama-3.1-8b-instant
   - Running GPQA-Diamond...
   - Running IFEval...
   - Running ChatDoctor (BERTScore)...

🔬 Evaluating Model: openai/gpt-oss-20b
   ...

================================================================================
                       Model  Process  GPQA  IFEval  ChatDoc  Average
 llama-3.1-8b-instant (Base) Baseline  90.0    50.0     81.6     73.9
llama-3.1-8b-instant (Agent)  Agentic  90.0    70.0     82.8     80.9
...
================================================================================

✅ Research data collected. Detailed results in outputs/detailed_results.csv
📊 Generating visualizations...
✅ Visuals saved to outputs/
```

**Runtime**: ~8-10 minutes for all 4 models

### View Results

```bash
# Open results
cat outputs/detailed_results.csv

# View visualizations
open outputs/final_table_visual.png
open outputs/f1_improvement_graph.png
```

## 📊 Understanding the Output

### Results CSV

```csv
Model,Process,GPQA,IFEval,ChatDoc,Average
llama-3.1-8b-instant (Base),Baseline,90.0,50.0,81.6,73.9
llama-3.1-8b-instant (Agent),Agentic,90.0,70.0,82.8,80.9
```

**Columns**:

- **Model**: Which LLM was tested
- **Process**: Baseline (simple) or Agentic (enhanced)
- **GPQA**: Medical reasoning accuracy (%)
- **IFEval**: Instruction-following accuracy (%)
- **ChatDoc**: Patient consultation quality (F1 score)
- **Average**: Mean across all benchmarks

### Table Visual

Black & white table with:

- **Bold numbers** = Better score for that model
- Alternating gray rows for readability
- Black header with white text

### F1 Graph

Bar chart showing:

- Gray bars = Baseline ChatDoctor F1
- Green bars = Agentic ChatDoctor F1
- Compare heights to see improvement

## 🛠️ Customization

### Change Sample Size

Edit `run.py`:

```python
NUM_SAMPLES = 20  # Increase from 10 to 20
```

**Trade-off**: More samples = better statistics, longer runtime

### Add/Remove Models

Edit `run.py`:

```python
MODELS = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    # "moonshotai/kimi-k2-instruct-0905"  # Comment out to skip
]
```

### Modify Benchmarks

Edit task files:

- `tasks/gpqa.py` - Medical reasoning questions
- `tasks/ifeval.py` - Instruction-following tasks

Add new questions following the existing format.

## ❓ Troubleshooting

### API Rate Limits

**Error**: `429 Too Many Requests`

**Solution**: Add delay between requests

```python
# In run.py, add:
import time
time.sleep(1)  # Wait 1 second between calls
```

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'bert_score'`

**Solution**: Reinstall requirements

```bash
pip install -r requirements.txt --force-reinstall
```

### API Key Issues

**Error**: `AuthenticationError`

**Solution**: Verify API key

```bash
echo $GROQ_API_KEY  # Should print your key
export GROQ_API_KEY="your_real_key"
```

### Out of Memory

**Error**: `MemoryError` or system slowdown

**Solution**: Reduce samples or run models sequentially

```python
NUM_SAMPLES = 5  # Reduce from 10
# Or remove models from MODELS list
```

## 🔍 Next Steps

### For Research

1. **Read Methodology**: See `docs/METHODOLOGY.md` for detailed explanations
2. **Analyze Results**: Check `docs/RESULTS_ANALYSIS.md` for insights
3. **Understand Architecture**: Read `docs/PROJECT_OVERVIEW.md`

### For Development

1. **Modify Agent**: Edit `core/agent.py` to try different prompting strategies
2. **Add Benchmarks**: Create new task files in `tasks/`
3. **Customize Visuals**: Edit `generate_table.py` for different plots

### For Publication

1. **Increase Samples**: Set `NUM_SAMPLES = 50`
2. **Run Multiple Times**: Check reproducibility
3. **Add Statistical Tests**: Implement t-tests and effect sizes
4. **Human Evaluation**: Have medical experts review outputs

## 📚 Documentation Index

- **Quick Start** (this file): Getting up and running
- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)**: Full project documentation
- **[METHODOLOGY.md](docs/METHODOLOGY.md)**: Benchmark design and evaluation protocol
- **[RESULTS_ANALYSIS.md](docs/RESULTS_ANALYSIS.md)**: Detailed findings with visualizations
- **[README.md](README.md)**: Project summary

## 💬 Getting Help

- **GitHub Issues**: github.com/yourusername/doctor_eval/issues
- **Email**: your.email@example.com
- **Documentation**: Check the `docs/` folder

## ✅ Quick Checklist

Before running:

- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip list | grep langchain`)
- [ ] GROQ_API_KEY environment variable set
- [ ] In `doctor_eval/` directory
- [ ] Internet connection active

After running:

- [ ] Check `outputs/detailed_results.csv` exists
- [ ] View `outputs/final_table_visual.png`
- [ ] View `outputs/f1_improvement_graph.png`
- [ ] Verify expected results (75% models improved)

---

**Happy Evaluating! 🎉**

For detailed documentation, see the `docs/` folder.
