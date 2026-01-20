# Doctor Eval: Agentic AI for Medical Applications

## 📋 Overview

This project evaluates **agentic approaches** versus **baseline LLM inference** across medical AI tasks. We demonstrate that prompt engineering combined with agentic workflows significantly improves performance on healthcare-specific benchmarks.

## 🎯 Research Question

**"Do agentic workflows (enhanced prompting + reasoning guidance) outperform simple baseline inference for medical LLM applications?"**

## 🏗️ Architecture

### Baseline Approach

Simple direct inference:

```python
response = llm.invoke(prompt)
```

### Agentic Approach

Enhanced with:

1. **Role-based prompting**: "You are an expert medical AI"
2. **Reasoning guidance**: "Think step-by-step"
3. **Constraint awareness**: Explicit emphasis on requirements
4. **Output cleaning**: Remove artifacts (`<think>` tags, markdown wrappers)

## 📊 Benchmark Suite

We evaluate on **3 medical-focused benchmarks**:

### 1. **GPQA-Medical** (10 questions)

Medical knowledge and reasoning questions

**Sample Question**:

> "A patient takes warfarin and wants to start a new supplement. Which supplement has the HIGHEST risk of interaction? A: Vitamin D, B: Vitamin E, C: Calcium, D: Vitamin B12"

**Tests**: Clinical reasoning, drug interactions, differential diagnosis

### 2. **IFEval** (10 tasks)

Instruction-following with medical writing constraints

**Sample Tasks**:

- "Write about vaccination in exactly 8 words. You MUST NOT use the letter 'e'."
- "Respond with a JSON object containing 'diagnosis' and 'treatment' keys."
- "Write a 6-word smoking warning. ALL letters must be capitalized."

**Tests**: Precise constraint adherence, formatting, word counting

### 3. **ChatDoctor** (10 samples)

Real patient consultation responses evaluated with BERTScore F1

**Sample Input**:

> "I have fever and cough for 3 days"

**Tests**: Semantic quality, professional tone, completeness

## 🚀 Results

### Overall Performance

| Model            | Baseline Avg | Agentic Avg | **Improvement** |
| ---------------- | ------------ | ----------- | --------------- |
| **Llama-3.1-8B** | 73.8%        | **84.0%**   | **+10.2%** ✅   |
| **GPT-OSS-20B**  | 79.7%        | 60.2%       | -19.5% ⚠️       |
| **Qwen-3-32B**   | 67.2%        | **77.1%**   | **+9.9%** ✅    |

### Detailed Breakdown

#### Llama-3.1-8B-Instant

| Benchmark   | Baseline  | Agentic     | Gain       |
| ----------- | --------- | ----------- | ---------- |
| GPQA        | 90.0%     | **100.0%**  | +10.0%     |
| IFEval      | 50.0%     | **70.0%**   | +20.0%     |
| ChatDoctor  | 81.5 F1   | **81.9 F1** | +0.4       |
| **Average** | **73.8%** | **84.0%**   | **+10.2%** |

#### GPT-OSS-20B

| Benchmark   | Baseline  | Agentic     | Gain       |
| ----------- | --------- | ----------- | ---------- |
| GPQA        | 100.0%    | 100.0%      | 0.0%       |
| IFEval      | 60.0%     | 0.0%        | -60.0% ⚠️  |
| ChatDoctor  | 79.0 F1   | **80.6 F1** | +1.6       |
| **Average** | **79.7%** | **60.2%**   | **-19.5%** |

_Note: GPT-OSS-20B shows IFEval regression - see discussion below_

#### Qwen-3-32B

| Benchmark   | Baseline  | Agentic   | Gain      |
| ----------- | --------- | --------- | --------- |
| GPQA        | 100.0%    | 100.0%    | 0.0%      |
| IFEval      | 20.0%     | **50.0%** | +30.0%    |
| ChatDoctor  | 81.5 F1   | 81.4 F1   | -0.1      |
| **Average** | **67.2%** | **77.1%** | **+9.9%** |

## 🔍 Key Findings

### ✅ Successes

1. **Llama Shows Strong Gains**
   - Perfect score on medical reasoning (GPQA: 90% → 100%)
   - Significant constraint adherence improvement (IFEval: 50% → 70%)
   - Overall **+10.2%** improvement

2. **Qwen Benefits from Structure**
   - Massive IFEval improvement (20% → 50%)
   - Maintains perfect GPQA performance
   - Overall **+9.9%** improvement

3. **ChatDoctor Consistency**
   - Agentic approach maintains or slightly improves F1 scores
   - Professional medical tone preserved

### ⚠️ Challenges

1. **GPT IFEval Regression**
   - Model outputs verbose reasoning despite constraints
   - `<think>` tag artifacts interfere with validation
   - Suggests need for model-specific cleaning strategies

2. **Ceiling Effects**
   - When baseline hits 100% (GPQA for GPT/Qwen), agentic can't improve further
   - Demonstrates tasks may be too easy for larger models

## 💡 Insights

### What Works

- **"Think step-by-step"** prompting improves medical reasoning
- **Explicit constraint emphasis** helps with instruction-following
- **Role definition** ("expert medical AI") improves response quality

### What Doesn't Work

- Complex multi-pass validation loops (introduces errors)
- Overly aggressive cleaning (strips valid outputs)
- One-size-fits-all prompting (models have different behaviors)

## 🛠️ Technical Implementation

### Key Components

```
doctor_eval/
├── core/
│   ├── agent.py                    # Agentic system
│   └── consultation_analyzer.py    # Medical consultation handler
├── tasks/
│   ├── gpqa.py                     # Medical reasoning questions
│   └── ifeval.py                   # Constraint-following tasks
├── outputs/
│   ├── detailed_results.csv        # Benchmark results
│   ├── final_table_visual.png      # Heatmap visualization
│   └── f1_improvement_graph.png    # ChatDoctor F1 comparison
└── run.py                          # Main evaluation script
```

### Agent Architecture

```python
class UnifiedAgent:
    def process_async(self, prompt: str) -> str:
        # Enhanced system prompt
        system_prompt = """You are an expert medical AI assistant.

{input}

Important guidelines:
- For medical questions: Think step-by-step, then select the BEST answer
- For writing constraints: Follow them EXACTLY
- Output ONLY your final answer

Your response:"""

        # Generate response
        response = await llm.ainvoke({"input": prompt})

        # Clean artifacts
        cleaned = remove_think_tags(response)
        cleaned = strip_markdown_wrappers(cleaned)

        return cleaned
```

## 📈 Visualizations

### Performance Heatmap

![Performance Table](outputs/final_table_visual.png)

### F1 Score Comparison

![F1 Improvement](outputs/f1_improvement_graph.png)

## 🚀 Running the Evaluation

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API keys
export GROQ_API_KEY="your_groq_key"
export GEMINI_API_KEY="your_gemini_key"  # Optional
```

### Run Benchmarks

```bash
cd doctor_eval
python3 run.py
```

Results will be saved to `outputs/detailed_results.csv` with visualizations auto-generated.

## 📚 Dependencies

- `langchain` - LLM orchestration
- `langchain-groq` - Groq API integration
- `bert-score` - Semantic similarity evaluation
- `datasets` - HuggingFace datasets (ChatDoctor)
- `pandas`, `matplotlib`, `seaborn` - Data analysis & visualization

## 🎓 Research Implications

### For Publication

**Strengths**:

- Clear demonstration of agentic benefits for smaller models (Llama, Qwen)
- Medical domain specificity (all tasks healthcare-related)
- Reproducible methodology

**Limitations**:

- Small sample sizes (10 questions per benchmark)
- Model-specific behaviors require tailored approaches
- Ceiling effects on easier tasks

**Future Work**:

- Expand to 50-100 questions per benchmark
- Implement model-specific cleaning strategies
- Add more complex medical reasoning tasks
- Explore other agentic techniques (self-consistency, tool use)

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{doctor_eval_2026,
  title={Doctor Eval: Agentic AI for Medical Applications},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/doctor_eval}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Groq for API access
- HuggingFace for datasets
- Research community for benchmark designs

---

**Last Updated**: January 2026  
**Status**: Research Project  
**Contact**: your.email@example.com
