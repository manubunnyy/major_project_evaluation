# Doctor Eval: Project Overview

## 📖 Introduction

**Doctor Eval** is a research project that systematically evaluates whether **agentic AI workflows** provide measurable performance improvements over standard baseline inference for medical language model applications.

### What is Agentic AI?

Agentic AI refers to systems that go beyond simple prompt-response patterns by incorporating:

- **Role-based prompting**: Defining the AI's expertise and perspective
- **Reasoning guidance**: Encouraging step-by-step thinking
- **Constraint awareness**: Explicitly emphasizing requirements
- **Output refinement**: Cleaning artifacts and formatting

### Research Motivation

As large language models (LLMs) become increasingly deployed in healthcare settings, understanding which prompting strategies maximize performance is critical for:

- **Patient Safety**: Ensuring accurate medical information
- **Regulatory Compliance**: Meeting healthcare standards
- **Cost Optimization**: Getting better results from smaller, cheaper models
- **Practical Deployment**: Understanding when complexity adds value

## 🎯 Research Question

> **"Do agentic workflows (enhanced prompting + reasoning guidance) outperform simple baseline inference for medical LLM applications across different model sizes and architectures?"**

### Hypothesis

We hypothesize that agentic approaches will:

1. Improve performance on **complex reasoning tasks** (medical diagnosis, differential diagnosis)
2. Enhance **instruction-following accuracy** (constraint adherence, formatting)
3. Maintain or improve **semantic quality** (professional medical communication)

## 🏗️ Architecture

### Baseline System

The baseline represents standard LLM usage:

```python
# Simple prompt → response
def baseline(prompt):
    return llm.invoke(prompt)
```

**Characteristics**:

- Direct question-answer
- No role definition
- No reasoning guidance
- Minimal post-processing

### Agentic System

The agentic system enhances the baseline with structured prompting:

```python
class UnifiedAgent:
    async def process_async(self, prompt: str) -> str:
        # Enhanced system prompt
        system_prompt = """You are an expert medical AI assistant.

{input}

Important guidelines:
- For medical questions: Think step-by-step, then select the BEST answer
- For writing constraints: Follow them EXACTLY (word counts, forbidden letters, format)
- Output ONLY your final answer - no explanations unless asked

Your response:"""

        response = await llm.ainvoke({"input": prompt})

        # Clean artifacts
        cleaned = remove_think_tags(response)
        cleaned = strip_markdown_wrappers(cleaned)

        return cleaned
```

**Enhancements**:

1. **Role Definition**: "expert medical AI assistant"
2. **Reasoning Guidance**: "Think step-by-step"
3. **Constraint Emphasis**: Explicit requirements
4. **Output Cleaning**: Remove `<think>` tags, markdown wrappers

### Key Components

```
doctor_eval/
├── core/
│   ├── agent.py                    # Agentic system implementation
│   └── consultation_analyzer.py    # Medical consultation handler
├── tasks/
│   ├── gpqa.py                     # Medical reasoning questions (10)
│   └── ifeval.py                   # Constraint-following tasks (10)
├── outputs/
│   ├── detailed_results.csv        # Raw benchmark results
│   ├── final_table_visual.png      # Performance comparison table
│   └── f1_improvement_graph.png    # ChatDoctor F1 scores
├── run.py                          # Main evaluation orchestrator
├── generate_table.py               # Visualization generator
└── README.md                       # Documentation
```

## 🔬 Evaluation Methodology

### Model Selection

We evaluate **4 models** spanning different architectures and sizes:

| Model                    | Parameters | Provider               | Architecture |
| ------------------------ | ---------- | ---------------------- | ------------ |
| **Llama-3.1-8B-Instant** | 8B         | Meta (via Groq)        | Decoder-only |
| **GPT-OSS-20B**          | 20B        | OpenAI (via Groq)      | Decoder-only |
| **Qwen-3-32B**           | 32B        | Alibaba (via Groq)     | Decoder-only |
| **Kimi-K2-Instruct**     | ~30B\*     | Moonshot AI (via Groq) | Decoder-only |

\*Estimated based on performance characteristics

### Benchmark Suite

Each model is evaluated on **3 benchmarks** with **10 samples each**:

1. **GPQA-Medical**: Medical knowledge & reasoning (accuracy %)
2. **IFEval**: Instruction-following with constraints (accuracy %)
3. **ChatDoctor**: Patient consultation quality (BERTScore F1)

### Evaluation Protocol

For each model:

```
1. Run 10 GPQA questions (Baseline mode)
2. Run 10 GPQA questions (Agentic mode)
3. Run 10 IFEval tasks (Baseline mode)
4. Run 10 IFEval tasks (Agentic mode)
5. Run 10 ChatDoctor samples (Baseline mode)
6. Run 10 ChatDoctor samples (Agentic mode)
7. Calculate average scores per benchmark
8. Compare Baseline vs Agentic
```

Total: **240 API calls** (4 models × 30 samples × 2 modes)

## 💡 Design Decisions

### Why these benchmarks?

1. **GPQA-Medical**: Tests clinical knowledge and reasoning
   - Reflects real-world diagnostic scenarios
   - Multiple-choice format reduces ambiguity
   - Covers diverse medical domains

2. **IFEval**: Tests precise instruction-following
   - Critical for healthcare compliance
   - Measures constraint adherence (a key agent capability)
   - Objective pass/fail validation

3. **ChatDoctor**: Tests semantic communication quality
   - Evaluates patient-facing language
   - BERTScore captures medical accuracy
   - Professional tone matters in healthcare

### Why 10 samples per benchmark?

**Trade-off analysis**:

- **Statistical significance**: Minimum for trend detection
- **API cost**: Manageable (240 total calls)
- **Runtime**: ~8-10 minutes total
- **Reproducibility**: Fast enough for iteration

For publication, we recommend 50-100 samples for stronger statistical power.

### Why simple agentic approach?

We deliberately chose a **minimalist** strategy to:

1. **Isolate the effect** of basic prompt engineering
2. **Avoid overfitting** to specific tasks
3. **Ensure reproducibility** (no complex hyperparameters)
4. **Maintain clarity** for academic communication

## 🛠️ Technical Implementation

### Dependencies

```python
# LLM Integration
langchain==0.1.0
langchain-groq==0.0.1
langchain-google-genai==0.0.1

# Evaluation
bert-score==0.3.13
datasets==2.14.0

# Data & Visualization
pandas==2.1.0
matplotlib==3.7.0
seaborn==0.12.0
numpy==1.24.0
```

### API Integration

```python
def _init_llm(self, temperature: float):
    if "gemini" in self.mn.lower():
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=self.mn,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature
        )

    # Groq for all other models
    gn = "llama-3.3-70b-versatile" if "gpt-oss" in self.mn else self.mn
    return ChatGroq(
        model_name=gn,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature
    )
```

### Cleaning Pipeline

````python
def _clean_output(self, text: str) -> str:
    # 1. Remove reasoning artifacts
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. Strip markdown wrappers
    if text.startswith("```") and text.endswith("```"):
        text = extract_code_block(text)

    # 3. Trim whitespace
    return text.strip()
````

## 🔄 Reproducibility

### Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/yourusername/doctor_eval
cd doctor_eval

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API keys
export GROQ_API_KEY="your_groq_api_key"
export GEMINI_API_KEY="your_gemini_key"  # Optional

# 5. Run evaluation
cd doctor_eval
python3 run.py
```

### Expected Runtime

- **Per model**: ~2 minutes
- **Total (4 models)**: ~8-10 minutes
- **Output**: CSV file + 2 visualizations

### Configuration

Edit `run.py` to customize:

```python
NUM_SAMPLES = 10  # Samples per benchmark
MODELS = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905"
]
```

## 📊 Output Artifacts

1. **`outputs/detailed_results.csv`**
   - Raw scores for all models × benchmarks × modes
   - Columns: Model, Process, GPQA, IFEval, ChatDoc, Average

2. **`outputs/final_table_visual.png`**
   - Black & white comparison table
   - Bold highlighting for winning scores

3. **`outputs/f1_improvement_graph.png`**
   - Bar chart of ChatDoctor F1 scores
   - Baseline vs Agentic comparison

## 🎓 Academic Context

### Related Work

This project builds on:

1. **Prompt Engineering**: Wei et al. (2022) - Chain-of-Thought reasoning
2. **Medical AI Evaluation**: Singhal et al. (2023) - Med-PaLM benchmarks
3. **Instruction Following**: Zhou et al. (2023) - IFEval framework

### Novel Contributions

1. **Medical-specific agentic evaluation**
   - First systematic comparison for healthcare LLMs
   - Multiple architecture testing

2. **Practical minimalism**
   - Simple, reproducible approach
   - No complex multi-agent systems

3. **Constraint-aware agents**
   - Emphasis on instruction-following in medical context
   - Regulatory compliance implications

## 🔮 Future Directions

### Short Term

1. Increase sample sizes (10 → 50-100)
2. Add statistical significance testing (t-tests, Cohen's d)
3. Model-specific optimization strategies

### Medium Term

1. Implement advanced agentic techniques:
   - Self-consistency (majority voting)
   - Tool augmentation (programmatic validators)
   - Retrieval-augmented generation (RAG)

2. Expand benchmarks:
   - Medical diagnosis cases
   - Treatment planning scenarios
   - Clinical note generation

### Long Term

1. Real-world deployment study
2. Human expert evaluation
3. Regulatory compliance testing

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{doctor_eval_2026,
  title={Doctor Eval: Systematic Evaluation of Agentic Workflows for Medical AI},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/doctor_eval},
  note={Research project evaluating agentic AI in healthcare applications}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 💬 Contact

For questions or collaboration:

- **Email**: your.email@example.com
- **GitHub Issues**: github.com/yourusername/doctor_eval/issues
- **Twitter**: @yourusername

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Research Project
