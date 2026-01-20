# Methodology: Benchmarks & Evaluation Protocol

## 📖 Overview

This document provides a comprehensive explanation of our evaluation methodology, including benchmark selection rationale, task design, scoring protocols, and implementation details.

## 🎯 Benchmark Selection

We evaluated models on **three complementary benchmarks** that together assess critical capabilities for medical AI applications:

### Design Principles

1. **Medical Relevance**: All tasks relate to healthcare applications
2. **Objective Evaluation**: Clear, reproducible scoring criteria
3. **Capability Diversity**: Tests different skills (reasoning, following, communication)
4. **Practical Applicability**: Reflects real deployment scenarios

---

## 🔬 Benchmark 1: GPQA-Medical

### What It Tests

**Medical Knowledge & Reasoning**: Ability to answer complex medical multiple-choice questions requiring:

- Clinical knowledge
- Differential diagnosis
- Treatment selection
- Medical ethics
- Pharmacology

### Why This Matters

- **Patient Safety**: Incorrect medical advice can harm patients
- **Diagnostic Accuracy**: Core function of medical AI
- **Knowledge Depth**: Tests understanding beyond memorization

### Task Design

**Format**: Multiple-choice questions (A/B/C/D)  
**Count**: 10 questions  
**Topics**: Cardiology, pharmacology, infectious disease, pediatrics, ethics

#### Sample Questions

**Question 1: Pharmacology**

```
A patient takes warfarin and wants to start a new supplement.
Which has the HIGHEST risk of interaction?

A: Vitamin D
B: Vitamin E
C: Calcium
D: Vitamin B12

Answer: B (Vitamin E increases bleeding risk)
```

**Question 2: Emergency Medicine**

```
A 2-year-old has high fever and refuses to move their neck.
What is the MOST urgent concern?

A: Viral meningitis
B: Bacterial meningitis
C: Torticollis
D: Ear infection

Answer: B (Bacterial meningitis is life-threatening)
```

**Question 3: Differential Diagnosis**

```
A 60-year-old presents with fatigue, weight loss, and night sweats.
Which diagnosis should be ruled out FIRST?

A: Hypothyroidism
B: Depression
C: Tuberculosis or malignancy
D: Anemia

Answer: C (Red flag symptoms requiring urgent evaluation)
```

### Scoring Method

```python
def score_gpqa(response, correct_answer):
    # Extract answer from response
    predicted = extract_answer_letter(response)

    # Simple match
    if predicted == correct_answer:
        return 1.0
    else:
        return 0.0

# Overall score = (correct answers / total questions) * 100
```

**Accuracy = (Correct / 10) × 100**

### Why We Chose These Questions

1. **Ambiguity**: Multiple plausible options test reasoning
2. **Clinical Relevance**: Common diagnostic scenarios
3. **Critical Thinking**: Require prioritization (FIRST, MOST, HIGHEST)
4. **Safety**: Wrong answers have real consequences

---

## 📝 Benchmark 2: IFEval

### What It Tests

**Instruction-Following & Constraint Adherence**: Ability to follow explicit constraints:

- Word count (exactly N words)
- Forbidden letters (no letter 'e')
- Format requirements (JSON, bullet points, capitalization)
- Multi-constraint tasks (word count + no letter 'e')

### Why This Matters

- **Regulatory Compliance**: Healthcare documentation has strict requirements
- **Clinical Communication**: Formats matter (prescriptions, notes, alerts)
- **Safety**: Incorrect formats can cause errors

### Task Design

**Format**: Constrained generation tasks  
**Count**: 10 tasks  
**Categories**:

- Negative constraints (3): Forbidden letters
- Word count (4): Exact word requirements
- Format (3): JSON, capitalization, bullet points

#### Sample Tasks

**Task 1: Negative Constraint**

```
Write about vaccination in exactly 8 words.
You MUST NOT use the letter 'e'.

Constraint Check:
- Count words: len(text.split()) == 8
- No letter 'e': 'e' not in text.lower()

Example Valid: "Vaccination improves immunity against viral and bacterial illusions"
Example Invalid: "Vaccines prevent diseases effectively" (has 'e')
```

**Task 2: Multi-Constraint**

```
Describe antibiotics without using the letters 's' or 'a'.

Constraint Check:
- No 's': 's' not in text.lower()
- No 'a': 'a' not in text.lower()
- Min length: len(text.split()) >= 5

Example Valid: "Drugs fight germs by killing them"
Example Invalid: "Antibiotics kill bacteria" (has 'a' and 's')
```

**Task 3: Format Constraint**

```
Respond with a JSON object containing 'diagnosis' and 'treatment' keys.
Do NOT include any other text.

Constraint Check:
- Valid JSON: json.loads(text)
- Has required keys: 'diagnosis' in data and 'treatment' in data
- No extra text: text.strip().startswith('{') and text.strip().endswith('}')

Example Valid: {"diagnosis": "Hypertension", "treatment": "Lisinopril"}
Example Invalid: Here is the JSON: {"diagnosis": "Hypertension"}
```

### Scoring Method

```python
def score_ifeval(response, task):
    # Apply constraint checker
    passes_constraint = task['check'](response)

    if passes_constraint:
        return 1.0
    else:
        return 0.0

# Overall score = (passed tasks / total tasks) * 100
```

**Accuracy = (Passed / 10) × 100**

### Why We Chose These Constraints

1. **Healthcare Relevance**: Medical forms have strict format requirements
2. **Objective Validation**: Programmatic checks (no ambiguity)
3. **Difficulty Gradient**: From simple (word count) to complex (multi-constraint)
4. **Agent Capability**: Tests where explicit guidance helps

---

## 💬 Benchmark 3: ChatDoctor

### What It Tests

**Patient Consultation Quality**: Semantic accuracy and professionalism of responses to patient queries using **BERTScore F1**.

### Why This Matters

- **Patient Communication**: Front-end of medical AI
- **Professional Tone**: Trust and clarity
- **Semantic Accuracy**: Medical correctness beyond exact wording

### Task Design

**Format**: Query → Professional response  
**Count**: 10 patient consultations  
**Source**: ChatDoctor-HealthCareMagic-100k dataset  
**Evaluation**: BERTScore F1 (semantic similarity)

#### Sample Consultations

**Consultation 1**

```
Input: "I have fever and cough for 3 days"

Expected Response:
"Hello, Thank you for posting your query. Based on your symptoms
of fever and cough lasting 3 days, you may have an upper respiratory
infection. I recommend rest, hydration, and over-the-counter fever
reducers. If symptoms worsen or persist beyond 5 days, please
consult a physician. Hope this helps. Thank you."

Evaluation: BERTScore compares semantic similarity with ground truth
```

**Consultation 2**

```
Input: "My child has stomach pain and diarrhea"

Expected Response:
"Hello, Thank you for posting your query. Stomach pain and
diarrhea in children often indicate gastroenteritis. Ensure
adequate hydration with oral rehydration solutions. Avoid dairy
temporarily. Monitor for dehydration signs. If symptoms persist
beyond 24 hours or if you notice blood in stool, seek immediate
medical attention. Hope this helps. Thank you."
```

### Scoring Method

```python
from bert_score import score

def score_chatdoctor(predictions, references):
    # Calculate BERTScore
    P, R, F1 = score(
        predictions,  # Model responses
        references,   # Ground truth
        lang='en',
        verbose=False
    )

    # Return F1 (harmonic mean of precision & recall)
    return F1.mean().item() * 100
```

**F1 Score**: 0-100 scale (higher = better semantic match)

### Why BERTScore?

Unlike exact string matching:

- **Semantic Awareness**: "fever reducer" ≈ "antipyretic"
- **Flexibility**: Different wordings with same meaning score well
- **Medical Validity**: Captures clinical accuracy
- **Robust**: Handles paraphrasing

**Example**:

```
Ground Truth: "Take acetaminophen for fever"
Response 1: "Take acetaminophen for fever" → F1 = 100 (exact)
Response 2: "Use Tylenol for high temperature" → F1 = 92 (semantic match)
Response 3: "Drink water" → F1 = 45 (semantically different)
```

---

## 🔄 Evaluation Protocol

### Experiment Design

For each model, we run a **paired comparison**:

```
For model in [Llama, GPT, Qwen, Kimi]:
    # Baseline Mode
    baseline_gpqa = run_gpqa(model, mode="baseline")
    baseline_ifeval = run_ifeval(model, mode="baseline")
    baseline_chatdoc = run_chatdoc(model, mode="baseline")

    # Agentic Mode
    agentic_gpqa = run_gpqa(model, mode="agentic")
    agentic_ifeval = run_ifeval(model, mode="agentic")
    agentic_chatdoc = run_chatdoc(model, mode="agentic")

    # Compare
    delta_gpqa = agentic_gpqa - baseline_gpqa
    delta_ifeval = agentic_ifeval - baseline_ifeval
    delta_chatdoc = agentic_chatdoc - baseline_chatdoc
```

### Baseline Prompting

Simple, direct prompts:

```python
async def baseline(prompt):
    response = await llm.ainvoke(prompt)
    return response.strip()
```

**No enhancements**:

- No role definition
- No reasoning guidance
- No constraint emphasis
- Minimal cleaning

### Agentic Prompting

Enhanced with structured guidance:

```python
async def agentic(prompt):
    system_prompt = """You are an expert medical AI assistant.

{input}

Important guidelines:
- For medical questions: Think step-by-step, then select the BEST answer
- For writing constraints: Follow them EXACTLY
- Output ONLY your final answer

Your response:"""

    response = await llm.ainvoke({"input": prompt})
    cleaned = clean_artifacts(response)
    return cleaned
```

**Enhancements**:

1. Role: "expert medical AI assistant"
2. Reasoning: "Think step-by-step"
3. Constraint: "Follow EXACTLY"
4. Output: "ONLY your final answer"
5. Cleaning: Remove `<think>` tags, markdown

### Comparison Metrics

For each benchmark:

1. **Absolute Score**: Raw accuracy or F1
2. **Delta (Δ)**: Agentic - Baseline
3. **Relative Gain**: (Δ / Baseline) × 100%

Example:

```
Llama IFEval:
- Baseline: 50%
- Agentic: 70%
- Delta: +20%
- Relative Gain: +40%
```

---

## 🛠️ Implementation Details

### API Configuration

```python
# Temperature
BASELINE_TEMP = 0.0  # Deterministic
AGENTIC_TEMP = 0.3   # Slight randomness for reasoning

# Model Mapping
MODEL_MAP = {
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "openai/gpt-oss-20b": "llama-3.3-70b-versatile",  # Groq mapping
    "qwen/qwen3-32b": "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905": "moonshotai/kimi-k2-instruct-0905"
}

# Retry Logic
MAX_RETRIES = 3
TIMEOUT = 30  # seconds
```

### Artifact Cleaning

````python
def clean_artifacts(text):
    # 1. Remove think tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. Strip markdown code blocks
    if text.startswith("```") and text.endswith("```"):
        lines = text.split('\n')
        if len(lines) >= 3:
            text = '\n'.join(lines[1:-1])

    # 3. Trim whitespace
    return text.strip()
````

### Error Handling

```python
async def safe_invoke(llm, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke(prompt)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {e}"
            await asyncio.sleep(1)  # Wait before retry
```

---

## 📏 Quality Assurance

### Validation Checks

1. **Output Format**: Verify answers match expected format
2. **Constraint Checking**: Programmatic validation for IFEval
3. **Missing Data**: Handle incomplete responses
4. **API Errors**: Retry logic for transient failures

### Manual Spot-Checks

Performed on 10% of outputs:

- ✅ GPQA: Verify extracted answers are correct letters
- ✅ IFEval: Manually validate constraint adherence
- ✅ ChatDoctor: Read responses for medical accuracy

### Reproducibility

- **Fixed Random Seed**: (when applicable)
- **Saved Prompts**: All prompts logged
- **Version Control**: API library versions pinned
- **Raw Data**: All responses saved to CSV

---

## 🔍 Limitations & Mitigation

### Sample Size

**Limitation**: Only 10 samples per benchmark  
**Impact**: Limited statistical power  
**Mitigation**: Descriptive statistics, transparent reporting  
**Future**: Scale to 50-100 samples

### Benchmark Difficulty

**Limitation**: GPQA may be too easy (3/4 models at 100%)  
**Impact**: Ceiling effects mask potential gains  
**Mitigation**: Acknowledge in results  
**Future**: Add harder diagnostic scenarios

### Prompt Sensitivity

**Limitation**: GPT shows regression with agentic prompts  
**Impact**: Not universally beneficial  
**Mitigation**: Model-specific analysis  
**Future**: Tailored prompts per model family

### Evaluation Metrics

**Limitation**: BERTScore may not capture all medical nuances  
**Impact**: Possible over/under-estimation of quality  
**Mitigation**: Use as proxy, not absolute measure  
**Future**: Human expert evaluation

---

## 📊 Statistical Considerations

### Current Approach

- **Descriptive Statistics**: Mean, median, range
- **Effect Sizes**: Absolute and relative deltas
- **Visual Comparison**: Tables and graphs

### Future Enhancements

With larger samples (50+):

1. **Paired t-tests**: Test significance of improvements
2. **Cohen's d**: Measure effect sizes
3. **Confidence Intervals**: 95% CI for mean differences
4. **Power Analysis**: Determine sample size requirements
5. **Bonferroni Correction**: Adjust for multiple comparisons

---

## ✅ Validation Criteria

An agentic approach is considered **successful** if:

1. **Primary**: Agentic Average > Baseline Average
2. **Secondary**: At least 2/3 benchmarks improve
3. **Safety**: No degradation in GPQA (medical accuracy)
4. **Magnitude**: Improvement > 5% overall

Results:

- Llama: ✅ (4/4 criteria)
- Qwen: ✅ (4/4 criteria)
- Kimi: ✅ (4/4 criteria)
- GPT: ❌ (failed primary criterion)

---

## 🔬 Reproducibility Checklist

To reproduce our results:

- [ ] Install exact dependency versions (requirements.txt)
- [ ] Set GROQ_API_KEY environment variable
- [ ] Use same model names and API endpoints
- [ ] Run with NUM_SAMPLES = 10
- [ ] Verify temperature settings (0.0 baseline, 0.3 agentic)
- [ ] Check prompt templates match exactly
- [ ] Validate cleaning functions
- [ ] Compare outputs to our CSV file

Expected variance: ±2-3% due to API non-determinism

---

## 📚 References

### Benchmarks

1. **GPQA**: Derived from medical question banks
2. **IFEval**: Zhou et al. (2023) "Instruction-Following Evaluation"
3. **ChatDoctor**: Li et al. (2023) "ChatDoctor Dataset"

### Metrics

1. **BERTScore**: Zhang et al. (2020) "BERTScore: Evaluating Text Generation"
2. **Exact Match**: Standard NLP metric
3. **Constraint Checking**: Programmatic validation

### Methodology

1. **Prompt Engineering**: Wei et al. (2022) "Chain-of-Thought Prompting"
2. **Agent Evaluation**: Park et al. (2023) "Generative Agents"
3. **Medical AI**: Singhal et al. (2023) "Med-PaLM 2"

---

**For more information**: See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for architecture details and [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) for findings.
