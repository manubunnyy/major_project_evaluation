# Unified Health Report Evaluation System

This project is a comprehensive evaluation framework for medical health report analysis. It compares a **Baseline** (single-shot LLM) approach against a **Sequential Agentic Chain** using various state-of-the-art models.
## Link to Other README files:
[results](doctor_eval/docs/RESULTS_ANALYSIS.md)
## 🚀 Overview

The system uses a 4-step sequential reasoning chain to analyze medical lab results:

1.  **Extraction**: Filtering noise and extracting key metrics (values, ranges, units).
2.  **Risk Analysis**: Focusing ONLY on abnormal or concerning results.
3.  **Positive Analysis**: Identifying healthy markers and optimal levels.
4.  **Clinical Synthesis**: Producing a high-level concise clinical summary.

## 📊 Evaluation Results

The agentic approach consistently outperforms the baseline across multiple models:

| MODEL                    | BASE (%) | AGENT (%) | CHANGE    |
| :----------------------- | :------- | :-------- | :-------- |
| **gemini-2.5-flash**     | 80.73    | 89.41     | **+8.68** |
| **openai/gpt-oss-20b**   | 76.52    | 84.64     | **+8.12** |
| **llama-3.1-8b-instant** | 78.76    | 81.67     | **+2.91** |
| **qwen/qwen3-32b**       | 81.20    | 81.72     | **+0.52** |

_Scores are calculated using BERTScore (F1) against professional ground truth summaries._

## 📁 Project Structure

- `agent_folder/`: Core logic for the unified evaluation.
  - `run_evaluation.py`: Main script to run the benchmark.
  - `core/health_analyzer.py`: The multi-agent implementation.
  - `data/`: PDF reports and ground truth data.
- `prompt_folder/`: Legacy research scripts and prompt engineering experiments.
- `requirements.txt`: Python dependencies.


## 🧠 Key Features

- **Sequential Chain**: Passed extracted data between agents to maintain context while minimizing token usage.
- **Auto-Truncation**: Automatically handles large PDFs by truncating text to stay within model limits.
- **BERTScore Metrics**: Uses semantic similarity instead of simple keyword matching for medical accuracy.
- **Multi-Model Support**: Native integration with Google Gemini and Groq (Llama, Qwen, etc.).
