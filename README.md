# LLM Benchmarking Suite 
 
<img width="1172" height="626" alt="image" src="https://github.com/user-attachments/assets/f96517a0-792c-452a-b75c-9a74cfa65d4e" />
   

A comprehensive benchmarking suite for comparing LLM models across multiple dimensions: quality, speed, cost, and task-specific performance. Built with LangChain, LangGraph, and Groq API.

## Features 

- **Multi-Model Support**: Groq-hosted models including Llama, Mixtral, Gemma, and more
- **Quality Metrics**: BLEU, ROUGE, BERTScore for text quality evaluation
- **Performance Metrics**: Latency (avg, p50, p95, p99), throughput, tokens per second
- **Cost Analysis**: Cost per request, cost per 1K tokens, monthly projections
- **Task-Specific Benchmarks**: Summarization, classification, Q&A, code generation, reasoning
- **Web-Enriched Benchmarks**: Real-time data testing with Serper API integration
- **Interactive UI**: Streamlit-based dashboard for easy benchmarking
- **LangGraph Workflow**: Orchestrated benchmark execution with state management
- **Comprehensive Reports**: HTML, Markdown, and JSON report generation with Plotly visualizations

## Project Structure

```
llm-benchmarking-suite/
├── app.py                      # Streamlit UI application
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── src/
│   ├── __init__.py
│   ├── model_client.py         # Unified LangChain/Groq model interface
│   ├── evaluator.py            # LangGraph-based evaluation engine
│   ├── report_generator.py     # Report generation with visualizations
│   ├── web_search.py           # Serper API integration
│   ├── utils.py                # Helper functions
│   └── benchmarks/
│       ├── __init__.py
│       ├── quality_benchmark.py    # BLEU, ROUGE, BERTScore
│       ├── speed_benchmark.py      # Latency/throughput tests
│       ├── cost_benchmark.py       # Cost analysis
│       └── task_benchmark.py       # Task-specific tests
├── datasets/
│   └── sample_datasets.json    # Example test datasets
├── results/
│   └── reports/                # Generated reports
├── examples/
│   └── run_benchmark.py        # CLI usage examples
└── tests/
    └── test_benchmarks.py      # Unit tests
```

## Installation

1. Clone the repository and navigate to the project:
```bash
cd llm-benchmarking-suite
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Copy example and edit with your API keys
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

## Quick Start

### Using the Streamlit UI

```bash
streamlit run app.py
```

This opens an interactive dashboard where you can:
- Select models to benchmark
- Choose benchmark types (quality, speed, cost, task-specific)
- Use built-in datasets or custom prompts
- View results with interactive charts
- Generate and download reports

### Using Python API

```python
from src.evaluator import BenchmarkEvaluator

# Initialize with your API key
evaluator = BenchmarkEvaluator(api_key="your_groq_api_key")

# Define test dataset
dataset = [
    {"prompt": "Explain machine learning in simple terms."},
    {"prompt": "What is the difference between Python and JavaScript?"},
    {"prompt": "Describe cloud computing benefits."},
]

# Run comprehensive benchmark
results = evaluator.run(
    models=["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    dataset=dataset,
    benchmark_types=["comprehensive"],
    system_prompt="You are a helpful assistant."
)

# Compare models
comparison = evaluator.compare_models(results)

# Generate report
evaluator.generate_report(results, format="html")
```

### Using CLI

```bash
# Full benchmark
python examples/run_benchmark.py --mode full

# Quick speed test
python examples/run_benchmark.py --mode quick

# Single model test
python examples/run_benchmark.py --mode single
```

<img width="1918" height="867" alt="image" src="https://github.com/user-attachments/assets/c678414a-cda2-4632-9283-262d5fe7b439" />

<img width="1895" height="832" alt="image" src="https://github.com/user-attachments/assets/7c7df369-c2c9-4cc1-adbd-5c149a4a1e2d" />


## Supported Models

The suite supports Groq-hosted models:

| Model | Input Cost (per 1M) | Output Cost (per 1M) |
|-------|---------------------|----------------------|
| openai/gpt-oss-120b | $0.05 | $0.10 |
| llama-3.3-70b-versatile | $0.59 | $0.79 |
| llama-3.1-8b-instant | $0.05 | $0.08 |
| mixtral-8x7b-32768 | $0.24 | $0.24 |
| gemma2-9b-it | $0.20 | $0.20 |

## Benchmark Types

### Quality Benchmark
Evaluates output quality using:
- **BLEU Score**: N-gram overlap with reference
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L
- **BERTScore**: Semantic similarity using embeddings

### Speed Benchmark
Measures performance:
- Average latency (ms)
- P50, P95, P99 latency
- Tokens per second throughput
- Error rate

### Cost Benchmark
Tracks costs:
- Total cost per benchmark run
- Average cost per request
- Cost per 1K tokens
- Monthly cost projections

### Task-Specific Benchmark
Tests specific capabilities:
- Summarization
- Classification
- Question Answering
- Code Generation
- Translation
- Creative Writing
- Reasoning

## Web Search Integration

Use Serper API for dynamic, real-time benchmarks:

```python
from src.web_search import WebSearchClient, WebEnrichedBenchmark

search_client = WebSearchClient(api_key="your_serper_key")
web_benchmark = WebEnrichedBenchmark(search_client)

# Create fact-checking dataset from current web data
dataset = web_benchmark.create_fact_check_dataset(
    topics=["artificial intelligence", "climate change"],
    questions_per_topic=3
)
```

## Sample Datasets

The suite includes sample datasets for:
- Summarization (news articles)
- Sentiment classification
- Reading comprehension Q&A
- Code generation
- Logical reasoning
- General prompts

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

Edit `config.py` or use environment variables:

```python
# config.py settings
BENCHMARK_TIMEOUT = 120           # Request timeout
MAX_CONCURRENT_REQUESTS = 5       # Parallel request limit
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.7
```

## Report Formats

Generate reports in multiple formats:

```python
# HTML with interactive Plotly charts
evaluator.generate_report(results, format="html")

# Markdown for documentation
evaluator.generate_report(results, format="markdown")

# JSON for programmatic access
evaluator.generate_report(results, format="json")
```

## Architecture

The suite uses:
- **LangChain**: Unified model interface via `langchain-groq`
- **LangGraph**: Workflow orchestration for benchmark execution
- **Groq API**: Fast LLM inference
- **Streamlit**: Interactive web UI
- **Plotly**: Interactive visualizations
- **Pydantic**: Configuration validation

## Use Cases

- Model selection for production applications
- Cost optimization analysis
- Performance benchmarking before deployment
- Quality assurance for LLM outputs
- Comparing prompt strategies
- Tracking model performance over time

## License

MIT License
