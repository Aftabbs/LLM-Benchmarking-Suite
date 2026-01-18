"""
Example: Running LLM Benchmarks

This example demonstrates how to use the LLM Benchmarking Suite
to compare multiple models across quality, speed, and cost dimensions.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.evaluator import BenchmarkEvaluator
from src.model_client import ModelClient
from src.report_generator import ReportGenerator
from config import DATASETS_DIR
import json


def main():
    """Run a comprehensive benchmark comparison."""

    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set")
        print("Please set your API key: export GROQ_API_KEY=your_key_here")
        return

    print("=" * 60)
    print("LLM Benchmarking Suite - Example")
    print("=" * 60)

    # Define models to compare
    models = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    print(f"\nModels to benchmark: {models}")

    # Load sample dataset
    dataset_path = DATASETS_DIR / "sample_datasets.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    # Use general prompts for benchmarking
    dataset = datasets.get("general", {}).get("data", [])
    print(f"Dataset size: {len(dataset)} samples")

    # Initialize evaluator
    evaluator = BenchmarkEvaluator(api_key=api_key)

    # Run comprehensive benchmark
    print("\nRunning benchmarks...")
    print("-" * 40)

    results = evaluator.run(
        models=models,
        dataset=dataset,
        benchmark_types=["comprehensive"],
        system_prompt="You are a helpful assistant. Provide clear and concise responses.",
    )

    # Add comparison
    comparison = evaluator.compare_models(results)
    results.update(comparison)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nRun ID: {results.get('run_id')}")
    print(f"Status: {results.get('status')}")

    # Rankings
    print("\n📊 Rankings:")
    rankings = results.get("rankings", {})

    if rankings.get("quality"):
        print(f"  Quality:  {' > '.join(rankings['quality'])}")
    if rankings.get("speed"):
        print(f"  Speed:    {' > '.join(rankings['speed'])}")
    if rankings.get("cost"):
        print(f"  Cost:     {' > '.join(rankings['cost'])}")
    if rankings.get("overall"):
        print(f"  Overall:  {' > '.join(rankings['overall'])}")

    # Recommendations
    print("\n💡 Recommendations:")
    for rec in results.get("recommendations", []):
        print(f"  • {rec}")

    # Detailed metrics
    print("\n📈 Detailed Metrics:")

    if results.get("speed_results"):
        print("\n  Speed Metrics:")
        for model, data in results["speed_results"].items():
            metrics = data.get("metrics", {})
            print(f"    {model}:")
            print(f"      Avg Latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
            print(f"      Tokens/sec:  {metrics.get('avg_tokens_per_second', 0):.2f}")

    if results.get("cost_results"):
        print("\n  Cost Metrics:")
        for model, data in results["cost_results"].items():
            metrics = data.get("metrics", {})
            print(f"    {model}:")
            print(f"      Total Cost:    ${metrics.get('total_cost', 0):.6f}")
            print(f"      Cost/Request:  ${metrics.get('avg_cost_per_request', 0):.6f}")

    # Generate report
    print("\n📄 Generating report...")
    report_path = evaluator.generate_report(results, format="html")
    print(f"Report saved to: {report_path}")

    # Save results
    results_path = evaluator.save_results(results)
    print(f"Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


def quick_benchmark():
    """Run a quick speed benchmark."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    print("Running quick benchmark...")

    # Simple test prompts
    prompts = [
        "What is machine learning?",
        "Explain the concept of recursion.",
        "What are the benefits of cloud computing?",
    ]

    evaluator = BenchmarkEvaluator(api_key=api_key)

    results = evaluator.run_quick(
        models=["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        test_prompts=prompts,
        num_iterations=3,
    )

    print("\nQuick Benchmark Results:")
    for model, data in results.get("speed_results", {}).items():
        metrics = data.get("metrics", {})
        print(f"  {model}: {metrics.get('avg_latency_ms', 0):.2f}ms avg latency")


def single_model_test():
    """Test a single model."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    print("Testing single model...")

    client = ModelClient(
        model_name="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.7,
    )

    response = client.invoke(
        "Explain quantum computing in simple terms.",
        system_prompt="You are a helpful science educator."
    )

    print(f"\nModel: {response.model}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    print(f"Tokens: {response.total_tokens}")
    print(f"Cost: ${response.cost:.6f}")
    print(f"\nResponse:\n{response.content}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Benchmark Examples")
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "single"],
        default="full",
        help="Benchmark mode to run"
    )

    args = parser.parse_args()

    if args.mode == "full":
        main()
    elif args.mode == "quick":
        quick_benchmark()
    else:
        single_model_test()
