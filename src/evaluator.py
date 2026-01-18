"""
Benchmark Evaluator - Main Evaluation Engine.

Orchestrates benchmark execution using LangGraph for workflow management.
Provides a unified interface for running all benchmark types.
"""

import asyncio
from typing import Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from src.model_client import ModelClient
from src.benchmarks.quality_benchmark import QualityBenchmark, QualityBenchmarkResult
from src.benchmarks.speed_benchmark import SpeedBenchmark, SpeedBenchmarkResult
from src.benchmarks.cost_benchmark import CostBenchmark, CostBenchmarkResult
from src.benchmarks.task_benchmark import TaskBenchmark, TaskBenchmarkResult, TaskType
from src.report_generator import ReportGenerator
from src.utils import generate_run_id, save_results, calculate_p_value
from config import get_config, RESULTS_DIR


class BenchmarkType(Enum):
    """Available benchmark types."""
    QUALITY = "quality"
    SPEED = "speed"
    COST = "cost"
    TASK = "task"
    COMPREHENSIVE = "comprehensive"


class EvaluationState(BaseModel):
    """State for the evaluation workflow."""

    # Configuration
    run_id: str = Field(default_factory=generate_run_id)
    models: list[str] = Field(default_factory=list)
    benchmark_types: list[str] = Field(default_factory=list)
    task_type: Optional[str] = None
    dataset: list[dict] = Field(default_factory=list)
    system_prompt: Optional[str] = None

    # Results
    quality_results: dict[str, dict] = Field(default_factory=dict)
    speed_results: dict[str, dict] = Field(default_factory=dict)
    cost_results: dict[str, dict] = Field(default_factory=dict)
    task_results: dict[str, dict] = Field(default_factory=dict)

    # Metadata
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "pending"
    errors: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


@dataclass
class ComprehensiveResult:
    """Comprehensive benchmark result across all dimensions."""

    run_id: str
    models: list[str]
    quality_results: dict[str, QualityBenchmarkResult]
    speed_results: dict[str, SpeedBenchmarkResult]
    cost_results: dict[str, CostBenchmarkResult]
    task_results: dict[str, TaskBenchmarkResult] = field(default_factory=dict)
    rankings: dict[str, list[str]] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "models": self.models,
            "quality_results": {
                k: v.to_dict() for k, v in self.quality_results.items()
            },
            "speed_results": {
                k: v.to_dict() for k, v in self.speed_results.items()
            },
            "cost_results": {
                k: v.to_dict() for k, v in self.cost_results.items()
            },
            "task_results": {
                k: v.to_dict() for k, v in self.task_results.items()
            },
            "rankings": self.rankings,
            "recommendations": self.recommendations,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class BenchmarkEvaluator:
    """
    Main benchmark evaluator with LangGraph workflow.

    Provides a unified interface for running benchmarks across
    multiple models and benchmark types.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            api_key: API key for model access
        """
        config = get_config()
        self.api_key = api_key or config.api.groq_api_key
        self.config = config

        # Initialize benchmarks
        self.quality_benchmark = QualityBenchmark(use_bert_score=False)
        self.speed_benchmark = SpeedBenchmark()
        self.cost_benchmark = CostBenchmark()

        # Model clients cache
        self._clients: dict[str, ModelClient] = {}

        # Build the workflow
        self._workflow = self._build_workflow()

    def _get_client(self, model_name: str) -> ModelClient:
        """Get or create a model client."""
        if model_name not in self._clients:
            self._clients[model_name] = ModelClient(
                model_name=model_name,
                api_key=self.api_key,
            )
        return self._clients[model_name]

    def add_model(
        self,
        model_name: str,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Add a model for benchmarking.

        Args:
            model_name: Name of the model
            api_key: Optional specific API key for this model
        """
        self._clients[model_name] = ModelClient(
            model_name=model_name,
            api_key=api_key or self.api_key,
        )

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph evaluation workflow."""

        def initialize(state: EvaluationState) -> dict:
            """Initialize the evaluation run."""
            return {
                "started_at": datetime.now().isoformat(),
                "status": "running",
            }

        def run_quality_benchmark(state: EvaluationState) -> dict:
            """Run quality benchmarks for all models."""
            if "quality" not in state.benchmark_types and "comprehensive" not in state.benchmark_types:
                return {}

            results = {}
            for model_name in state.models:
                try:
                    client = self._get_client(model_name)
                    test_cases = [
                        {"prompt": d.get("prompt", d.get("text", "")),
                         "reference": d.get("reference", d.get("expected", ""))}
                        for d in state.dataset
                        if "reference" in d or "expected" in d
                    ]

                    if test_cases:
                        result = self.quality_benchmark.run(
                            client, test_cases, state.system_prompt
                        )
                        results[model_name] = result.to_dict()
                except Exception as e:
                    state.errors.append(f"Quality benchmark error for {model_name}: {str(e)}")

            return {"quality_results": results}

        def run_speed_benchmark(state: EvaluationState) -> dict:
            """Run speed benchmarks for all models."""
            if "speed" not in state.benchmark_types and "comprehensive" not in state.benchmark_types:
                return {}

            results = {}
            for model_name in state.models:
                try:
                    client = self._get_client(model_name)
                    prompts = [
                        d.get("prompt", d.get("text", ""))
                        for d in state.dataset
                    ]

                    if prompts:
                        result = self.speed_benchmark.run(
                            client, prompts, state.system_prompt
                        )
                        results[model_name] = result.to_dict()
                except Exception as e:
                    state.errors.append(f"Speed benchmark error for {model_name}: {str(e)}")

            return {"speed_results": results}

        def run_cost_benchmark(state: EvaluationState) -> dict:
            """Run cost benchmarks for all models."""
            if "cost" not in state.benchmark_types and "comprehensive" not in state.benchmark_types:
                return {}

            results = {}
            for model_name in state.models:
                try:
                    client = self._get_client(model_name)
                    prompts = [
                        d.get("prompt", d.get("text", ""))
                        for d in state.dataset
                    ]

                    if prompts:
                        result = self.cost_benchmark.run(
                            client, prompts, state.system_prompt
                        )
                        results[model_name] = result.to_dict()
                except Exception as e:
                    state.errors.append(f"Cost benchmark error for {model_name}: {str(e)}")

            return {"cost_results": results}

        def run_task_benchmark(state: EvaluationState) -> dict:
            """Run task-specific benchmarks for all models."""
            if "task" not in state.benchmark_types:
                return {}

            if not state.task_type:
                return {}

            results = {}
            task_type = TaskType(state.task_type)
            task_benchmark = TaskBenchmark(task_type)

            for model_name in state.models:
                try:
                    client = self._get_client(model_name)
                    result = task_benchmark.run(
                        client, state.dataset, state.system_prompt
                    )
                    results[model_name] = result.to_dict()
                except Exception as e:
                    state.errors.append(f"Task benchmark error for {model_name}: {str(e)}")

            return {"task_results": results}

        def finalize(state: EvaluationState) -> dict:
            """Finalize the evaluation run."""
            return {
                "completed_at": datetime.now().isoformat(),
                "status": "completed" if not state.errors else "completed_with_errors",
            }

        # Build the graph
        workflow = StateGraph(EvaluationState)

        # Add nodes
        workflow.add_node("initialize", initialize)
        workflow.add_node("quality", run_quality_benchmark)
        workflow.add_node("speed", run_speed_benchmark)
        workflow.add_node("cost", run_cost_benchmark)
        workflow.add_node("task", run_task_benchmark)
        workflow.add_node("finalize", finalize)

        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "quality")
        workflow.add_edge("quality", "speed")
        workflow.add_edge("speed", "cost")
        workflow.add_edge("cost", "task")
        workflow.add_edge("task", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def run(
        self,
        models: list[str],
        dataset: list[dict],
        benchmark_types: list[str] = None,
        task_type: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Run benchmarks on specified models.

        Args:
            models: List of model names to benchmark
            dataset: Test dataset
            benchmark_types: List of benchmark types to run
            task_type: Task type for task-specific benchmarks
            system_prompt: Optional system prompt

        Returns:
            Complete benchmark results
        """
        if benchmark_types is None:
            benchmark_types = ["comprehensive"]

        # Ensure all models are registered
        for model in models:
            if model not in self._clients:
                self.add_model(model)

        # Create initial state
        initial_state = EvaluationState(
            models=models,
            benchmark_types=benchmark_types,
            task_type=task_type,
            dataset=dataset,
            system_prompt=system_prompt,
        )

        # Run the workflow
        result = self._workflow.invoke(initial_state)

        return {
            "run_id": result["run_id"],
            "models": result["models"],
            "quality_results": result["quality_results"],
            "speed_results": result["speed_results"],
            "cost_results": result["cost_results"],
            "task_results": result["task_results"],
            "started_at": result["started_at"],
            "completed_at": result["completed_at"],
            "status": result["status"],
            "errors": result["errors"],
        }

    def run_quick(
        self,
        models: list[str],
        test_prompts: list[str],
        num_iterations: int = 3,
    ) -> dict:
        """
        Run a quick benchmark with just prompts.

        Args:
            models: List of model names
            test_prompts: List of test prompts
            num_iterations: Number of iterations per model

        Returns:
            Quick benchmark results
        """
        dataset = [{"prompt": p} for p in test_prompts]
        return self.run(
            models=models,
            dataset=dataset,
            benchmark_types=["speed", "cost"],
        )

    def compare_models(
        self,
        results: dict,
    ) -> dict:
        """
        Compare models based on benchmark results.

        Args:
            results: Benchmark results from run()

        Returns:
            Comparison with rankings and recommendations
        """
        rankings = {
            "quality": [],
            "speed": [],
            "cost": [],
            "overall": [],
        }

        # Rank by quality
        if results.get("quality_results"):
            quality_sorted = sorted(
                results["quality_results"].items(),
                key=lambda x: x[1].get("metrics", {}).get("overall_quality", 0),
                reverse=True,
            )
            rankings["quality"] = [m[0] for m in quality_sorted]

        # Rank by speed (lower latency = better)
        if results.get("speed_results"):
            speed_sorted = sorted(
                results["speed_results"].items(),
                key=lambda x: x[1].get("metrics", {}).get("avg_latency_ms", float("inf")),
            )
            rankings["speed"] = [m[0] for m in speed_sorted]

        # Rank by cost (lower = better)
        if results.get("cost_results"):
            cost_sorted = sorted(
                results["cost_results"].items(),
                key=lambda x: x[1].get("metrics", {}).get("total_cost", float("inf")),
            )
            rankings["cost"] = [m[0] for m in cost_sorted]

        # Calculate overall ranking (weighted)
        scores = {}
        for model in results["models"]:
            score = 0

            if model in rankings["quality"]:
                idx = rankings["quality"].index(model)
                score += (len(rankings["quality"]) - idx) * 0.4

            if model in rankings["speed"]:
                idx = rankings["speed"].index(model)
                score += (len(rankings["speed"]) - idx) * 0.3

            if model in rankings["cost"]:
                idx = rankings["cost"].index(model)
                score += (len(rankings["cost"]) - idx) * 0.3

            scores[model] = score

        rankings["overall"] = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Generate recommendations
        recommendations = []
        if rankings["overall"]:
            top_model = rankings["overall"][0]
            recommendations.append(f"Best overall: {top_model}")

        if rankings["quality"]:
            recommendations.append(f"Best quality: {rankings['quality'][0]}")

        if rankings["speed"]:
            recommendations.append(f"Fastest: {rankings['speed'][0]}")

        if rankings["cost"]:
            recommendations.append(f"Most cost-effective: {rankings['cost'][0]}")

        return {
            "rankings": rankings,
            "scores": scores,
            "recommendations": recommendations,
        }

    def generate_report(
        self,
        results: dict,
        format: str = "html",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a benchmark report.

        Args:
            results: Benchmark results
            format: Output format (html, json, markdown)
            output_path: Optional output path

        Returns:
            Path to generated report
        """
        generator = ReportGenerator()
        comparison = self.compare_models(results)

        report_data = {
            **results,
            **comparison,
        }

        if output_path is None:
            output_path = RESULTS_DIR / "reports" / f"report_{results['run_id']}.{format}"

        return generator.generate(report_data, format, output_path)

    def save_results(
        self,
        results: dict,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Save benchmark results to file.

        Args:
            results: Benchmark results
            output_path: Optional output path

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = RESULTS_DIR / f"results_{results['run_id']}.json"

        return str(save_results(results, output_path))
