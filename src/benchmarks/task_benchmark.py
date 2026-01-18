"""
Task-Specific Benchmark Module.

Evaluates LLMs on specific tasks:
- Summarization
- Classification
- Question Answering
- Code Generation
- Translation
"""

from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.model_client import ModelClient, ModelResponse
from src.benchmarks.quality_benchmark import QualityBenchmark, QualityMetrics
from src.utils import calculate_statistics, StatisticalResult


class TaskType(Enum):
    """Supported task types."""
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    QUESTION_ANSWERING = "qa"
    CODE_GENERATION = "code"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative"
    REASONING = "reasoning"


@dataclass
class TaskMetrics:
    """Container for task-specific metrics."""

    task_type: str
    accuracy: float = 0.0  # For classification/QA
    quality_score: float = 0.0  # For generation tasks
    task_completion_rate: float = 0.0
    format_compliance: float = 0.0  # Did the output follow expected format

    # Task-specific metrics
    custom_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "accuracy": self.accuracy,
            "quality_score": self.quality_score,
            "task_completion_rate": self.task_completion_rate,
            "format_compliance": self.format_compliance,
            "custom_metrics": self.custom_metrics,
        }


@dataclass
class TaskBenchmarkResult:
    """Result of a task-specific benchmark run."""

    model_name: str
    task_type: str
    metrics: TaskMetrics
    quality_metrics: Optional[QualityMetrics] = None
    num_samples: int = 0
    raw_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "metrics": self.metrics.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict() if self.quality_metrics else None,
            "num_samples": self.num_samples,
        }


# Task-specific prompt templates
TASK_TEMPLATES = {
    TaskType.SUMMARIZATION: """Summarize the following text in 2-3 sentences:

{text}

Summary:""",

    TaskType.CLASSIFICATION: """Classify the following text into one of these categories: {categories}

Text: {text}

Category:""",

    TaskType.QUESTION_ANSWERING: """Answer the following question based on the context provided.

Context: {context}

Question: {question}

Answer:""",

    TaskType.CODE_GENERATION: """Write a function that {description}

Requirements:
{requirements}

Code:""",

    TaskType.TRANSLATION: """Translate the following text from {source_lang} to {target_lang}:

{text}

Translation:""",

    TaskType.CREATIVE_WRITING: """Write a {style} about {topic}.

Requirements: {requirements}

Output:""",

    TaskType.REASONING: """Solve the following problem step by step:

{problem}

Solution:""",
}


class TaskBenchmark:
    """
    Benchmark for evaluating LLMs on specific tasks.

    Supports multiple task types with task-specific evaluation metrics.
    """

    def __init__(
        self,
        task_type: TaskType,
        quality_benchmark: Optional[QualityBenchmark] = None,
    ):
        """
        Initialize the task benchmark.

        Args:
            task_type: Type of task to benchmark
            quality_benchmark: Optional quality benchmark for generation tasks
        """
        self.task_type = task_type
        self.quality_benchmark = quality_benchmark or QualityBenchmark(
            use_bert_score=False  # Faster for task benchmarks
        )
        self.template = TASK_TEMPLATES.get(task_type, "{text}")

    def _format_prompt(self, test_case: dict) -> str:
        """Format the prompt template with test case data."""
        return self.template.format(**test_case)

    def _evaluate_classification(
        self,
        prediction: str,
        expected: str,
        categories: list[str],
    ) -> dict:
        """Evaluate classification output."""
        prediction_clean = prediction.strip().lower()
        expected_clean = expected.strip().lower()

        # Check for exact match
        is_correct = prediction_clean == expected_clean

        # Check if prediction is a valid category
        is_valid = any(
            cat.lower() in prediction_clean
            for cat in categories
        )

        return {
            "is_correct": is_correct,
            "is_valid_category": is_valid,
            "predicted": prediction_clean,
            "expected": expected_clean,
        }

    def _evaluate_qa(
        self,
        prediction: str,
        expected: str,
    ) -> dict:
        """Evaluate Q&A output."""
        prediction_clean = prediction.strip().lower()
        expected_clean = expected.strip().lower()

        # Check for exact match
        exact_match = prediction_clean == expected_clean

        # Check for contains match
        contains_match = expected_clean in prediction_clean

        return {
            "exact_match": exact_match,
            "contains_match": contains_match,
            "predicted": prediction_clean,
            "expected": expected_clean,
        }

    def _evaluate_code(
        self,
        prediction: str,
        test_cases: Optional[list[dict]] = None,
    ) -> dict:
        """Evaluate code generation output."""
        # Check for code block presence
        has_code = "def " in prediction or "function " in prediction

        # Check for common syntax errors (basic check)
        has_basic_structure = any([
            "return " in prediction,
            "print(" in prediction,
            "console.log(" in prediction,
        ])

        return {
            "has_code": has_code,
            "has_basic_structure": has_basic_structure,
            "code_length": len(prediction),
        }

    def run(
        self,
        client: ModelClient,
        test_cases: list[dict],
        system_prompt: Optional[str] = None,
    ) -> TaskBenchmarkResult:
        """
        Run task-specific benchmark.

        Args:
            client: ModelClient to evaluate
            test_cases: List of test cases (format depends on task type)
            system_prompt: Optional system prompt

        Returns:
            TaskBenchmarkResult with task metrics
        """
        results = []
        correct_count = 0
        valid_count = 0
        quality_scores = []

        for case in test_cases:
            prompt = self._format_prompt(case)
            response = client.invoke(prompt, system_prompt)
            prediction = response.content

            result = {
                "prompt": prompt,
                "prediction": prediction,
                "latency_ms": response.latency_ms,
                "tokens": response.total_tokens,
            }

            # Task-specific evaluation
            if self.task_type == TaskType.CLASSIFICATION:
                eval_result = self._evaluate_classification(
                    prediction,
                    case.get("expected", ""),
                    case.get("categories", []),
                )
                result.update(eval_result)
                if eval_result["is_correct"]:
                    correct_count += 1
                if eval_result["is_valid_category"]:
                    valid_count += 1

            elif self.task_type == TaskType.QUESTION_ANSWERING:
                eval_result = self._evaluate_qa(
                    prediction,
                    case.get("expected", ""),
                )
                result.update(eval_result)
                if eval_result["exact_match"] or eval_result["contains_match"]:
                    correct_count += 1
                valid_count += 1

            elif self.task_type == TaskType.CODE_GENERATION:
                eval_result = self._evaluate_code(
                    prediction,
                    case.get("test_cases"),
                )
                result.update(eval_result)
                if eval_result["has_code"]:
                    valid_count += 1
                if eval_result["has_basic_structure"]:
                    correct_count += 1

            elif self.task_type in [
                TaskType.SUMMARIZATION,
                TaskType.TRANSLATION,
                TaskType.CREATIVE_WRITING,
            ]:
                # Use quality benchmark for generation tasks
                if "reference" in case:
                    quality = self.quality_benchmark.evaluate_single(
                        prediction,
                        case["reference"],
                    )
                    quality_scores.append(quality.overall_quality)
                    result["quality_score"] = quality.overall_quality
                valid_count += 1

            results.append(result)

        # Calculate aggregate metrics
        num_samples = len(test_cases)
        accuracy = correct_count / num_samples if num_samples > 0 else 0
        completion_rate = valid_count / num_samples if num_samples > 0 else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        metrics = TaskMetrics(
            task_type=self.task_type.value,
            accuracy=accuracy,
            quality_score=avg_quality,
            task_completion_rate=completion_rate,
            format_compliance=completion_rate,
        )

        return TaskBenchmarkResult(
            model_name=client.model_name,
            task_type=self.task_type.value,
            metrics=metrics,
            num_samples=num_samples,
            raw_results=results,
        )

    @staticmethod
    def create_summarization_benchmark() -> "TaskBenchmark":
        """Create a summarization benchmark."""
        return TaskBenchmark(TaskType.SUMMARIZATION)

    @staticmethod
    def create_classification_benchmark() -> "TaskBenchmark":
        """Create a classification benchmark."""
        return TaskBenchmark(TaskType.CLASSIFICATION)

    @staticmethod
    def create_qa_benchmark() -> "TaskBenchmark":
        """Create a Q&A benchmark."""
        return TaskBenchmark(TaskType.QUESTION_ANSWERING)

    @staticmethod
    def create_code_benchmark() -> "TaskBenchmark":
        """Create a code generation benchmark."""
        return TaskBenchmark(TaskType.CODE_GENERATION)

    def compare(
        self,
        results: list[TaskBenchmarkResult],
    ) -> dict:
        """
        Compare task results across multiple models.

        Args:
            results: List of benchmark results

        Returns:
            Comparison summary
        """
        comparison = {
            "task_type": self.task_type.value,
            "models": [],
            "best_accuracy": None,
            "best_quality": None,
            "best_completion": None,
        }

        best_accuracy = 0
        best_quality = 0
        best_completion = 0

        for result in results:
            model_summary = {
                "model": result.model_name,
                "accuracy": result.metrics.accuracy,
                "quality_score": result.metrics.quality_score,
                "completion_rate": result.metrics.task_completion_rate,
            }
            comparison["models"].append(model_summary)

            if result.metrics.accuracy > best_accuracy:
                best_accuracy = result.metrics.accuracy
                comparison["best_accuracy"] = result.model_name

            if result.metrics.quality_score > best_quality:
                best_quality = result.metrics.quality_score
                comparison["best_quality"] = result.model_name

            if result.metrics.task_completion_rate > best_completion:
                best_completion = result.metrics.task_completion_rate
                comparison["best_completion"] = result.model_name

        return comparison
