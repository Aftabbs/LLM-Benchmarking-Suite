"""
Unit tests for LLM Benchmarking Suite.

Tests the core functionality of benchmark modules,
model client, and evaluation engine.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_client import ModelClient, ModelResponse, MODEL_PRICING
from src.benchmarks.quality_benchmark import QualityBenchmark, QualityMetrics
from src.benchmarks.speed_benchmark import SpeedBenchmark, SpeedMetrics
from src.benchmarks.cost_benchmark import CostBenchmark, CostMetrics
from src.benchmarks.task_benchmark import TaskBenchmark, TaskType
from src.utils import (
    calculate_statistics,
    normalize_score,
    format_duration,
    format_cost,
    truncate_text,
)


class TestUtils:
    """Tests for utility functions."""

    def test_calculate_statistics_empty(self):
        """Test statistics with empty list."""
        result = calculate_statistics([])
        assert result.mean == 0.0
        assert result.sample_size == 0

    def test_calculate_statistics_single(self):
        """Test statistics with single value."""
        result = calculate_statistics([5.0])
        assert result.mean == 5.0
        assert result.median == 5.0
        assert result.sample_size == 1

    def test_calculate_statistics_multiple(self):
        """Test statistics with multiple values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_statistics(values)
        assert result.mean == 3.0
        assert result.median == 3.0
        assert result.min_val == 1.0
        assert result.max_val == 5.0
        assert result.sample_size == 5

    def test_normalize_score_higher_is_better(self):
        """Test normalization when higher is better."""
        score = normalize_score(75, 0, 100, higher_is_better=True)
        assert score == 0.75

    def test_normalize_score_lower_is_better(self):
        """Test normalization when lower is better."""
        score = normalize_score(25, 0, 100, higher_is_better=False)
        assert score == 0.75

    def test_normalize_score_clipping(self):
        """Test that scores are clipped to 0-1."""
        score = normalize_score(150, 0, 100)
        assert score == 1.0

        score = normalize_score(-50, 0, 100)
        assert score == 0.0

    def test_format_duration_ms(self):
        """Test duration formatting for milliseconds."""
        assert format_duration(500) == "500.00ms"

    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        assert format_duration(5000) == "5.00s"

    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        result = format_duration(90000)
        assert "1m" in result

    def test_format_cost(self):
        """Test cost formatting."""
        assert "$" in format_cost(0.001)
        assert "$" in format_cost(1.50)

    def test_truncate_text_short(self):
        """Test truncation with short text."""
        text = "Hello"
        assert truncate_text(text, 100) == "Hello"

    def test_truncate_text_long(self):
        """Test truncation with long text."""
        text = "This is a very long text that should be truncated"
        result = truncate_text(text, 20)
        assert len(result) == 20
        assert result.endswith("...")


class TestModelClient:
    """Tests for ModelClient."""

    def test_model_pricing_exists(self):
        """Test that model pricing is defined."""
        assert len(MODEL_PRICING) > 0
        assert "openai/gpt-oss-120b" in MODEL_PRICING

    def test_model_response_tokens_per_second(self):
        """Test tokens per second calculation."""
        response = ModelResponse(
            content="Test response",
            model="test-model",
            latency_ms=1000,
            input_tokens=10,
            output_tokens=50,
            total_tokens=60,
        )
        assert response.tokens_per_second == 50.0

    def test_model_response_zero_latency(self):
        """Test tokens per second with zero latency."""
        response = ModelResponse(
            content="Test",
            model="test",
            latency_ms=0,
            input_tokens=10,
            output_tokens=50,
            total_tokens=60,
        )
        assert response.tokens_per_second == 0.0

    def test_supported_models(self):
        """Test getting supported models."""
        with patch.object(ModelClient, '_create_client', return_value=Mock()):
            client = ModelClient(
                model_name="test-model",
                api_key="test-key"
            )
            models = client.supported_models
            assert isinstance(models, list)
            assert len(models) > 0


class TestQualityBenchmark:
    """Tests for QualityBenchmark."""

    def test_bleu_calculation(self):
        """Test BLEU score calculation."""
        benchmark = QualityBenchmark(use_bert_score=False)

        # Same text should have high BLEU
        score = benchmark.calculate_bleu(
            "The quick brown fox",
            "The quick brown fox"
        )
        assert score > 50  # BLEU is 0-100

    def test_rouge_calculation(self):
        """Test ROUGE score calculation."""
        benchmark = QualityBenchmark(use_bert_score=False)

        scores = benchmark.calculate_rouge(
            "The quick brown fox jumps",
            "The quick brown fox"
        )

        assert "rouge_1" in scores
        assert "rouge_2" in scores
        assert "rouge_l" in scores
        assert all(0 <= v <= 1 for v in scores.values())

    def test_quality_metrics_to_dict(self):
        """Test QualityMetrics serialization."""
        metrics = QualityMetrics(
            bleu_score=50.0,
            rouge_1=0.8,
            rouge_2=0.6,
            rouge_l=0.7,
        )
        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result["bleu_score"] == 50.0


class TestSpeedBenchmark:
    """Tests for SpeedBenchmark."""

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        benchmark = SpeedBenchmark()
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        p50 = benchmark._calculate_percentile(values, 50)
        assert 50 <= p50 <= 60

        p95 = benchmark._calculate_percentile(values, 95)
        assert p95 >= 90

    def test_percentile_empty_list(self):
        """Test percentile with empty list."""
        benchmark = SpeedBenchmark()
        result = benchmark._calculate_percentile([], 50)
        assert result == 0.0

    def test_speed_metrics_to_dict(self):
        """Test SpeedMetrics serialization."""
        metrics = SpeedMetrics(
            avg_latency_ms=100.0,
            p50_latency_ms=95.0,
            p95_latency_ms=150.0,
            total_requests=10,
            successful_requests=9,
        )
        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result["avg_latency_ms"] == 100.0


class TestCostBenchmark:
    """Tests for CostBenchmark."""

    def test_cost_calculation(self):
        """Test cost calculation."""
        benchmark = CostBenchmark()

        cost = benchmark.calculate_cost(
            "openai/gpt-oss-120b",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost > 0
        assert isinstance(cost, float)

    def test_custom_pricing(self):
        """Test custom pricing."""
        custom_pricing = {
            "custom-model": (1.0, 2.0)  # $1 per 1M input, $2 per 1M output
        }
        benchmark = CostBenchmark(custom_pricing=custom_pricing)

        pricing = benchmark.get_pricing("custom-model")
        assert pricing == (1.0, 2.0)

    def test_monthly_cost_estimate(self):
        """Test monthly cost estimation."""
        benchmark = CostBenchmark()

        estimate = benchmark.estimate_monthly_cost(
            model_name="openai/gpt-oss-120b",
            requests_per_day=100,
            avg_input_tokens=500,
            avg_output_tokens=200,
        )

        assert "daily_cost" in estimate
        assert "monthly_cost" in estimate
        assert estimate["monthly_cost"] > estimate["daily_cost"]


class TestTaskBenchmark:
    """Tests for TaskBenchmark."""

    def test_task_types(self):
        """Test that all task types are valid."""
        for task_type in TaskType:
            benchmark = TaskBenchmark(task_type)
            assert benchmark.task_type == task_type

    def test_classification_evaluation(self):
        """Test classification evaluation."""
        benchmark = TaskBenchmark(TaskType.CLASSIFICATION)

        result = benchmark._evaluate_classification(
            prediction="positive",
            expected="positive",
            categories=["positive", "negative", "neutral"]
        )

        assert result["is_correct"] == True
        assert result["is_valid_category"] == True

    def test_qa_evaluation(self):
        """Test Q&A evaluation."""
        benchmark = TaskBenchmark(TaskType.QUESTION_ANSWERING)

        result = benchmark._evaluate_qa(
            prediction="The answer is 42",
            expected="42"
        )

        assert result["contains_match"] == True

    def test_code_evaluation(self):
        """Test code evaluation."""
        benchmark = TaskBenchmark(TaskType.CODE_GENERATION)

        result = benchmark._evaluate_code(
            prediction="def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
        )

        assert result["has_code"] == True
        assert result["has_basic_structure"] == True

    def test_factory_methods(self):
        """Test factory methods."""
        summarization = TaskBenchmark.create_summarization_benchmark()
        assert summarization.task_type == TaskType.SUMMARIZATION

        classification = TaskBenchmark.create_classification_benchmark()
        assert classification.task_type == TaskType.CLASSIFICATION

        qa = TaskBenchmark.create_qa_benchmark()
        assert qa.task_type == TaskType.QUESTION_ANSWERING

        code = TaskBenchmark.create_code_benchmark()
        assert code.task_type == TaskType.CODE_GENERATION


class TestIntegration:
    """Integration tests (require mocking)."""

    @patch('src.model_client.ChatGroq')
    def test_model_client_invoke(self, mock_groq):
        """Test model client invocation."""
        # Setup mock
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
        mock_groq.return_value.invoke.return_value = mock_response

        client = ModelClient(
            model_name="test-model",
            api_key="test-key"
        )

        response = client.invoke("Test prompt")

        assert response.content == "Test response"
        assert response.model == "test-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
