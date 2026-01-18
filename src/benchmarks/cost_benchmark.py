"""
Cost Benchmark Module.

Evaluates LLM cost metrics:
- Cost per 1K tokens
- Cost per task
- Cost efficiency (quality per dollar)
"""

from typing import Optional
from dataclasses import dataclass, field

from src.model_client import ModelClient, ModelResponse, MODEL_PRICING
from src.utils import calculate_statistics, StatisticalResult


@dataclass
class CostMetrics:
    """Container for cost evaluation metrics."""

    total_cost: float = 0.0
    avg_cost_per_request: float = 0.0
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    cost_per_1k_total_tokens: float = 0.0

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    # Efficiency metrics (if quality is provided)
    cost_efficiency: float = 0.0  # Quality per dollar

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_cost": self.total_cost,
            "avg_cost_per_request": self.avg_cost_per_request,
            "cost_per_1k_input_tokens": self.cost_per_1k_input_tokens,
            "cost_per_1k_output_tokens": self.cost_per_1k_output_tokens,
            "cost_per_1k_total_tokens": self.cost_per_1k_total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "cost_efficiency": self.cost_efficiency,
        }


@dataclass
class CostBenchmarkResult:
    """Result of a cost benchmark run."""

    model_name: str
    metrics: CostMetrics
    cost_stats: StatisticalResult = None
    token_stats: dict[str, StatisticalResult] = field(default_factory=dict)
    raw_costs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "metrics": self.metrics.to_dict(),
            "cost_stats": self.cost_stats.to_dict() if self.cost_stats else None,
            "token_stats": {
                k: v.to_dict() for k, v in self.token_stats.items()
            },
        }


class CostBenchmark:
    """
    Benchmark for evaluating LLM cost metrics.

    Tracks token usage and calculates cost based on model pricing.
    """

    def __init__(
        self,
        custom_pricing: Optional[dict[str, tuple[float, float]]] = None,
    ):
        """
        Initialize the cost benchmark.

        Args:
            custom_pricing: Optional custom pricing per model
                           Format: {model_name: (input_price_per_1m, output_price_per_1m)}
        """
        self.custom_pricing = custom_pricing or {}

    def get_pricing(self, model_name: str) -> tuple[float, float]:
        """
        Get pricing for a model.

        Returns:
            Tuple of (input_price_per_1m, output_price_per_1m)
        """
        if model_name in self.custom_pricing:
            return self.custom_pricing[model_name]

        pricing = MODEL_PRICING.get(model_name)
        if pricing:
            return (pricing.input_price, pricing.output_price)

        # Default pricing
        return (0.10, 0.10)

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in dollars
        """
        input_price, output_price = self.get_pricing(model_name)

        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price

        return input_cost + output_cost

    def run(
        self,
        client: ModelClient,
        test_prompts: list[str],
        system_prompt: Optional[str] = None,
    ) -> CostBenchmarkResult:
        """
        Run cost benchmark on a model.

        Args:
            client: ModelClient to evaluate
            test_prompts: List of prompts to test with
            system_prompt: Optional system prompt

        Returns:
            CostBenchmarkResult with cost metrics
        """
        costs = []
        input_tokens_list = []
        output_tokens_list = []

        for prompt in test_prompts:
            response = client.invoke(prompt, system_prompt)

            costs.append(response.cost)
            input_tokens_list.append(response.input_tokens)
            output_tokens_list.append(response.output_tokens)

        # Calculate aggregate metrics
        total_cost = sum(costs)
        total_input = sum(input_tokens_list)
        total_output = sum(output_tokens_list)
        total_tokens = total_input + total_output
        num_requests = len(test_prompts)

        # Calculate stats
        cost_stats = calculate_statistics(costs)
        token_stats = {
            "input_tokens": calculate_statistics(
                [float(t) for t in input_tokens_list]
            ),
            "output_tokens": calculate_statistics(
                [float(t) for t in output_tokens_list]
            ),
        }

        # Get per-1k pricing
        input_price, output_price = self.get_pricing(client.model_name)
        cost_per_1k_input = input_price / 1000
        cost_per_1k_output = output_price / 1000

        metrics = CostMetrics(
            total_cost=total_cost,
            avg_cost_per_request=total_cost / num_requests if num_requests > 0 else 0,
            cost_per_1k_input_tokens=cost_per_1k_input,
            cost_per_1k_output_tokens=cost_per_1k_output,
            cost_per_1k_total_tokens=(total_cost / total_tokens * 1000) if total_tokens > 0 else 0,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            avg_input_tokens=total_input / num_requests if num_requests > 0 else 0,
            avg_output_tokens=total_output / num_requests if num_requests > 0 else 0,
        )

        return CostBenchmarkResult(
            model_name=client.model_name,
            metrics=metrics,
            cost_stats=cost_stats,
            token_stats=token_stats,
            raw_costs=costs,
        )

    def calculate_efficiency(
        self,
        cost_result: CostBenchmarkResult,
        quality_score: float,
    ) -> float:
        """
        Calculate cost efficiency (quality per dollar).

        Args:
            cost_result: Cost benchmark result
            quality_score: Quality score (0-1)

        Returns:
            Cost efficiency score
        """
        if cost_result.metrics.total_cost > 0:
            return quality_score / cost_result.metrics.total_cost
        return 0.0

    def estimate_monthly_cost(
        self,
        model_name: str,
        requests_per_day: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
    ) -> dict:
        """
        Estimate monthly cost for a usage pattern.

        Args:
            model_name: Model to estimate for
            requests_per_day: Average daily requests
            avg_input_tokens: Average input tokens per request
            avg_output_tokens: Average output tokens per request

        Returns:
            Cost estimation breakdown
        """
        daily_input = requests_per_day * avg_input_tokens
        daily_output = requests_per_day * avg_output_tokens

        daily_cost = self.calculate_cost(model_name, daily_input, daily_output)
        weekly_cost = daily_cost * 7
        monthly_cost = daily_cost * 30

        return {
            "model": model_name,
            "requests_per_day": requests_per_day,
            "daily_tokens": daily_input + daily_output,
            "daily_cost": daily_cost,
            "weekly_cost": weekly_cost,
            "monthly_cost": monthly_cost,
            "yearly_cost": monthly_cost * 12,
        }

    def compare(
        self,
        results: list[CostBenchmarkResult],
    ) -> dict:
        """
        Compare cost results across multiple models.

        Args:
            results: List of benchmark results

        Returns:
            Comparison summary
        """
        comparison = {
            "models": [],
            "cheapest_overall": None,
            "cheapest_per_request": None,
            "best_token_value": None,
        }

        cheapest_total = float("inf")
        cheapest_per_request = float("inf")
        best_token_value = float("inf")

        for result in results:
            model_summary = {
                "model": result.model_name,
                "total_cost": result.metrics.total_cost,
                "avg_cost_per_request": result.metrics.avg_cost_per_request,
                "cost_per_1k_tokens": result.metrics.cost_per_1k_total_tokens,
            }
            comparison["models"].append(model_summary)

            if result.metrics.total_cost < cheapest_total:
                cheapest_total = result.metrics.total_cost
                comparison["cheapest_overall"] = result.model_name

            if result.metrics.avg_cost_per_request < cheapest_per_request:
                cheapest_per_request = result.metrics.avg_cost_per_request
                comparison["cheapest_per_request"] = result.model_name

            if result.metrics.cost_per_1k_total_tokens < best_token_value:
                best_token_value = result.metrics.cost_per_1k_total_tokens
                comparison["best_token_value"] = result.model_name

        return comparison
