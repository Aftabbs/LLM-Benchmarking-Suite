"""
Speed Benchmark Module.

Evaluates LLM performance metrics:
- Latency (time to first token, total time)
- Throughput (tokens per second)
- Concurrency handling
"""

import time
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from src.model_client import ModelClient, ModelResponse
from src.utils import calculate_statistics, StatisticalResult


@dataclass
class SpeedMetrics:
    """Container for speed evaluation metrics."""

    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    avg_tokens_per_second: float = 0.0
    avg_output_tokens: float = 0.0

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "avg_output_tokens": self.avg_output_tokens,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": self.error_rate,
        }


@dataclass
class SpeedBenchmarkResult:
    """Result of a speed benchmark run."""

    model_name: str
    metrics: SpeedMetrics
    latency_stats: StatisticalResult = None
    throughput_stats: StatisticalResult = None
    raw_latencies: list[float] = field(default_factory=list)
    raw_throughputs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "metrics": self.metrics.to_dict(),
            "latency_stats": self.latency_stats.to_dict() if self.latency_stats else None,
            "throughput_stats": self.throughput_stats.to_dict() if self.throughput_stats else None,
        }


class SpeedBenchmark:
    """
    Benchmark for evaluating LLM speed and performance.

    Measures latency, throughput, and reliability metrics.
    """

    def __init__(
        self,
        num_iterations: int = 10,
        warmup_iterations: int = 2,
        max_concurrent: int = 5,
    ):
        """
        Initialize the speed benchmark.

        Args:
            num_iterations: Number of test iterations
            warmup_iterations: Number of warmup runs (not counted)
            max_concurrent: Maximum concurrent requests for load testing
        """
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.max_concurrent = max_concurrent

    def _calculate_percentile(
        self,
        values: list[float],
        percentile: float,
    ) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

    def run_single_request(
        self,
        client: ModelClient,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> tuple[Optional[ModelResponse], Optional[str]]:
        """
        Run a single request and measure performance.

        Returns:
            Tuple of (response, error_message)
        """
        try:
            response = client.invoke(prompt, system_prompt)
            return response, None
        except Exception as e:
            return None, str(e)

    def run(
        self,
        client: ModelClient,
        test_prompts: list[str],
        system_prompt: Optional[str] = None,
    ) -> SpeedBenchmarkResult:
        """
        Run speed benchmark on a model.

        Args:
            client: ModelClient to evaluate
            test_prompts: List of prompts to test with
            system_prompt: Optional system prompt

        Returns:
            SpeedBenchmarkResult with performance metrics
        """
        latencies = []
        throughputs = []
        output_tokens_list = []
        failed_count = 0

        # Warmup runs
        for i in range(self.warmup_iterations):
            prompt = test_prompts[i % len(test_prompts)]
            self.run_single_request(client, prompt, system_prompt)

        # Actual benchmark runs
        for i in range(self.num_iterations):
            prompt = test_prompts[i % len(test_prompts)]
            response, error = self.run_single_request(
                client, prompt, system_prompt
            )

            if response:
                latencies.append(response.latency_ms)
                throughputs.append(response.tokens_per_second)
                output_tokens_list.append(response.output_tokens)
            else:
                failed_count += 1

        # Calculate metrics
        total_requests = self.num_iterations
        successful_requests = total_requests - failed_count

        if latencies:
            latency_stats = calculate_statistics(latencies)
            throughput_stats = calculate_statistics(throughputs)

            metrics = SpeedMetrics(
                avg_latency_ms=latency_stats.mean,
                p50_latency_ms=self._calculate_percentile(latencies, 50),
                p95_latency_ms=self._calculate_percentile(latencies, 95),
                p99_latency_ms=self._calculate_percentile(latencies, 99),
                min_latency_ms=latency_stats.min_val,
                max_latency_ms=latency_stats.max_val,
                avg_tokens_per_second=throughput_stats.mean,
                avg_output_tokens=sum(output_tokens_list) / len(output_tokens_list),
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_count,
                error_rate=failed_count / total_requests if total_requests > 0 else 0,
            )
        else:
            latency_stats = None
            throughput_stats = None
            metrics = SpeedMetrics(
                total_requests=total_requests,
                failed_requests=failed_count,
                error_rate=1.0,
            )

        return SpeedBenchmarkResult(
            model_name=client.model_name,
            metrics=metrics,
            latency_stats=latency_stats,
            throughput_stats=throughput_stats,
            raw_latencies=latencies,
            raw_throughputs=throughputs,
        )

    async def run_concurrent(
        self,
        client: ModelClient,
        test_prompts: list[str],
        concurrency: int = 5,
        system_prompt: Optional[str] = None,
    ) -> SpeedBenchmarkResult:
        """
        Run concurrent load test.

        Args:
            client: ModelClient to evaluate
            test_prompts: List of prompts to test
            concurrency: Number of concurrent requests
            system_prompt: Optional system prompt

        Returns:
            SpeedBenchmarkResult with concurrent performance metrics
        """
        latencies = []
        throughputs = []
        output_tokens_list = []
        failed_count = 0

        async def make_request(prompt: str):
            try:
                response = await client.ainvoke(prompt, system_prompt)
                return response, None
            except Exception as e:
                return None, str(e)

        # Create batches of concurrent requests
        for i in range(0, len(test_prompts), concurrency):
            batch = test_prompts[i:i + concurrency]
            tasks = [make_request(p) for p in batch]
            results = await asyncio.gather(*tasks)

            for response, error in results:
                if response:
                    latencies.append(response.latency_ms)
                    throughputs.append(response.tokens_per_second)
                    output_tokens_list.append(response.output_tokens)
                else:
                    failed_count += 1

        # Calculate metrics
        total_requests = len(test_prompts)
        successful_requests = total_requests - failed_count

        if latencies:
            latency_stats = calculate_statistics(latencies)
            throughput_stats = calculate_statistics(throughputs)

            metrics = SpeedMetrics(
                avg_latency_ms=latency_stats.mean,
                p50_latency_ms=self._calculate_percentile(latencies, 50),
                p95_latency_ms=self._calculate_percentile(latencies, 95),
                p99_latency_ms=self._calculate_percentile(latencies, 99),
                min_latency_ms=latency_stats.min_val,
                max_latency_ms=latency_stats.max_val,
                avg_tokens_per_second=throughput_stats.mean,
                avg_output_tokens=sum(output_tokens_list) / len(output_tokens_list),
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_count,
                error_rate=failed_count / total_requests if total_requests > 0 else 0,
            )
        else:
            latency_stats = None
            throughput_stats = None
            metrics = SpeedMetrics(
                total_requests=total_requests,
                failed_requests=failed_count,
                error_rate=1.0,
            )

        return SpeedBenchmarkResult(
            model_name=client.model_name,
            metrics=metrics,
            latency_stats=latency_stats,
            throughput_stats=throughput_stats,
            raw_latencies=latencies,
            raw_throughputs=throughputs,
        )

    def compare(
        self,
        results: list[SpeedBenchmarkResult],
    ) -> dict:
        """
        Compare speed results across multiple models.

        Args:
            results: List of benchmark results

        Returns:
            Comparison summary
        """
        comparison = {
            "models": [],
            "fastest_avg": None,
            "fastest_p95": None,
            "highest_throughput": None,
            "most_reliable": None,
        }

        fastest_avg = float("inf")
        fastest_p95 = float("inf")
        highest_throughput = 0
        lowest_error_rate = float("inf")

        for result in results:
            model_summary = {
                "model": result.model_name,
                "avg_latency_ms": result.metrics.avg_latency_ms,
                "p95_latency_ms": result.metrics.p95_latency_ms,
                "tokens_per_second": result.metrics.avg_tokens_per_second,
                "error_rate": result.metrics.error_rate,
            }
            comparison["models"].append(model_summary)

            if result.metrics.avg_latency_ms < fastest_avg:
                fastest_avg = result.metrics.avg_latency_ms
                comparison["fastest_avg"] = result.model_name

            if result.metrics.p95_latency_ms < fastest_p95:
                fastest_p95 = result.metrics.p95_latency_ms
                comparison["fastest_p95"] = result.model_name

            if result.metrics.avg_tokens_per_second > highest_throughput:
                highest_throughput = result.metrics.avg_tokens_per_second
                comparison["highest_throughput"] = result.model_name

            if result.metrics.error_rate < lowest_error_rate:
                lowest_error_rate = result.metrics.error_rate
                comparison["most_reliable"] = result.model_name

        return comparison
