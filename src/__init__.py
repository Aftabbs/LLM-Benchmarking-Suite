"""
LLM Benchmarking Suite - Source Package

A comprehensive benchmarking suite for comparing LLM models
across multiple dimensions: quality, speed, cost, and task-specific performance.
"""

from src.model_client import ModelClient, ModelResponse
from src.evaluator import BenchmarkEvaluator
from src.report_generator import ReportGenerator

__all__ = [
    "ModelClient",
    "ModelResponse",
    "BenchmarkEvaluator",
    "ReportGenerator",
]

__version__ = "1.0.0"
