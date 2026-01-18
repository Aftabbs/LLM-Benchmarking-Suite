"""
Configuration module for LLM Benchmarking Suite.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"

# Ensure directories exist
DATASETS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


class ModelConfig(BaseModel):
    """Configuration for LLM models."""

    name: str = Field(default="openai/gpt-oss-120b")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout: int = Field(default=120)


class APIConfig(BaseModel):
    """API configuration settings."""

    groq_api_key: Optional[str] = Field(default=None)
    serper_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load API configuration from environment variables."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            serper_api_key=os.getenv("SERPER_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )


class BenchmarkConfig(BaseModel):
    """Benchmark execution configuration."""

    timeout: int = Field(default=120)
    max_concurrent_requests: int = Field(default=5)
    num_iterations: int = Field(default=3)
    warmup_iterations: int = Field(default=1)

    # Metric weights for composite scoring
    quality_weight: float = Field(default=0.4)
    speed_weight: float = Field(default=0.3)
    cost_weight: float = Field(default=0.3)


class AppConfig(BaseModel):
    """Main application configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig.from_env)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    # Supported models for benchmarking
    supported_models: list[str] = Field(default=[
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> AppConfig:
    """Update configuration with new values."""
    global config
    config = AppConfig(**kwargs)
    return config
