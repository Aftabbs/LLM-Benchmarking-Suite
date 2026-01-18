"""
Unified Model Client for LLM Benchmarking Suite.

Provides a consistent interface for interacting with multiple LLM providers
using LangChain and Groq integration.
"""

import time
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_config, ModelConfig


class ModelProvider(Enum):
    """Supported model providers."""
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelResponse:
    """Standardized response from model invocation."""

    content: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens generated per second."""
        if self.latency_ms > 0:
            return (self.output_tokens / self.latency_ms) * 1000
        return 0.0


@dataclass
class ModelPricing:
    """Pricing information for a model (per 1M tokens)."""

    input_price: float  # Price per 1M input tokens
    output_price: float  # Price per 1M output tokens


# Pricing table for supported models (per 1M tokens)
MODEL_PRICING = {
    "openai/gpt-oss-120b": ModelPricing(input_price=0.05, output_price=0.10),
    "llama-3.3-70b-versatile": ModelPricing(input_price=0.59, output_price=0.79),
    "llama-3.1-8b-instant": ModelPricing(input_price=0.05, output_price=0.08),
    "mixtral-8x7b-32768": ModelPricing(input_price=0.24, output_price=0.24),
    "gemma2-9b-it": ModelPricing(input_price=0.20, output_price=0.20),
}


class ModelClient:
    """
    Unified client for interacting with LLM models.

    Supports multiple providers through LangChain abstractions,
    with primary focus on Groq API integration.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
    ):
        """
        Initialize the model client.

        Args:
            model_name: Name of the model to use
            api_key: API key for the provider
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        config = get_config()

        self.model_name = model_name or config.model.name
        self.api_key = api_key or config.api.groq_api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Initialize the LangChain model
        self._client = self._create_client()
        self._output_parser = StrOutputParser()

    def _create_client(self) -> ChatGroq:
        """Create the appropriate LangChain client."""
        return ChatGroq(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: ~4 characters per token for English
        return len(text) // 4

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for the request."""
        pricing = MODEL_PRICING.get(
            self.model_name,
            ModelPricing(input_price=0.10, output_price=0.10)
        )

        input_cost = (input_tokens / 1_000_000) * pricing.input_price
        output_cost = (output_tokens / 1_000_000) * pricing.output_price

        return input_cost + output_cost

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """
        Invoke the model with a prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            ModelResponse with generated content and metrics
        """
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Measure latency
        start_time = time.perf_counter()

        response = self._client.invoke(messages)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract content
        content = response.content

        # Get token counts from response metadata if available
        usage_metadata = getattr(response, 'usage_metadata', None)

        if usage_metadata:
            input_tokens = usage_metadata.get('input_tokens', 0)
            output_tokens = usage_metadata.get('output_tokens', 0)
            total_tokens = usage_metadata.get('total_tokens', 0)
        else:
            # Estimate tokens
            input_text = (system_prompt or "") + prompt
            input_tokens = self._estimate_tokens(input_text)
            output_tokens = self._estimate_tokens(content)
            total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost = self._calculate_cost(input_tokens, output_tokens)

        return ModelResponse(
            content=content,
            model=self.model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            metadata={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """
        Asynchronously invoke the model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            ModelResponse with generated content and metrics
        """
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        start_time = time.perf_counter()

        response = await self._client.ainvoke(messages)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        content = response.content

        usage_metadata = getattr(response, 'usage_metadata', None)

        if usage_metadata:
            input_tokens = usage_metadata.get('input_tokens', 0)
            output_tokens = usage_metadata.get('output_tokens', 0)
            total_tokens = usage_metadata.get('total_tokens', 0)
        else:
            input_text = (system_prompt or "") + prompt
            input_tokens = self._estimate_tokens(input_text)
            output_tokens = self._estimate_tokens(content)
            total_tokens = input_tokens + output_tokens

        cost = self._calculate_cost(input_tokens, output_tokens)

        return ModelResponse(
            content=content,
            model=self.model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            metadata={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )

    def batch_invoke(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
    ) -> list[ModelResponse]:
        """
        Invoke the model with multiple prompts.

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt for all

        Returns:
            List of ModelResponse objects
        """
        return [
            self.invoke(prompt, system_prompt)
            for prompt in prompts
        ]

    def create_chain(
        self,
        prompt_template: str,
        input_variables: list[str],
    ):
        """
        Create a LangChain chain with the model.

        Args:
            prompt_template: The prompt template string
            input_variables: List of input variable names

        Returns:
            Runnable chain
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self._client | self._output_parser
        return chain

    def update_model(self, model_name: str) -> None:
        """
        Update the model being used.

        Args:
            model_name: New model name
        """
        self.model_name = model_name
        self._client = self._create_client()

    def update_temperature(self, temperature: float) -> None:
        """
        Update the sampling temperature.

        Args:
            temperature: New temperature value
        """
        self.temperature = temperature
        self._client = self._create_client()

    @property
    def supported_models(self) -> list[str]:
        """Get list of supported models."""
        return list(MODEL_PRICING.keys())

    def get_model_info(self) -> dict[str, Any]:
        """Get current model configuration info."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "pricing": MODEL_PRICING.get(self.model_name),
        }
