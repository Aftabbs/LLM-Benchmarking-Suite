"""
Web Search Integration Module.

Provides web search capabilities using Serper API for enriching
benchmark datasets and testing LLM performance on real-time data.
"""

import httpx
from typing import Optional, Any
from dataclasses import dataclass, field
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper

from config import get_config


@dataclass
class SearchResult:
    """Container for search results."""

    title: str
    link: str
    snippet: str
    position: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class WebSearchResponse:
    """Response from web search."""

    query: str
    results: list[SearchResult]
    total_results: int = 0
    search_time_ms: float = 0.0


class WebSearchClient:
    """
    Web search client using Serper API.

    Provides web search capabilities for:
    - Real-time data benchmarks
    - Fact-checking tests
    - Information retrieval evaluation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        num_results: int = 10,
    ):
        """
        Initialize the web search client.

        Args:
            api_key: Serper API key
            num_results: Number of results to return
        """
        config = get_config()
        self.api_key = api_key or config.api.serper_api_key
        self.num_results = num_results
        self.base_url = "https://google.serper.dev/search"

        # Initialize LangChain wrapper if available
        self._langchain_wrapper = None
        if self.api_key:
            try:
                self._langchain_wrapper = GoogleSerperAPIWrapper(
                    serper_api_key=self.api_key,
                    k=num_results,
                )
            except Exception:
                pass

    def search(
        self,
        query: str,
        search_type: str = "search",
        country: str = "us",
        language: str = "en",
    ) -> WebSearchResponse:
        """
        Perform a web search.

        Args:
            query: Search query
            search_type: Type of search (search, news, images)
            country: Country code for results
            language: Language code

        Returns:
            WebSearchResponse with results
        """
        if not self.api_key:
            return WebSearchResponse(
                query=query,
                results=[],
                total_results=0,
            )

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "gl": country,
            "hl": language,
            "num": self.num_results,
        }

        try:
            import time
            start_time = time.perf_counter()

            response = httpx.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()

            end_time = time.perf_counter()
            search_time = (end_time - start_time) * 1000

            data = response.json()

            # Parse organic results
            results = []
            organic = data.get("organic", [])

            for i, item in enumerate(organic):
                result = SearchResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    position=i + 1,
                    metadata={
                        "date": item.get("date"),
                        "source": item.get("source"),
                    }
                )
                results.append(result)

            return WebSearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time_ms=search_time,
            )

        except Exception as e:
            return WebSearchResponse(
                query=query,
                results=[],
                total_results=0,
            )

    def search_news(
        self,
        query: str,
        time_period: str = "day",
    ) -> WebSearchResponse:
        """
        Search for news articles.

        Args:
            query: Search query
            time_period: Time period (day, week, month)

        Returns:
            WebSearchResponse with news results
        """
        if not self.api_key:
            return WebSearchResponse(query=query, results=[])

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "type": "news",
            "tbs": f"qdr:{time_period[0]}",  # d, w, m
            "num": self.num_results,
        }

        try:
            response = httpx.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            news = data.get("news", [])

            for i, item in enumerate(news):
                result = SearchResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    position=i + 1,
                    metadata={
                        "date": item.get("date"),
                        "source": item.get("source"),
                        "imageUrl": item.get("imageUrl"),
                    }
                )
                results.append(result)

            return WebSearchResponse(
                query=query,
                results=results,
                total_results=len(results),
            )

        except Exception:
            return WebSearchResponse(query=query, results=[])

    def get_langchain_tool(self) -> Optional[Tool]:
        """
        Get a LangChain tool for web search.

        Returns:
            LangChain Tool for web search or None
        """
        if not self._langchain_wrapper:
            return None

        return Tool(
            name="web_search",
            description="Search the web for current information",
            func=self._langchain_wrapper.run,
        )

    def create_search_benchmark_data(
        self,
        queries: list[str],
    ) -> list[dict]:
        """
        Create benchmark dataset from search queries.

        Args:
            queries: List of search queries

        Returns:
            Dataset with search results as context
        """
        dataset = []

        for query in queries:
            response = self.search(query)

            if response.results:
                # Create context from search results
                context = "\n\n".join([
                    f"Source: {r.title}\n{r.snippet}"
                    for r in response.results[:3]
                ])

                dataset.append({
                    "prompt": f"Based on the following information, answer the question: {query}\n\nContext:\n{context}",
                    "query": query,
                    "context": context,
                    "sources": [r.link for r in response.results[:3]],
                })

        return dataset


class WebEnrichedBenchmark:
    """
    Benchmark that uses web search to create dynamic test cases.

    Tests LLM performance on real-time, fact-based questions.
    """

    def __init__(
        self,
        search_client: Optional[WebSearchClient] = None,
    ):
        """
        Initialize the web-enriched benchmark.

        Args:
            search_client: WebSearchClient instance
        """
        self.search_client = search_client or WebSearchClient()

    def create_fact_check_dataset(
        self,
        topics: list[str],
        questions_per_topic: int = 3,
    ) -> list[dict]:
        """
        Create a fact-checking dataset from web searches.

        Args:
            topics: List of topics to search
            questions_per_topic: Questions per topic

        Returns:
            Dataset for fact-checking benchmark
        """
        dataset = []

        fact_questions = [
            "What is the latest information about {topic}?",
            "What are the key facts about {topic}?",
            "Summarize recent developments in {topic}.",
        ]

        for topic in topics:
            # Get search results for context
            response = self.search_client.search(topic)

            if not response.results:
                continue

            # Create context from search results
            context = "\n\n".join([
                f"[{r.position}] {r.title}: {r.snippet}"
                for r in response.results[:5]
            ])

            for i, q_template in enumerate(fact_questions[:questions_per_topic]):
                question = q_template.format(topic=topic)

                dataset.append({
                    "prompt": f"Using the following search results, {question.lower()}\n\nSearch Results:\n{context}",
                    "topic": topic,
                    "question": question,
                    "sources": [r.link for r in response.results[:5]],
                    "context": context,
                })

        return dataset

    def create_news_qa_dataset(
        self,
        news_topics: list[str],
    ) -> list[dict]:
        """
        Create a news-based Q&A dataset.

        Args:
            news_topics: List of news topics

        Returns:
            Dataset for news Q&A benchmark
        """
        dataset = []

        for topic in news_topics:
            response = self.search_client.search_news(topic)

            if not response.results:
                continue

            # Use top news story as context
            top_story = response.results[0]

            dataset.append({
                "prompt": f"Based on this news headline and snippet, provide a brief analysis:\n\nHeadline: {top_story.title}\n\n{top_story.snippet}",
                "topic": topic,
                "source": top_story.link,
                "headline": top_story.title,
            })

        return dataset
