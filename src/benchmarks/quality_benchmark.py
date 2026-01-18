"""
Quality Benchmark Module.

Evaluates LLM output quality using multiple metrics:
- BLEU Score
- ROUGE Score
- BERTScore
- Semantic Similarity
"""

from typing import Optional
from dataclasses import dataclass, field
import numpy as np

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from src.model_client import ModelClient, ModelResponse
from src.utils import calculate_statistics, StatisticalResult


@dataclass
class QualityMetrics:
    """Container for quality evaluation metrics."""

    bleu_score: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bert_score_precision: float = 0.0
    bert_score_recall: float = 0.0
    bert_score_f1: float = 0.0
    semantic_similarity: float = 0.0

    # Composite score
    overall_quality: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "bleu_score": self.bleu_score,
            "rouge_1": self.rouge_1,
            "rouge_2": self.rouge_2,
            "rouge_l": self.rouge_l,
            "bert_score_precision": self.bert_score_precision,
            "bert_score_recall": self.bert_score_recall,
            "bert_score_f1": self.bert_score_f1,
            "semantic_similarity": self.semantic_similarity,
            "overall_quality": self.overall_quality,
        }


@dataclass
class QualityBenchmarkResult:
    """Result of a quality benchmark run."""

    model_name: str
    metrics: QualityMetrics
    metrics_stats: dict[str, StatisticalResult] = field(default_factory=dict)
    num_samples: int = 0
    raw_scores: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "metrics": self.metrics.to_dict(),
            "metrics_stats": {
                k: v.to_dict() for k, v in self.metrics_stats.items()
            },
            "num_samples": self.num_samples,
        }


class QualityBenchmark:
    """
    Benchmark for evaluating LLM output quality.

    Supports multiple quality metrics including BLEU, ROUGE,
    and BERTScore for comprehensive quality assessment.
    """

    def __init__(
        self,
        use_bert_score: bool = True,
        bert_model: str = "microsoft/deberta-xlarge-mnli",
    ):
        """
        Initialize the quality benchmark.

        Args:
            use_bert_score: Whether to compute BERTScore
            bert_model: Model to use for BERTScore
        """
        self.use_bert_score = use_bert_score
        self.bert_model = bert_model

        # Initialize scorers
        self._rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )
        self._bleu = BLEU()

        # Lazy load BERTScore
        self._bert_scorer = None

    def _get_bert_scorer(self):
        """Lazy load BERTScore to avoid slow imports."""
        if self._bert_scorer is None and self.use_bert_score:
            try:
                from bert_score import BERTScorer
                self._bert_scorer = BERTScorer(
                    model_type=self.bert_model,
                    lang="en",
                    rescale_with_baseline=True,
                )
            except ImportError:
                self.use_bert_score = False
                return None
        return self._bert_scorer

    def calculate_bleu(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        """
        Calculate BLEU score.

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            BLEU score (0-100)
        """
        result = self._bleu.sentence_score(
            prediction,
            [reference],
        )
        return result.score

    def calculate_rouge(
        self,
        prediction: str,
        reference: str,
    ) -> dict[str, float]:
        """
        Calculate ROUGE scores.

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        scores = self._rouge_scorer.score(reference, prediction)
        return {
            "rouge_1": scores["rouge1"].fmeasure,
            "rouge_2": scores["rouge2"].fmeasure,
            "rouge_l": scores["rougeL"].fmeasure,
        }

    def calculate_bert_score(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, list[float]]:
        """
        Calculate BERTScore.

        Args:
            predictions: List of generated texts
            references: List of reference texts

        Returns:
            Dictionary with precision, recall, F1 scores
        """
        scorer = self._get_bert_scorer()
        if scorer is None:
            return {
                "precision": [0.0] * len(predictions),
                "recall": [0.0] * len(predictions),
                "f1": [0.0] * len(predictions),
            }

        P, R, F1 = scorer.score(predictions, references)
        return {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist(),
        }

    def evaluate_single(
        self,
        prediction: str,
        reference: str,
    ) -> QualityMetrics:
        """
        Evaluate quality for a single prediction-reference pair.

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            QualityMetrics with all scores
        """
        # BLEU
        bleu = self.calculate_bleu(prediction, reference)

        # ROUGE
        rouge = self.calculate_rouge(prediction, reference)

        # BERTScore (single sample)
        bert_scores = self.calculate_bert_score([prediction], [reference])

        # Calculate overall quality (weighted average)
        overall = (
            (bleu / 100) * 0.2 +  # BLEU is 0-100, normalize
            rouge["rouge_l"] * 0.3 +
            bert_scores["f1"][0] * 0.5
        )

        return QualityMetrics(
            bleu_score=bleu,
            rouge_1=rouge["rouge_1"],
            rouge_2=rouge["rouge_2"],
            rouge_l=rouge["rouge_l"],
            bert_score_precision=bert_scores["precision"][0],
            bert_score_recall=bert_scores["recall"][0],
            bert_score_f1=bert_scores["f1"][0],
            overall_quality=overall,
        )

    def run(
        self,
        client: ModelClient,
        test_cases: list[dict],
        system_prompt: Optional[str] = None,
    ) -> QualityBenchmarkResult:
        """
        Run quality benchmark on a model.

        Args:
            client: ModelClient to evaluate
            test_cases: List of dicts with 'prompt' and 'reference' keys
            system_prompt: Optional system prompt

        Returns:
            QualityBenchmarkResult with aggregated metrics
        """
        predictions = []
        references = []
        raw_scores = {
            "bleu": [],
            "rouge_1": [],
            "rouge_2": [],
            "rouge_l": [],
        }

        # Generate predictions
        for case in test_cases:
            prompt = case["prompt"]
            reference = case["reference"]

            response = client.invoke(prompt, system_prompt)
            prediction = response.content

            predictions.append(prediction)
            references.append(reference)

            # Calculate per-sample scores
            bleu = self.calculate_bleu(prediction, reference)
            rouge = self.calculate_rouge(prediction, reference)

            raw_scores["bleu"].append(bleu)
            raw_scores["rouge_1"].append(rouge["rouge_1"])
            raw_scores["rouge_2"].append(rouge["rouge_2"])
            raw_scores["rouge_l"].append(rouge["rouge_l"])

        # Calculate BERTScore for all samples at once
        bert_scores = self.calculate_bert_score(predictions, references)
        raw_scores["bert_f1"] = bert_scores["f1"]

        # Calculate statistics
        metrics_stats = {
            metric: calculate_statistics(scores)
            for metric, scores in raw_scores.items()
        }

        # Calculate aggregate metrics
        avg_metrics = QualityMetrics(
            bleu_score=metrics_stats["bleu"].mean,
            rouge_1=metrics_stats["rouge_1"].mean,
            rouge_2=metrics_stats["rouge_2"].mean,
            rouge_l=metrics_stats["rouge_l"].mean,
            bert_score_f1=metrics_stats.get("bert_f1", StatisticalResult(
                mean=0, median=0, std_dev=0, min_val=0, max_val=0,
                confidence_interval_95=(0, 0), sample_size=0
            )).mean,
        )

        # Calculate overall quality
        avg_metrics.overall_quality = (
            (avg_metrics.bleu_score / 100) * 0.2 +
            avg_metrics.rouge_l * 0.3 +
            avg_metrics.bert_score_f1 * 0.5
        )

        return QualityBenchmarkResult(
            model_name=client.model_name,
            metrics=avg_metrics,
            metrics_stats=metrics_stats,
            num_samples=len(test_cases),
            raw_scores=raw_scores,
        )

    def compare(
        self,
        results: list[QualityBenchmarkResult],
    ) -> dict:
        """
        Compare quality results across multiple models.

        Args:
            results: List of benchmark results

        Returns:
            Comparison summary
        """
        comparison = {
            "models": [],
            "best_overall": None,
            "best_bleu": None,
            "best_rouge": None,
            "best_bert": None,
        }

        best_overall_score = -1
        best_bleu_score = -1
        best_rouge_score = -1
        best_bert_score = -1

        for result in results:
            model_summary = {
                "model": result.model_name,
                "overall_quality": result.metrics.overall_quality,
                "bleu": result.metrics.bleu_score,
                "rouge_l": result.metrics.rouge_l,
                "bert_f1": result.metrics.bert_score_f1,
            }
            comparison["models"].append(model_summary)

            if result.metrics.overall_quality > best_overall_score:
                best_overall_score = result.metrics.overall_quality
                comparison["best_overall"] = result.model_name

            if result.metrics.bleu_score > best_bleu_score:
                best_bleu_score = result.metrics.bleu_score
                comparison["best_bleu"] = result.model_name

            if result.metrics.rouge_l > best_rouge_score:
                best_rouge_score = result.metrics.rouge_l
                comparison["best_rouge"] = result.model_name

            if result.metrics.bert_score_f1 > best_bert_score:
                best_bert_score = result.metrics.bert_score_f1
                comparison["best_bert"] = result.model_name

        return comparison
