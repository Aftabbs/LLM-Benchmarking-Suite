"""
Report Generator Module.

Generates comprehensive benchmark reports with visualizations.
Supports multiple output formats: HTML, JSON, Markdown.
"""

import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class ReportGenerator:
    """
    Generates benchmark reports with visualizations.

    Supports HTML, JSON, and Markdown output formats.
    """

    def __init__(self):
        """Initialize the report generator."""
        self.charts = []

    def generate(
        self,
        results: dict,
        format: str = "html",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a benchmark report.

        Args:
            results: Benchmark results dictionary
            format: Output format (html, json, markdown)
            output_path: Path to save the report

        Returns:
            Path to the generated report
        """
        if format == "html":
            content = self._generate_html(results)
        elif format == "json":
            content = self._generate_json(results)
        elif format == "markdown":
            content = self._generate_markdown(results)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            return str(output_path)

        return content

    def _create_quality_chart(self, quality_results: dict) -> go.Figure:
        """Create quality comparison chart."""
        models = list(quality_results.keys())
        metrics = ["bleu_score", "rouge_l", "bert_score_f1", "overall_quality"]

        data = []
        for metric in metrics:
            values = []
            for model in models:
                model_data = quality_results[model].get("metrics", {})
                value = model_data.get(metric, 0)
                # Normalize BLEU to 0-1 for comparison
                if metric == "bleu_score":
                    value = value / 100
                values.append(value)

            data.append(go.Bar(name=metric.replace("_", " ").title(), x=models, y=values))

        fig = go.Figure(data=data)
        fig.update_layout(
            title="Quality Metrics Comparison",
            barmode="group",
            yaxis_title="Score",
            xaxis_title="Model",
        )

        return fig

    def _create_speed_chart(self, speed_results: dict) -> go.Figure:
        """Create speed comparison chart."""
        models = list(speed_results.keys())

        # Latency chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Average Latency (ms)", "Tokens per Second")
        )

        latencies = [
            speed_results[m].get("metrics", {}).get("avg_latency_ms", 0)
            for m in models
        ]
        throughputs = [
            speed_results[m].get("metrics", {}).get("avg_tokens_per_second", 0)
            for m in models
        ]

        fig.add_trace(
            go.Bar(x=models, y=latencies, name="Latency"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=throughputs, name="Throughput"),
            row=1, col=2
        )

        fig.update_layout(title="Speed Metrics Comparison", showlegend=False)

        return fig

    def _create_cost_chart(self, cost_results: dict) -> go.Figure:
        """Create cost comparison chart."""
        models = list(cost_results.keys())

        costs = [
            cost_results[m].get("metrics", {}).get("total_cost", 0)
            for m in models
        ]

        fig = go.Figure(data=[
            go.Bar(x=models, y=costs, marker_color="green")
        ])

        fig.update_layout(
            title="Cost Comparison",
            yaxis_title="Total Cost ($)",
            xaxis_title="Model",
        )

        return fig

    def _create_overall_comparison(self, results: dict) -> go.Figure:
        """Create overall comparison radar chart."""
        models = results.get("models", [])
        if not models:
            return None

        categories = ["Quality", "Speed", "Cost", "Reliability"]

        fig = go.Figure()

        for model in models:
            values = []

            # Quality score
            quality_data = results.get("quality_results", {}).get(model, {})
            quality = quality_data.get("metrics", {}).get("overall_quality", 0)
            values.append(quality)

            # Speed score (normalized)
            speed_data = results.get("speed_results", {}).get(model, {})
            latency = speed_data.get("metrics", {}).get("avg_latency_ms", 10000)
            speed_score = max(0, 1 - (latency / 10000))  # Normalize
            values.append(speed_score)

            # Cost score (normalized, lower is better)
            cost_data = results.get("cost_results", {}).get(model, {})
            cost = cost_data.get("metrics", {}).get("total_cost", 1)
            cost_score = max(0, 1 - (cost * 10))  # Normalize
            values.append(cost_score)

            # Reliability score
            error_rate = speed_data.get("metrics", {}).get("error_rate", 0)
            reliability = 1 - error_rate
            values.append(reliability)

            # Close the radar
            values.append(values[0])
            categories_closed = categories + [categories[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                name=model,
                fill="toself",
            ))

        fig.update_layout(
            title="Overall Model Comparison",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
        )

        return fig

    def _generate_html(self, results: dict) -> str:
        """Generate HTML report."""
        charts_html = []

        # Quality chart
        if results.get("quality_results"):
            chart = self._create_quality_chart(results["quality_results"])
            charts_html.append(chart.to_html(full_html=False, include_plotlyjs=False))

        # Speed chart
        if results.get("speed_results"):
            chart = self._create_speed_chart(results["speed_results"])
            charts_html.append(chart.to_html(full_html=False, include_plotlyjs=False))

        # Cost chart
        if results.get("cost_results"):
            chart = self._create_cost_chart(results["cost_results"])
            charts_html.append(chart.to_html(full_html=False, include_plotlyjs=False))

        # Overall comparison
        if results.get("models"):
            chart = self._create_overall_comparison(results)
            if chart:
                charts_html.append(chart.to_html(full_html=False, include_plotlyjs=False))

        # Build rankings table
        rankings_html = self._create_rankings_table(results)

        # Build summary
        summary_html = self._create_summary(results)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Benchmark Report - {results.get('run_id', 'Unknown')}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .recommendation {{
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin: 10px 0;
        }}
        .metric {{
            display: inline-block;
            background: #e3f2fd;
            padding: 5px 10px;
            border-radius: 5px;
            margin: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Benchmark Report</h1>
        <p>Run ID: {results.get('run_id', 'Unknown')}</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Models: {', '.join(results.get('models', []))}</p>
    </div>

    <div class="card">
        <h2>Summary</h2>
        {summary_html}
    </div>

    <div class="card">
        <h2>Rankings</h2>
        {rankings_html}
    </div>

    <div class="card">
        <h2>Recommendations</h2>
        {''.join(f'<div class="recommendation">{r}</div>' for r in results.get('recommendations', []))}
    </div>

    {''.join(f'<div class="card">{c}</div>' for c in charts_html)}

    <div class="card">
        <h2>Detailed Results</h2>
        <details>
            <summary>View Raw JSON</summary>
            <pre>{json.dumps(results, indent=2, default=str)}</pre>
        </details>
    </div>
</body>
</html>
"""
        return html

    def _create_rankings_table(self, results: dict) -> str:
        """Create rankings table HTML."""
        rankings = results.get("rankings", {})
        if not rankings:
            return "<p>No rankings available</p>"

        html = "<table><tr><th>Category</th><th>1st</th><th>2nd</th><th>3rd</th></tr>"

        for category, models in rankings.items():
            row = f"<tr><td><strong>{category.title()}</strong></td>"
            for i in range(3):
                if i < len(models):
                    row += f"<td>{models[i]}</td>"
                else:
                    row += "<td>-</td>"
            row += "</tr>"
            html += row

        html += "</table>"
        return html

    def _create_summary(self, results: dict) -> str:
        """Create summary HTML."""
        models = results.get("models", [])
        html = f"<p><strong>Models tested:</strong> {len(models)}</p>"

        if results.get("quality_results"):
            html += f"<span class='metric'>Quality benchmarks completed</span>"
        if results.get("speed_results"):
            html += f"<span class='metric'>Speed benchmarks completed</span>"
        if results.get("cost_results"):
            html += f"<span class='metric'>Cost benchmarks completed</span>"
        if results.get("task_results"):
            html += f"<span class='metric'>Task benchmarks completed</span>"

        if results.get("errors"):
            html += f"<p style='color: red;'>Errors: {len(results['errors'])}</p>"

        return html

    def _generate_json(self, results: dict) -> str:
        """Generate JSON report."""
        return json.dumps(results, indent=2, default=str)

    def _generate_markdown(self, results: dict) -> str:
        """Generate Markdown report."""
        md = f"""# LLM Benchmark Report

**Run ID:** {results.get('run_id', 'Unknown')}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Models:** {', '.join(results.get('models', []))}

## Summary

- Models tested: {len(results.get('models', []))}
- Quality benchmarks: {'Yes' if results.get('quality_results') else 'No'}
- Speed benchmarks: {'Yes' if results.get('speed_results') else 'No'}
- Cost benchmarks: {'Yes' if results.get('cost_results') else 'No'}
- Task benchmarks: {'Yes' if results.get('task_results') else 'No'}

## Rankings

| Category | 1st | 2nd | 3rd |
|----------|-----|-----|-----|
"""

        rankings = results.get("rankings", {})
        for category, models in rankings.items():
            row = f"| {category.title()} |"
            for i in range(3):
                if i < len(models):
                    row += f" {models[i]} |"
                else:
                    row += " - |"
            md += row + "\n"

        md += "\n## Recommendations\n\n"
        for rec in results.get("recommendations", []):
            md += f"- {rec}\n"

        if results.get("quality_results"):
            md += "\n## Quality Results\n\n"
            md += "| Model | BLEU | ROUGE-L | BERTScore F1 | Overall |\n"
            md += "|-------|------|---------|--------------|----------|\n"
            for model, data in results["quality_results"].items():
                metrics = data.get("metrics", {})
                md += f"| {model} | {metrics.get('bleu_score', 0):.2f} | "
                md += f"{metrics.get('rouge_l', 0):.3f} | "
                md += f"{metrics.get('bert_score_f1', 0):.3f} | "
                md += f"{metrics.get('overall_quality', 0):.3f} |\n"

        if results.get("speed_results"):
            md += "\n## Speed Results\n\n"
            md += "| Model | Avg Latency (ms) | P95 Latency | Tokens/sec |\n"
            md += "|-------|------------------|-------------|------------|\n"
            for model, data in results["speed_results"].items():
                metrics = data.get("metrics", {})
                md += f"| {model} | {metrics.get('avg_latency_ms', 0):.2f} | "
                md += f"{metrics.get('p95_latency_ms', 0):.2f} | "
                md += f"{metrics.get('avg_tokens_per_second', 0):.2f} |\n"

        if results.get("cost_results"):
            md += "\n## Cost Results\n\n"
            md += "| Model | Total Cost | Avg/Request | Cost/1K Tokens |\n"
            md += "|-------|------------|-------------|----------------|\n"
            for model, data in results["cost_results"].items():
                metrics = data.get("metrics", {})
                md += f"| {model} | ${metrics.get('total_cost', 0):.6f} | "
                md += f"${metrics.get('avg_cost_per_request', 0):.6f} | "
                md += f"${metrics.get('cost_per_1k_total_tokens', 0):.6f} |\n"

        return md

    def create_comparison_dataframe(self, results: dict) -> pd.DataFrame:
        """
        Create a pandas DataFrame for comparison.

        Args:
            results: Benchmark results

        Returns:
            DataFrame with comparison data
        """
        data = []
        for model in results.get("models", []):
            row = {"model": model}

            # Quality metrics
            quality = results.get("quality_results", {}).get(model, {}).get("metrics", {})
            row["quality_score"] = quality.get("overall_quality", 0)
            row["bleu"] = quality.get("bleu_score", 0)
            row["rouge_l"] = quality.get("rouge_l", 0)

            # Speed metrics
            speed = results.get("speed_results", {}).get(model, {}).get("metrics", {})
            row["latency_ms"] = speed.get("avg_latency_ms", 0)
            row["tokens_per_sec"] = speed.get("avg_tokens_per_second", 0)

            # Cost metrics
            cost = results.get("cost_results", {}).get(model, {}).get("metrics", {})
            row["total_cost"] = cost.get("total_cost", 0)
            row["cost_per_request"] = cost.get("avg_cost_per_request", 0)

            data.append(row)

        return pd.DataFrame(data)
