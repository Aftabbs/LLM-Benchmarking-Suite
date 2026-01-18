"""
LLM Benchmarking Suite - Streamlit Application.

A comprehensive UI for benchmarking and comparing LLM models.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="LLM Benchmarking Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import project modules
from config import get_config, DATASETS_DIR, RESULTS_DIR
from src.model_client import ModelClient, MODEL_PRICING
from src.evaluator import BenchmarkEvaluator
from src.benchmarks.quality_benchmark import QualityBenchmark
from src.benchmarks.speed_benchmark import SpeedBenchmark
from src.benchmarks.cost_benchmark import CostBenchmark
from src.benchmarks.task_benchmark import TaskBenchmark, TaskType
from src.report_generator import ReportGenerator
from src.web_search import WebSearchClient, WebEnrichedBenchmark
from src.utils import load_dataset, format_duration, format_cost


# Session state initialization
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "running" not in st.session_state:
    st.session_state.running = False


def load_sample_datasets():
    """Load sample datasets from file."""
    dataset_path = DATASETS_DIR / "sample_datasets.json"
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_available_models():
    """Get list of available models."""
    return list(MODEL_PRICING.keys())


def create_quality_chart(results: dict) -> go.Figure:
    """Create quality comparison chart."""
    if not results:
        return None

    models = list(results.keys())
    metrics = ["bleu_score", "rouge_l", "bert_score_f1", "overall_quality"]
    metric_labels = ["BLEU", "ROUGE-L", "BERTScore F1", "Overall"]

    data = []
    for i, metric in enumerate(metrics):
        values = []
        for model in models:
            model_data = results[model].get("metrics", {})
            value = model_data.get(metric, 0)
            if metric == "bleu_score":
                value = value / 100
            values.append(value)
        data.append(go.Bar(name=metric_labels[i], x=models, y=values))

    fig = go.Figure(data=data)
    fig.update_layout(
        title="Quality Metrics Comparison",
        barmode="group",
        yaxis_title="Score",
        xaxis_title="Model",
        height=400,
    )
    return fig


def create_speed_chart(results: dict) -> go.Figure:
    """Create speed comparison chart."""
    if not results:
        return None

    models = list(results.keys())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Average Latency (ms)", "Tokens per Second")
    )

    latencies = [
        results[m].get("metrics", {}).get("avg_latency_ms", 0)
        for m in models
    ]
    throughputs = [
        results[m].get("metrics", {}).get("avg_tokens_per_second", 0)
        for m in models
    ]

    fig.add_trace(
        go.Bar(x=models, y=latencies, name="Latency", marker_color="#FF6B6B"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=models, y=throughputs, name="Throughput", marker_color="#4ECDC4"),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False)
    return fig


def create_cost_chart(results: dict) -> go.Figure:
    """Create cost comparison chart."""
    if not results:
        return None

    models = list(results.keys())
    costs = [
        results[m].get("metrics", {}).get("total_cost", 0)
        for m in models
    ]

    fig = go.Figure(data=[
        go.Bar(x=models, y=costs, marker_color="#95E1D3")
    ])

    fig.update_layout(
        title="Total Cost Comparison",
        yaxis_title="Cost ($)",
        xaxis_title="Model",
        height=400,
    )
    return fig


def create_radar_chart(results: dict) -> go.Figure:
    """Create radar chart for overall comparison."""
    models = results.get("models", [])
    if not models:
        return None

    categories = ["Quality", "Speed", "Cost", "Reliability"]

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, model in enumerate(models):
        values = []

        quality_data = results.get("quality_results", {}).get(model, {})
        quality = quality_data.get("metrics", {}).get("overall_quality", 0)
        values.append(quality)

        speed_data = results.get("speed_results", {}).get(model, {})
        latency = speed_data.get("metrics", {}).get("avg_latency_ms", 10000)
        speed_score = max(0, 1 - (latency / 10000))
        values.append(speed_score)

        cost_data = results.get("cost_results", {}).get(model, {})
        cost = cost_data.get("metrics", {}).get("total_cost", 1)
        cost_score = max(0, 1 - (cost * 10))
        values.append(cost_score)

        error_rate = speed_data.get("metrics", {}).get("error_rate", 0)
        reliability = 1 - error_rate
        values.append(reliability)

        values.append(values[0])
        categories_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            name=model,
            fill="toself",
            line_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title="Overall Model Comparison",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500,
    )
    return fig


def main():
    """Main application."""

    # Header
    st.title("📊 LLM Benchmarking Suite")
    st.markdown("*Compare LLM models across quality, speed, cost, and task performance*")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Keys
        st.subheader("API Keys")
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Enter your Groq API key"
        )
        serper_key = st.text_input(
            "Serper API Key (Optional)",
            type="password",
            value=os.getenv("SERPER_API_KEY", ""),
            help="For web-enriched benchmarks"
        )

        st.divider()

        # Model Selection
        st.subheader("Model Selection")
        available_models = get_available_models()
        selected_models = st.multiselect(
            "Select Models to Benchmark",
            available_models,
            default=available_models[:2] if len(available_models) >= 2 else available_models,
        )

        st.divider()

        # Benchmark Type
        st.subheader("Benchmark Types")
        run_quality = st.checkbox("Quality Benchmark", value=True)
        run_speed = st.checkbox("Speed Benchmark", value=True)
        run_cost = st.checkbox("Cost Benchmark", value=True)
        run_task = st.checkbox("Task-Specific Benchmark", value=False)

        if run_task:
            task_type = st.selectbox(
                "Task Type",
                [t.value for t in TaskType],
                index=0,
            )
        else:
            task_type = None

        st.divider()

        # Advanced Settings
        with st.expander("Advanced Settings"):
            num_iterations = st.slider("Number of Iterations", 1, 20, 5)
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
            max_tokens = st.slider("Max Tokens", 256, 4096, 2048)

    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Run Benchmark",
        "📈 Results",
        "📋 Datasets",
        "🔍 Web Search",
        "📄 Reports"
    ])

    # Tab 1: Run Benchmark
    with tab1:
        st.header("Run Benchmark")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Dataset selection
            st.subheader("Select Dataset")
            datasets = load_sample_datasets()
            dataset_names = list(datasets.keys())

            selected_dataset = st.selectbox(
                "Choose a dataset",
                dataset_names,
                index=0 if dataset_names else None,
            )

            # Custom prompts option
            use_custom = st.checkbox("Use custom prompts instead")

            if use_custom:
                custom_prompts = st.text_area(
                    "Enter prompts (one per line)",
                    height=200,
                    placeholder="Enter your test prompts here...\nOne prompt per line."
                )
            else:
                if selected_dataset and selected_dataset in datasets:
                    dataset_info = datasets[selected_dataset]
                    st.info(f"**{dataset_info.get('name', selected_dataset)}**: {dataset_info.get('description', '')}")
                    st.write(f"Number of samples: {len(dataset_info.get('data', []))}")

        with col2:
            # System prompt
            st.subheader("System Prompt")
            system_prompt = st.text_area(
                "Optional system prompt",
                height=150,
                placeholder="Enter an optional system prompt..."
            )

        # Run button
        st.divider()

        if st.button("🚀 Run Benchmark", type="primary", disabled=st.session_state.running):
            if not groq_key:
                st.error("Please enter your Groq API key")
            elif not selected_models:
                st.error("Please select at least one model")
            else:
                st.session_state.running = True

                # Prepare benchmark types
                benchmark_types = []
                if run_quality:
                    benchmark_types.append("quality")
                if run_speed:
                    benchmark_types.append("speed")
                if run_cost:
                    benchmark_types.append("cost")
                if run_task:
                    benchmark_types.append("task")

                if not benchmark_types:
                    benchmark_types = ["comprehensive"]

                # Prepare dataset
                if use_custom and custom_prompts:
                    prompts = [p.strip() for p in custom_prompts.split("\n") if p.strip()]
                    dataset = [{"prompt": p} for p in prompts]
                elif selected_dataset and selected_dataset in datasets:
                    dataset = datasets[selected_dataset].get("data", [])
                else:
                    dataset = []

                if not dataset:
                    st.error("No dataset available")
                    st.session_state.running = False
                else:
                    # Run benchmark with progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        evaluator = BenchmarkEvaluator(api_key=groq_key)

                        status_text.text("Running benchmarks...")
                        progress_bar.progress(25)

                        results = evaluator.run(
                            models=selected_models,
                            dataset=dataset,
                            benchmark_types=benchmark_types,
                            task_type=task_type,
                            system_prompt=system_prompt if system_prompt else None,
                        )

                        progress_bar.progress(75)

                        # Add comparison
                        comparison = evaluator.compare_models(results)
                        results.update(comparison)

                        progress_bar.progress(100)
                        status_text.text("Benchmark complete!")

                        st.session_state.benchmark_results = results
                        st.success("✅ Benchmark completed successfully!")

                    except Exception as e:
                        st.error(f"Error running benchmark: {str(e)}")

                    finally:
                        st.session_state.running = False
                        progress_bar.empty()
                        status_text.empty()

    # Tab 2: Results
    with tab2:
        st.header("Benchmark Results")

        if st.session_state.benchmark_results:
            results = st.session_state.benchmark_results

            # Summary cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Models Tested",
                    len(results.get("models", []))
                )

            with col2:
                if results.get("rankings", {}).get("overall"):
                    st.metric(
                        "Best Overall",
                        results["rankings"]["overall"][0]
                    )

            with col3:
                if results.get("rankings", {}).get("speed"):
                    st.metric(
                        "Fastest",
                        results["rankings"]["speed"][0]
                    )

            with col4:
                if results.get("rankings", {}).get("cost"):
                    st.metric(
                        "Most Cost-Effective",
                        results["rankings"]["cost"][0]
                    )

            st.divider()

            # Charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                if results.get("quality_results"):
                    fig = create_quality_chart(results["quality_results"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                if results.get("cost_results"):
                    fig = create_cost_chart(results["cost_results"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            with chart_col2:
                if results.get("speed_results"):
                    fig = create_speed_chart(results["speed_results"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                # Radar chart
                fig = create_radar_chart(results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Detailed tables
            st.subheader("Detailed Results")

            detail_tab1, detail_tab2, detail_tab3 = st.tabs([
                "Quality", "Speed", "Cost"
            ])

            with detail_tab1:
                if results.get("quality_results"):
                    df_data = []
                    for model, data in results["quality_results"].items():
                        metrics = data.get("metrics", {})
                        df_data.append({
                            "Model": model,
                            "BLEU": f"{metrics.get('bleu_score', 0):.2f}",
                            "ROUGE-1": f"{metrics.get('rouge_1', 0):.3f}",
                            "ROUGE-L": f"{metrics.get('rouge_l', 0):.3f}",
                            "BERTScore F1": f"{metrics.get('bert_score_f1', 0):.3f}",
                            "Overall": f"{metrics.get('overall_quality', 0):.3f}",
                        })
                    st.dataframe(pd.DataFrame(df_data), use_container_width=True)

            with detail_tab2:
                if results.get("speed_results"):
                    df_data = []
                    for model, data in results["speed_results"].items():
                        metrics = data.get("metrics", {})
                        df_data.append({
                            "Model": model,
                            "Avg Latency (ms)": f"{metrics.get('avg_latency_ms', 0):.2f}",
                            "P95 Latency (ms)": f"{metrics.get('p95_latency_ms', 0):.2f}",
                            "Tokens/sec": f"{metrics.get('avg_tokens_per_second', 0):.2f}",
                            "Error Rate": f"{metrics.get('error_rate', 0):.2%}",
                        })
                    st.dataframe(pd.DataFrame(df_data), use_container_width=True)

            with detail_tab3:
                if results.get("cost_results"):
                    df_data = []
                    for model, data in results["cost_results"].items():
                        metrics = data.get("metrics", {})
                        df_data.append({
                            "Model": model,
                            "Total Cost": f"${metrics.get('total_cost', 0):.6f}",
                            "Cost/Request": f"${metrics.get('avg_cost_per_request', 0):.6f}",
                            "Cost/1K Tokens": f"${metrics.get('cost_per_1k_total_tokens', 0):.6f}",
                            "Total Tokens": metrics.get('total_tokens', 0),
                        })
                    st.dataframe(pd.DataFrame(df_data), use_container_width=True)

            # Recommendations
            if results.get("recommendations"):
                st.divider()
                st.subheader("📌 Recommendations")
                for rec in results["recommendations"]:
                    st.info(rec)

        else:
            st.info("No benchmark results yet. Run a benchmark to see results here.")

    # Tab 3: Datasets
    with tab3:
        st.header("Benchmark Datasets")

        datasets = load_sample_datasets()

        for name, dataset in datasets.items():
            with st.expander(f"📁 {dataset.get('name', name)}"):
                st.write(f"**Description:** {dataset.get('description', 'N/A')}")
                st.write(f"**Task Type:** {dataset.get('task_type', 'general')}")
                st.write(f"**Samples:** {len(dataset.get('data', []))}")

                if st.checkbox(f"Show samples for {name}", key=f"show_{name}"):
                    data = dataset.get("data", [])[:5]
                    for i, item in enumerate(data):
                        st.markdown(f"**Sample {i+1}:**")
                        st.json(item)

    # Tab 4: Web Search
    with tab4:
        st.header("Web-Enriched Benchmarks")

        if serper_key:
            search_client = WebSearchClient(api_key=serper_key)
            web_benchmark = WebEnrichedBenchmark(search_client)

            st.subheader("Create Dynamic Dataset")

            col1, col2 = st.columns(2)

            with col1:
                topics = st.text_area(
                    "Enter topics (one per line)",
                    placeholder="artificial intelligence\nclimate change\nquantum computing",
                    height=150
                )

            with col2:
                dataset_type = st.selectbox(
                    "Dataset Type",
                    ["Fact Check", "News Q&A"]
                )

            if st.button("Generate Dataset"):
                topic_list = [t.strip() for t in topics.split("\n") if t.strip()]

                if topic_list:
                    with st.spinner("Searching and creating dataset..."):
                        if dataset_type == "Fact Check":
                            dataset = web_benchmark.create_fact_check_dataset(topic_list)
                        else:
                            dataset = web_benchmark.create_news_qa_dataset(topic_list)

                        if dataset:
                            st.success(f"Created {len(dataset)} test cases")

                            st.subheader("Generated Dataset")
                            for i, item in enumerate(dataset[:5]):
                                with st.expander(f"Test Case {i+1}"):
                                    st.write(f"**Topic:** {item.get('topic', 'N/A')}")
                                    st.write(f"**Prompt:** {item.get('prompt', '')[:500]}...")
                        else:
                            st.warning("No results found for the given topics")
                else:
                    st.warning("Please enter at least one topic")

            # Quick search test
            st.divider()
            st.subheader("Quick Search Test")

            query = st.text_input("Search Query")
            if st.button("Search"):
                if query:
                    response = search_client.search(query)
                    if response.results:
                        for result in response.results[:5]:
                            st.markdown(f"**[{result.title}]({result.link})**")
                            st.write(result.snippet)
                            st.divider()
                    else:
                        st.info("No results found")

        else:
            st.warning("Enter your Serper API key in the sidebar to use web search features")

    # Tab 5: Reports
    with tab5:
        st.header("Generate Reports")

        if st.session_state.benchmark_results:
            results = st.session_state.benchmark_results

            col1, col2 = st.columns(2)

            with col1:
                report_format = st.selectbox(
                    "Report Format",
                    ["HTML", "Markdown", "JSON"]
                )

            with col2:
                if st.button("Generate Report"):
                    generator = ReportGenerator()

                    format_map = {
                        "HTML": "html",
                        "Markdown": "markdown",
                        "JSON": "json"
                    }

                    report_content = generator.generate(
                        results,
                        format=format_map[report_format]
                    )

                    if report_format == "HTML":
                        st.components.v1.html(report_content, height=600, scrolling=True)
                    elif report_format == "Markdown":
                        st.markdown(report_content)
                    else:
                        st.json(json.loads(report_content))

                    # Download button
                    st.download_button(
                        label=f"Download {report_format} Report",
                        data=report_content,
                        file_name=f"benchmark_report.{format_map[report_format]}",
                        mime="text/html" if report_format == "HTML" else "text/plain"
                    )

            # Save results
            st.divider()
            st.subheader("Save Results")

            if st.button("Save Results to File"):
                output_path = RESULTS_DIR / f"results_{results.get('run_id', 'unknown')}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, default=str)

                st.success(f"Results saved to: {output_path}")

        else:
            st.info("Run a benchmark first to generate reports")


if __name__ == "__main__":
    main()
