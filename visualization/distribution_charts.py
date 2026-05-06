"""Distribution charts — box, violin, density histogram, pie donuts."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from .bar_charts import DIM_COLORS, PRIMARY, CHART_FONT, CHART_TITLE_FONT, LAYOUT_BASE

PIE_COLORS = [
    "hsl(218, 45%, 55%)",
    "hsl(6, 42%, 52%)",
    "hsl(155, 32%, 42%)",
    "hsl(35, 55%, 50%)",
    "hsl(270, 25%, 48%)",
    "hsl(185, 30%, 44%)",
    "hsl(22, 68%, 48%)",
    "hsl(45, 25%, 45%)",
]


def trs_box_plot(analyses_df: pd.DataFrame) -> go.Figure:
    """Box plot: TRS distribution per model."""
    models = sorted(analyses_df["model_name"].unique())
    fig = go.Figure()
    for i, model in enumerate(models):
        subset = analyses_df[analyses_df["model_name"] == model]
        color = DIM_COLORS[i % len(DIM_COLORS)]
        fig.add_trace(
            go.Box(
                y=subset["trs"],
                name=model,
                marker_color=color,
                boxmean="sd",
                hovertemplate="%{y:.2f}<extra>%{full_name}</extra>",
            )
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="TRS Distribution by Model", font=CHART_TITLE_FONT),
        yaxis=dict(title="TRS (1–5)", range=[0.3, 5.7], dtick=1, gridcolor="hsl(38,15%,88%)"),
        xaxis=dict(title=None, gridcolor="hsl(38,15%,88%)"),
        height=450,
        showlegend=False,
    )
    return fig


def dimension_violin_plot(analyses_df: pd.DataFrame, dimension: str = "gbs") -> go.Figure:
    """Violin plot for a chosen dimension, per model."""
    dim_labels = {
        "gbs": "GBS", "trs": "TRS",
        "d1_score": "D1: Blame", "d2_score": "D2: Coverage",
        "d3_score": "D3: Rules", "d4_score": "D4: Framing", "d5_score": "D5: Lexical",
    }
    label = dim_labels.get(dimension, dimension)

    models = sorted(analyses_df["model_name"].unique())
    fig = go.Figure()
    for i, model in enumerate(models):
        subset = analyses_df[analyses_df["model_name"] == model]
        color = DIM_COLORS[i % len(DIM_COLORS)]
        fig.add_trace(
            go.Violin(
                y=subset[dimension],
                name=model,
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                marker_color=color,
                line_color=color,
                opacity=0.7,
                hovertemplate="%{y:.2f}<extra>%{full_name}</extra>",
            )
        )

    fig.add_hline(y=3, line_dash="dash", line_color="hsl(22,10%,70%)", opacity=0.5)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"{label} Distribution by Model", font=CHART_TITLE_FONT),
        yaxis=dict(title=f"{label} (1–5)", range=[0.3, 5.7], dtick=1, gridcolor="hsl(38,15%,88%)"),
        xaxis=dict(title=None, gridcolor="hsl(38,15%,88%)"),
        height=450,
        showlegend=False,
    )
    return fig


def density_histogram(analyses_df: pd.DataFrame, dimension: str = "trs") -> go.Figure:
    """Overlaid density histograms: one per model for a given dimension."""
    dim_labels = {
        "trs": "TRS", "gbs": "GBS",
        "d1_score": "D1: Blame", "d2_score": "D2: Coverage",
        "d3_score": "D3: Rules", "d4_score": "D4: Framing", "d5_score": "D5: Lexical",
    }
    label = dim_labels.get(dimension, dimension)

    models = sorted(analyses_df["model_name"].unique())
    fig = go.Figure()
    for i, model in enumerate(models):
        subset = analyses_df[analyses_df["model_name"] == model]
        color = DIM_COLORS[i % len(DIM_COLORS)]
        fig.add_trace(
            go.Histogram(
                x=subset[dimension],
                name=model,
                histnorm="probability density",
                nbinsx=25,
                opacity=0.45,
                marker_color=color,
                hovertemplate="%{x:.2f}: density %{y:.3f}<extra>%{full_name}</extra>",
            )
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"{label} Density by Model", font=CHART_TITLE_FONT),
        xaxis=dict(title=f"{label} (1–5)", range=[0.3, 5.7], dtick=1, gridcolor="hsl(38,15%,88%)"),
        yaxis=dict(title="Probability Density", gridcolor="hsl(38,15%,88%)"),
        barmode="overlay",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=11)),
    )
    return fig


def alignment_pie_chart(analyses_df: pd.DataFrame) -> go.Figure:
    """Donut chart: alignment distribution."""
    counts = analyses_df["alignment"].value_counts()
    fig = go.Figure(
        data=go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.35,
            marker_colors=PIE_COLORS[:len(counts)],
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Model Alignment Distribution", font=CHART_TITLE_FONT),
        height=420,
    )
    return fig


def category_pie_chart(analyses_df: pd.DataFrame) -> go.Figure:
    """Donut chart: prompt category distribution."""
    counts = analyses_df["prompt_category"].value_counts()
    fig = go.Figure(
        data=go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.35,
            marker_colors=PIE_COLORS[:len(counts)],
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Prompt Category Distribution", font=CHART_TITLE_FONT),
        height=420,
    )
    return fig
