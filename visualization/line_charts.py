"""Line chart visualizations — D1-D5 and T1-T5 dimension profiles."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from .bar_charts import DIM_COLORS, PRIMARY, CHART_FONT, CHART_TITLE_FONT, LAYOUT_BASE


def dimension_profile_lines(analyses_df: pd.DataFrame) -> go.Figure:
    """Multi-line chart: D1-D5 mean scores per model with neutral reference."""
    dims = ["d1_score", "d2_score", "d3_score", "d4_score", "d5_score"]
    dim_labels = ["D1: Blame", "D2: Coverage", "D3: Rules", "D4: Framing", "D5: Lexical"]
    models = sorted(analyses_df["model_name"].unique())

    fig = go.Figure()
    for i, model in enumerate(models):
        subset = analyses_df[analyses_df["model_name"] == model]
        means = [subset[d].mean() for d in dims]
        color = DIM_COLORS[i % len(DIM_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=dim_labels,
                y=means,
                mode="lines+markers",
                name=model,
                line=dict(color=color, width=2.2),
                marker=dict(size=7, color=color),
                hovertemplate="%{x}: %{y:.2f}<extra>%{full_name}</extra>",
            )
        )

    fig.add_hline(y=3, line_dash="dash", line_color="hsl(22,10%,70%)", opacity=0.5,
                  annotation_text="Neutral (3.00)", annotation_position="bottom right")

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="GBS Dimension Profile by Model (D1–D5)", font=CHART_TITLE_FONT),
        xaxis=dict(title=None, gridcolor="hsl(38,15%,88%)"),
        yaxis=dict(title="Mean Score (1–5)", range=[0.3, 5.7], dtick=1, gridcolor="hsl(38,15%,88%)"),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=11)),
    )
    return fig


def trs_dimension_profile_lines(analyses_df: pd.DataFrame) -> go.Figure:
    """Multi-line chart: T1-T5 mean scores per model."""
    dims = ["t1_score", "t2_score", "t3_score", "t4_score", "t5_score"]
    dim_labels = ["T1: Comp.", "T2: Engage", "T3: Spec.", "T4: Confr.", "T5: Evasion"]

    models = sorted(analyses_df["model_name"].unique())

    fig = go.Figure()
    for i, model in enumerate(models):
        subset = analyses_df[analyses_df["model_name"] == model]
        means = [subset[d].mean() for d in dims]
        color = DIM_COLORS[i % len(DIM_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=dim_labels,
                y=means,
                mode="lines+markers",
                name=model,
                line=dict(color=color, width=2.2),
                marker=dict(size=7, color=color),
                hovertemplate="%{x}: %{y:.2f}<extra>%{full_name}</extra>",
            )
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="TRS Dimension Profile by Model (T1–T5)", font=CHART_TITLE_FONT),
        xaxis=dict(title=None, gridcolor="hsl(38,15%,88%)"),
        yaxis=dict(title="Mean Score (1–5)", range=[0.3, 5.7], dtick=1, gridcolor="hsl(38,15%,88%)"),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=11)),
    )
    return fig
