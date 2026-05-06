"""Bar chart visualizations for TR and bias scores — warm palette."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

# Warm palette aligned with app design tokens
PRIMARY = "hsl(22, 68%, 48%)"
BLUE = "hsl(218, 45%, 48%)"
RED = "hsl(6, 42%, 48%)"
GREEN = "hsl(155, 32%, 40%)"
AMBER = "hsl(35, 55%, 46%)"
DIM_COLORS = [RED, BLUE, GREEN, AMBER]

CHART_FONT = dict(family="system-ui, -apple-system, sans-serif", color="hsl(22, 10%, 20%)")
CHART_TITLE_FONT = dict(family="system-ui, -apple-system, sans-serif", size=15, color="hsl(22, 10%, 15%)")

LAYOUT_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=CHART_FONT,
)


def mean_tr_by_model(tr_stats_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: mean TR score per model with error bars (SD)."""
    df = tr_stats_df.sort_values("Mean")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Mean"],
            y=df["Model"],
            orientation="h",
            error_x=dict(type="data", array=df["Std"]),
            text=[f"{m:.2f} (n={n})" for m, n in zip(df["Mean"], df["N"])],
            textposition="outside",
            marker_color=PRIMARY,
            hovertemplate="%{y}: %{x:.2f} ± %{error_x.array:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Mean TR Score by Model (Higher = More Transparent)", font=CHART_TITLE_FONT),
        xaxis=dict(title="TR Score (1–5)", range=[0.3, 5.7], dtick=1, gridcolor="hsl(38, 15%, 88%)"),
        yaxis=dict(title=None, gridcolor="hsl(38, 15%, 88%)"),
        height=max(250, len(df) * 50),
        margin=dict(l=10, r=80, t=40, b=10),
    )
    return fig


def mean_bias_by_model_grouped(bias_stats_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: D1-D4 mean scores per model."""
    dimensions = [
        ("D1: Responsibility", "D1_mean"),
        ("D2: Coverage", "D2_mean"),
        ("D3: Rule Citation", "D3_mean"),
        ("D4: Framing", "D4_mean"),
    ]

    fig = go.Figure()
    for (label, col), color in zip(dimensions, DIM_COLORS):
        fig.add_trace(
            go.Bar(
                name=label,
                x=bias_stats_df["Model"],
                y=bias_stats_df[col],
                marker_color=color,
                text=[f"{v:.2f}" for v in bias_stats_df[col]],
                textposition="auto",
            )
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="Bias Dimension Scores by Model (1=Pro-West, 3=Neutral, 5=Pro-China)",
            font=CHART_TITLE_FONT,
        ),
        xaxis=dict(title=None, gridcolor="hsl(38, 15%, 88%)"),
        yaxis=dict(title="Mean Score (1–5)", range=[0.3, 5.7], dtick=1, gridcolor="hsl(38, 15%, 88%)"),
        barmode="group",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    fig.add_hline(y=3, line_dash="dash", line_color="hsl(22, 10%, 70%)", opacity=0.6)
    return fig


def tr_by_prompt_heatmap(analyses_df: pd.DataFrame) -> go.Figure:
    """Heatmap: TR scores — rows=models, columns=prompts."""
    pivot = analyses_df.pivot_table(
        index="model_name", columns="prompt_name", values="tr_score", aggfunc="mean"
    )
    if pivot.empty:
        return go.Figure()

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, "hsl(6, 45%, 88%)"],
                [0.25, "hsl(35, 50%, 85%)"],
                [0.5, "hsl(38, 50%, 82%)"],
                [0.75, "hsl(22, 55%, 65%)"],
                [1.0, "hsl(22, 68%, 48%)"],
            ],
            zmin=1,
            zmax=5,
            text=[[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 13},
            colorbar=dict(
                title="TR Score",
                tickvals=[1, 2, 3, 4, 5],
                outlinewidth=0,
            ),
            hovertemplate="Model: %{y}<br>Prompt: %{x}<br>TR: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="TR Score Heatmap (Model × Question)", font=CHART_TITLE_FONT),
        height=max(250, len(pivot) * 50 + 100),
        xaxis=dict(title=None),
        yaxis=dict(title=None),
    )
    return fig
