"""Heatmap visualizations — V2.0 with D5 + TRS."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

CHART_FONT = dict(family="system-ui, -apple-system, sans-serif", color="hsl(22, 10%, 20%)")
CHART_TITLE_FONT = dict(family="system-ui, -apple-system, sans-serif", size=15, color="hsl(22, 10%, 15%)")

LAYOUT_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=CHART_FONT,
)

WARM_DIVERGING = [
    [0.0, "hsl(218, 45%, 70%)"],   # blue-ish (pro-west)
    [0.25, "hsl(218, 30%, 82%)"],
    [0.5, "hsl(38, 20%, 92%)"],     # cream (neutral)
    [0.75, "hsl(22, 40%, 75%)"],
    [1.0, "hsl(22, 68%, 48%)"],     # warm brown (pro-china)
]


def bias_heatmap(profile_df: pd.DataFrame) -> go.Figure:
    """Heatmap: rows=models, columns=D1..D5, cell color = mean score."""
    models = profile_df["Model"].tolist()
    dims = ["D1", "D2", "D3", "D4", "D5"]
    value_cols = ["D1_mean", "D2_mean", "D3_mean", "D4_mean", "D5_mean"]
    z = profile_df[value_cols].values
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=dims,
            y=models,
            colorscale=WARM_DIVERGING,
            zmin=1,
            zmax=5,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13},
            colorbar=dict(
                title="Mean Score",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1<br>Pro-West", "2", "3<br>Neutral", "4", "5<br>Pro-China"],
                outlinewidth=0,
            ),
            hovertemplate="Model: %{y}<br>%{x}: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Bias Score Heatmap (Model × Dimension, 5D)", font=CHART_TITLE_FONT),
        height=max(250, len(models) * 50 + 100),
        xaxis=dict(title=None, side="bottom"),
        yaxis=dict(title=None),
    )
    return fig


def model_comparison_heatmap(analyses_df: pd.DataFrame, dimension: str = "trs") -> go.Figure:
    """Heatmap comparing models across prompts for a given score dimension."""
    pivot = analyses_df.pivot_table(
        index="model_name",
        columns="prompt_name",
        values=dimension,
        aggfunc="mean",
    )
    if pivot.empty:
        return go.Figure()

    dim_labels = {
        "trs": "TRS Score",
        "t1_score": "T1: Completeness",
        "t2_score": "T2: Core Engagement",
        "t3_score": "T3: Specificity",
        "t4_score": "T4: Confrontation",
        "t5_score": "T5: Evasion Penalty",
        "d1_score": "D1: Blame Attribution",
        "d2_score": "D2: Coverage Balance",
        "d3_score": "D3: Normative Framework",
        "d4_score": "D4: Concluding Sentiment",
        "d5_score": "D5: Lexical Framing",
        "gbs": "GBS (Composite Bias)",
    }

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=WARM_DIVERGING,
            zmin=1,
            zmax=5,
            text=[[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(
                title=dim_labels.get(dimension, dimension),
                tickvals=[1, 2, 3, 4, 5],
                outlinewidth=0,
            ),
            hovertemplate="Model: %{y}<br>Prompt: %{x}<br>Score: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"{dim_labels.get(dimension, dimension)} — Model × Question",
            font=CHART_TITLE_FONT,
        ),
        height=max(250, len(pivot) * 50 + 100),
        xaxis=dict(title=None),
        yaxis=dict(title=None),
    )
    return fig
