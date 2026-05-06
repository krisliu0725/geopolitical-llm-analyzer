"""Radar chart visualization — V2.0 with D5 dimension."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

RADAR_COLORS = [
    "hsl(218, 45%, 48%)",  # muted blue
    "hsl(6, 42%, 48%)",     # muted red
    "hsl(155, 32%, 40%)",   # muted green
    "hsl(35, 55%, 46%)",    # warm amber
    "hsl(270, 25%, 45%)",   # muted purple
    "hsl(185, 30%, 40%)",   # muted teal
    "hsl(22, 68%, 48%)",    # primary warm brown
    "hsl(45, 25%, 40%)",    # olive
]

ALIGNMENT_COLORS = {
    "US/Western": "hsl(218, 50%, 42%)",
    "China/Non-Western": "hsl(6, 48%, 44%)",
    "Other/Unknown": "hsl(22, 8%, 50%)",
}

CHART_FONT = dict(family="system-ui, -apple-system, sans-serif", color="hsl(22, 10%, 20%)")
CHART_TITLE_FONT = dict(family="system-ui, -apple-system, sans-serif", size=15, color="hsl(22, 10%, 15%)")

LAYOUT_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=CHART_FONT,
)

CATEGORIES = [
    "D1: Blame",
    "D2: Coverage",
    "D3: Rules",
    "D4: Framing",
    "D5: Lexical",
]

VALUE_COLS = ["D1_mean", "D2_mean", "D3_mean", "D4_mean", "D5_mean"]


def bias_radar_per_model(profile_df: pd.DataFrame) -> go.Figure:
    """Overlaid radar chart showing 5D bias profiles for each model."""
    fig = go.Figure()
    for i, (_, row) in enumerate(profile_df.iterrows()):
        values = [row[c] for c in VALUE_COLS if c in row.index]
        if len(values) != 5:
            continue
        values_closed = values + [values[0]]
        cats_closed = CATEGORIES + [CATEGORIES[0]]
        color = RADAR_COLORS[i % len(RADAR_COLORS)]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                name=row["Model"],
                fill="toself",
                opacity=0.15,
                line=dict(color=color, width=2),
                hovertemplate="%{theta}: %{r:.2f}<extra>%{full_name}</extra>",
            )
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Bias Dimension Profiles by Model (5D)", font=CHART_TITLE_FONT),
        polar=dict(
            radialaxis=dict(
                range=[0, 5],
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1<br>Pro-West", "2", "3<br>Neutral", "4", "5<br>Pro-China"],
                gridcolor="hsl(38, 15%, 88%)",
            ),
            angularaxis=dict(gridcolor="hsl(38, 15%, 88%)"),
        ),
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def bias_radar_by_alignment(profile_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing average 5D bias profiles by alignment group."""
    alignments = profile_df["Alignment"].unique() if "Alignment" in profile_df.columns else []

    fig = go.Figure()
    for align in alignments:
        subset = profile_df[profile_df["Alignment"] == align]
        if subset.empty:
            continue
        values = [subset[c].mean() for c in VALUE_COLS if c in subset.columns]
        if len(values) != 5:
            continue
        values_closed = values + [values[0]]
        cats_closed = CATEGORIES + [CATEGORIES[0]]
        color = ALIGNMENT_COLORS.get(align, "hsl(22, 8%, 50%)")

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                name=align,
                fill="toself",
                opacity=0.15,
                line=dict(color=color, width=2.5),
                hovertemplate="%{theta}: %{r:.2f}<extra>%{full_name}</extra>",
            )
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Bias Profile by Model Alignment (5D)", font=CHART_TITLE_FONT),
        polar=dict(
            radialaxis=dict(
                range=[0, 5],
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1<br>Pro-West", "2", "3<br>Neutral", "4", "5<br>Pro-China"],
                gridcolor="hsl(38, 15%, 88%)",
            ),
            angularaxis=dict(gridcolor="hsl(38, 15%, 88%)"),
        ),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig
