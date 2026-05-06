"""Descriptive statistics for the V2.0 Analysis data model — TRS + GBS."""

from __future__ import annotations

import pandas as pd


def descriptive_stats_trs(analyses_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model TRS statistics from analyses DataFrame.

    Expects columns: model_name, trs (composite TRS = avg of T1-T5)
    Returns DataFrame: model, n, mean, std, min, max, median
    """
    stats = (
        analyses_df.groupby("model_name")["trs"]
        .agg(n="count", mean="mean", std="std", min="min", max="max", median="median")
        .reset_index()
    )
    stats.columns = ["Model", "N", "Mean", "Std", "Min", "Max", "Median"]
    stats["Mean"] = stats["Mean"].round(2)
    stats["Std"] = stats["Std"].round(2)
    stats["Min"] = stats["Min"].round(2)
    stats["Max"] = stats["Max"].round(2)
    stats["Median"] = stats["Median"].round(2)
    return stats.sort_values("Mean", ascending=False)


def descriptive_stats_bias(analyses_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model GBS dimension statistics — V2.0 with D5.

    Expects columns: model_name, d1_score, d2_score, d3_score, d4_score, d5_score
    Returns DataFrame: model, D1_mean, D1_std, ..., D5_mean, D5_std, gbs, n
    """
    dims = ["d1_score", "d2_score", "d3_score", "d4_score", "d5_score"]
    dim_labels = ["D1", "D2", "D3", "D4", "D5"]

    aggs = {}
    for key, label in zip(dims, dim_labels):
        aggs[f"{label}_mean"] = (key, "mean")
        aggs[f"{label}_std"] = (key, "std")
    aggs["n"] = ("d1_score", "count")

    stats = analyses_df.groupby("model_name").agg(**aggs).reset_index()
    stats.columns = ["Model"] + [f"{l}_{s}" for l in dim_labels for s in ["mean", "std"]] + ["N"]

    for col in stats.columns:
        if col not in ("Model", "N"):
            stats[col] = stats[col].round(2)

    mean_cols = [f"{l}_mean" for l in dim_labels]
    stats["GBS"] = stats[mean_cols].mean(axis=1).round(2)

    return stats.sort_values("GBS", ascending=False)


def per_model_bias_profile(analyses_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model bias profile with D1-D5 means and GBS composite.

    Expects columns: model_name, alignment, d1_score, d2_score, d3_score, d4_score, d5_score
    """
    dims = ["d1_score", "d2_score", "d3_score", "d4_score", "d5_score"]
    labels = [
        "D1: Blame", "D2: Coverage", "D3: Rules",
        "D4: Framing", "D5: Lexical",
    ]

    aggs = {}
    for key, label in zip(dims, labels):
        aggs[label] = (key, "mean")
    aggs["N"] = ("d1_score", "count")
    aggs["Alignment"] = ("alignment", "first")

    profile = analyses_df.groupby("model_name").agg(**aggs).reset_index()
    profile.columns = ["Model"] + [f"D{i+1}_mean" for i in range(5)] + ["N", "Alignment"]

    for c in [f"D{i+1}_mean" for i in range(5)]:
        profile[c] = profile[c].round(2)

    profile["GBS"] = profile[
        ["D1_mean", "D2_mean", "D3_mean", "D4_mean", "D5_mean"]
    ].mean(axis=1).round(2)

    return profile.sort_values("GBS", ascending=False)


def analyses_to_dataframe(analyses: list) -> pd.DataFrame:
    """Convert a list of Analysis ORM objects into a flat DataFrame for stats/viz.

    Joins with prompt and model_config to include metadata. V2.0.
    """
    rows = []
    for a in analyses:
        rows.append({
            "analysis_id": a.id,
            "prompt_id": a.prompt_id,
            "prompt_name": a.prompt.short_name if a.prompt else "N/A",
            "prompt_category": a.prompt.category if a.prompt else "N/A",
            "prompt_text": a.prompt.full_text if a.prompt else "",
            "model_config_id": a.model_config_id,
            "model_name": a.model_config.display_name if a.model_config else "N/A",
            "model_provider": a.model_config.provider if a.model_config else "N/A",
            "alignment": a.model_config.alignment if a.model_config else "Other/Unknown",
            "response_text": a.response_text,
            # TRS sub-dimensions
            "t1_score": a.t1_score,
            "t2_score": a.t2_score,
            "t3_score": a.t3_score,
            "t4_score": a.t4_score,
            "t5_score": a.t5_score,
            "trs": a.trs,
            # GBS sub-dimensions
            "d1_score": a.d1_score,
            "d2_score": a.d2_score,
            "d3_score": a.d3_score,
            "d4_score": a.d4_score,
            "d5_score": a.d5_score,
            "gbs": a.gbs,
            "composite_bias": a.composite_bias,
            "key_phrases": a.key_phrases,
            "notes": a.notes,
            "scorer_model": a.scorer_model,
            "analyzed_at": a.analyzed_at,
        })
    return pd.DataFrame(rows)
