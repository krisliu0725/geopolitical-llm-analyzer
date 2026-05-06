"""Visualization package — plotly charts for the statistics page, V2.0."""

from .bar_charts import mean_trs_by_model, mean_tr_by_model, mean_bias_by_model_grouped, trs_by_prompt_heatmap
from .radar_charts import bias_radar_per_model, bias_radar_by_alignment
from .heatmaps import bias_heatmap, model_comparison_heatmap

__all__ = [
    "mean_trs_by_model",
    "mean_tr_by_model",  # backward compat
    "mean_bias_by_model_grouped",
    "trs_by_prompt_heatmap",
    "bias_radar_per_model",
    "bias_radar_by_alignment",
    "bias_heatmap",
    "model_comparison_heatmap",
]
