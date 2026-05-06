"""Visualization package — plotly charts for the statistics page, V2.1."""

from .bar_charts import mean_trs_by_model, mean_tr_by_model, mean_bias_by_model_grouped, trs_by_prompt_heatmap
from .radar_charts import bias_radar_per_model, bias_radar_by_alignment
from .heatmaps import bias_heatmap, model_comparison_heatmap
from .line_charts import dimension_profile_lines, trs_dimension_profile_lines
from .distribution_charts import (
    trs_box_plot, dimension_violin_plot, density_histogram,
    alignment_pie_chart, category_pie_chart,
)

__all__ = [
    "mean_trs_by_model",
    "mean_tr_by_model",  # backward compat
    "mean_bias_by_model_grouped",
    "trs_by_prompt_heatmap",
    "bias_radar_per_model",
    "bias_radar_by_alignment",
    "bias_heatmap",
    "model_comparison_heatmap",
    "dimension_profile_lines",
    "trs_dimension_profile_lines",
    "trs_box_plot",
    "dimension_violin_plot",
    "density_histogram",
    "alignment_pie_chart",
    "category_pie_chart",
]
