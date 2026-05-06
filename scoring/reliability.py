"""Inter-rater reliability: ICC, Fleiss' Kappa, agreement percentages."""

import pandas as pd
import numpy as np

try:
    import pingouin as pg

    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

try:
    from statsmodels.stats.inter_rater import fleiss_kappa

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) interpretation of kappa values."""
    if kappa is None:
        return "N/A"
    if kappa < 0:
        return "Poor (< 0)"
    if kappa <= 0.20:
        return "Slight (0–0.20)"
    if kappa <= 0.40:
        return "Fair (0.21–0.40)"
    if kappa <= 0.60:
        return "Moderate (0.41–0.60)"
    if kappa <= 0.80:
        return "Substantial (0.61–0.80)"
    return "Almost Perfect (0.81–1.00)"


def compute_tr_icc(tr_scores_df: pd.DataFrame) -> dict:
    """Compute ICC(A,1) for TR scores via pingouin.

    Args:
        tr_scores_df: DataFrame with columns [response_id, coder_id, score]

    Returns:
        Dict with keys: icc, ci_95_lower, ci_95_upper, interpretation, n_responses, n_coders, error
    """
    if not HAS_PINGOUIN:
        return {"error": "pingouin not installed", "icc": None}

    n_responses = tr_scores_df["response_id"].nunique()
    n_coders = tr_scores_df["coder_id"].nunique()

    if n_responses < 3 or n_coders < 2:
        return {
            "icc": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
            "interpretation": "N/A",
            "n_responses": n_responses,
            "n_coders": n_coders,
            "error": f"Need ≥3 responses and ≥2 coders (got {n_responses}, {n_coders})",
        }

    try:
        result = pg.intraclass_corr(
            data=tr_scores_df,
            targets="response_id",
            raters="coder_id",
            ratings="score",
        )
        # ICC2 = two-way random effects, single rater (absolute agreement)
        icc_row = result[result["Type"] == "ICC2"]
        if icc_row.empty:
            return {"error": "ICC2 computation returned no results", "icc": None}

        row = icc_row.iloc[0]
        ci = row["CI95%"]
        icc_val = row["ICC"]
        return {
            "icc": round(icc_val, 4),
            "ci_95_lower": round(ci[0], 4) if hasattr(ci, "__getitem__") else None,
            "ci_95_upper": round(ci[1], 4) if hasattr(ci, "__getitem__") else None,
            "interpretation": interpret_kappa(icc_val),
            "n_responses": n_responses,
            "n_coders": n_coders,
        }
    except Exception as e:
        return {"error": str(e), "icc": None}


def compute_fleiss_kappa_all_dimensions(bias_scores_df: pd.DataFrame) -> dict:
    """Compute Fleiss' Kappa for each bias dimension D1-D4.

    Args:
        bias_scores_df: DataFrame with columns
            [response_id, coder_id, d1_responsibility_attribution,
             d2_coverage_balance, d3_rule_citation, d4_final_framing]

    Returns:
        Dict mapping dimension name to {kappa, interpretation, error}
    """
    if not HAS_STATSMODELS:
        return {
            "error": "statsmodels not installed",
        }

    dimensions = {
        "d1_responsibility_attribution": "d1_responsibility_attribution",
        "d2_coverage_balance": "d2_coverage_balance",
        "d3_rule_citation": "d3_rule_citation",
        "d4_final_framing": "d4_final_framing",
    }

    results = {}
    for dim_name, col in dimensions.items():
        try:
            pivot = bias_scores_df.pivot_table(
                index="response_id",
                columns=col,
                aggfunc="size",
                fill_value=0,
            )
            for cat in range(1, 6):
                if cat not in pivot.columns:
                    pivot[cat] = 0
            pivot = pivot[sorted(pivot.columns)]
            matrix = pivot.values.astype(int)

            if matrix.shape[0] < 2:
                results[dim_name] = {
                    "kappa": None,
                    "interpretation": "N/A",
                    "error": "Need ≥2 responses",
                }
                continue

            k_val = fleiss_kappa(matrix, method="fleiss")
            results[dim_name] = {
                "kappa": round(k_val, 4),
                "interpretation": interpret_kappa(k_val),
            }
        except Exception as e:
            results[dim_name] = {
                "kappa": None,
                "interpretation": "N/A",
                "error": str(e),
            }

    return results


def compute_agreement_metrics(tr_scores_df: pd.DataFrame) -> dict:
    """Compute agreement percentages across coders.

    Args:
        tr_scores_df: DataFrame with columns [response_id, coder_id, score]

    Returns:
        Dict with exact_agreement_pct, majority_agreement_pct, within_one_pct
    """
    pivot = tr_scores_df.pivot(index="response_id", columns="coder_id", values="score")
    n_total = len(pivot)

    if n_total == 0:
        return {
            "exact_agreement_pct": 0,
            "majority_agreement_pct": 0,
            "within_one_pct": 0,
        }

    # Exact: all values in row equal
    exact = (pivot.nunique(axis=1) == 1).sum() / n_total * 100

    # Within 1: max - min <= 1
    within_one = ((pivot.max(axis=1) - pivot.min(axis=1)) <= 1).sum() / n_total * 100

    # Majority: there exists a value that appears at least 2 times
    majority = 0
    for _, row in pivot.iterrows():
        counts = row.value_counts()
        if counts.max() >= 2:
            majority += 1
    majority_pct = majority / n_total * 100

    return {
        "exact_agreement_pct": round(exact, 2),
        "majority_agreement_pct": round(majority_pct, 2),
        "within_one_pct": round(within_one, 2),
    }
