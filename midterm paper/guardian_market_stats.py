"""Only the approved univariate contemporaneous and next-week descriptive OLS."""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from guardian_fgi_stats import correlations

PREDICTORS = {
    "broad_WSI_count": "primary",
    "broad_WSI_continuous": "continuous_robustness",
    "strict_WSI_count": "strict_robustness",
    "strict_WSI_continuous": "strict_continuous_robustness",
    "FGI_weekly_mean": "FGI_context_benchmark",
}
TARGETS = {"contemporaneous": "weekly_log_return", "exploratory_next_week": "next_week_log_return"}


def simple_ols(x, y):
    """OLS with intercept; conventional iid variance and two-sided Student t p-value."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    assert len(x) == len(y) and len(x) >= 3 and np.isfinite(x).all() and np.isfinite(y).all()
    centered_x, centered_y = x - x.mean(), y - y.mean()
    sxx = float(centered_x @ centered_x)
    syy = float(centered_y @ centered_y)
    assert sxx > 0 and syy > 0
    beta = float((centered_x @ centered_y) / sxx)
    alpha = float(y.mean() - beta * x.mean())
    residual = y - (alpha + beta * x)
    sse = float(residual @ residual)
    df = len(x) - 2
    se_beta = float(np.sqrt(sse / df / sxx))
    t_beta = beta / se_beta if se_beta else np.copysign(np.inf, beta)
    p_value = float(2 * student_t.sf(abs(t_beta), df))
    r_squared = float(1 - sse / syy)
    adjusted = float(1 - (1 - r_squared) * (len(x) - 1) / df)
    return {"n_observations": len(x), "alpha": alpha, "beta": beta, "se_beta": se_beta,
        "t_beta": float(t_beta), "p_value_beta_two_sided": p_value, "R_squared": r_squared,
        "adjusted_R_squared": adjusted, "residual_df": df, "residual_sigma": float(np.sqrt(sse / df))}


def analyze_market(merged):
    correlation_rows, ols_rows, merge_rows, pair_rows = [], [], [], []
    for predictor, role in PREDICTORS.items():
        for horizon, target in TARGETS.items():
            frame = merged.loc[merged.is_complete_week]
            a, b = frame[predictor].notna(), frame[target].notna()
            paired = frame.loc[a & b]
            meta = {"predictor": predictor, "role": role, "relationship": horizon, "return_variable": target}
            corr = correlations(paired[predictor], paired[target])
            fitted = simple_ols(paired[predictor], paired[target])
            correlation_rows.append({**meta, **corr})
            ols_rows.append({**meta, **fitted, "standard_error_type": "conventional OLS iid; not HAC/heteroskedasticity robust"})
            merge_rows.append({**meta, "calendar_complete_weeks": len(frame), "valid_predictor_weeks": int(a.sum()),
                "valid_return_weeks": int(b.sum()), "valid_paired_weeks": len(paired),
                "predictor_only_weeks": int((a & ~b).sum()), "return_only_weeks": int((~a & b).sum()),
                "neither_valid_weeks": int((~a & ~b).sum()),
                "first_matched_predictor_week": paired.week_start.min().date().isoformat(),
                "last_matched_predictor_week": paired.week_start.max().date().isoformat(),
                "first_matched_return_week": paired[("next_return_week_start" if horizon == "exploratory_next_week" else "week_start")].min().date().isoformat(),
                "last_matched_return_week": paired[("next_return_week_start" if horizon == "exploratory_next_week" else "week_start")].max().date().isoformat()})
            for row in frame.to_dict("records"):
                xv, yv = row[predictor], row[target]
                valid_x, valid_y = pd.notna(xv), pd.notna(yv)
                pair_rows.append({"predictor": predictor, "relationship": horizon, "week_start": row["week_start"],
                    "return_week_start": row["next_return_week_start"] if horizon == "exploratory_next_week" else row["week_start"],
                    "predictor_value": xv, "return_value": yv,
                    "pair_status": "matched" if valid_x and valid_y else "predictor_only" if valid_x else "return_only" if valid_y else "neither_valid"})
    correlations_table, ols_table, merge_table, pair_table = map(pd.DataFrame, (correlation_rows, ols_rows, merge_rows, pair_rows))
    compact = correlations_table.merge(ols_table[["predictor", "relationship", "alpha", "beta", "p_value_beta_two_sided", "R_squared"]],
        on=["predictor", "relationship"], validate="one_to_one")
    compact = compact.loc[compact.predictor.isin(["broad_WSI_count", "broad_WSI_continuous", "FGI_weekly_mean"])]
    return correlations_table, ols_table, merge_table, pair_table, compact


def consecutive_week_returns(prices):
    """The caller supplies a complete Monday grid; missing closes are never forward-filled."""
    assert isinstance(prices.index, pd.DatetimeIndex) and prices.index.is_unique
    assert prices.index.equals(pd.date_range(prices.index.min(), prices.index.max(), freq="7D"))
    return np.log(prices / prices.shift(1))
