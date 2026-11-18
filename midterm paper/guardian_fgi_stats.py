"""Calendar-aligned descriptive correlations; no p-values or predictive models."""
import numpy as np
import pandas as pd

METRICS = ("WSI_count", "WSI_continuous")
LAGS = range(-4, 5)


def correlations(x, y):
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(paired)
    if n < 3 or paired.x.nunique() < 2 or paired.y.nunique() < 2:
        return {"n_pairs": n, "pearson": np.nan, "spearman": np.nan}
    # Spearman is Pearson on average ranks, with ties ranked within each valid pair set.
    ranks = paired.rank(method="average")
    return {"n_pairs": n, "pearson": float(np.corrcoef(paired.x, paired.y)[0, 1]),
            "spearman": float(np.corrcoef(ranks.x, ranks.y)[0, 1])}


def shift_by_calendar_week(series, index, lag):
    """At week w return the observation at w+k, never the kth nonmissing row."""
    shifted = series.reindex(index + pd.Timedelta(weeks=lag)).copy()
    shifted.index = index
    return shifted


def contemporaneous(merged):
    rows = []
    for strategy, frame in merged.items():
        frame = frame.loc[frame.is_complete_week]
        for metric in METRICS:
            for fgi in ("FGI_weekly_mean", "FGI_weekly_last"):
                rows.append({"query_strategy": strategy, "WSI_metric": metric, "FGI_metric": fgi,
                    "analysis": "levels", "scope": "complete_weeks", **correlations(frame[metric], frame[fgi])})
    return pd.DataFrame(rows)


def lead_lag_tables(merged, differences=False):
    rows, delta_rows = [], []
    for strategy, frame in merged.items():
        frame = frame.set_index("week_start").sort_index()
        index = pd.date_range(frame.index.min(), frame.index.max(), freq="7D")
        assert frame.index.equals(index)
        complete = frame.is_complete_week
        fgi = frame.FGI_weekly_mean.where(complete)
        if differences:
            fgi = fgi.diff()  # missing current/previous calendar week remains missing
        for metric in METRICS:
            wsi = frame[metric].where(complete)
            if differences:
                wsi = wsi.diff()
                delta_rows.extend({"query_strategy": strategy, "week_start": w, "WSI_metric": metric,
                    "delta_WSI": wsi.loc[w], "delta_FGI_weekly_mean": fgi.loc[w]} for w in index)
            for lag in LAGS:
                future = shift_by_calendar_week(fgi, index, lag)
                rows.append({"query_strategy": strategy, "WSI_metric": metric, "FGI_metric": "FGI_weekly_mean",
                    "analysis": "first_differences" if differences else "levels", "scope": "complete_weeks",
                    "lag_weeks": lag, **correlations(wsi, future)})
    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def strongest_lags(table):
    rows = []
    for keys, subset in table.groupby(["analysis", "query_strategy", "WSI_metric"], sort=False):
        for method in ("pearson", "spearman"):
            valid = subset.dropna(subset=[method]).copy()
            best = valid[method].abs().max()
            tied = valid.loc[np.isclose(valid[method].abs(), best, rtol=0, atol=1e-12)].copy()
            tied["abs_lag"] = tied.lag_weeks.abs()
            selected = tied.sort_values(["abs_lag", "lag_weeks"]).iloc[0]
            rows.append({"analysis": keys[0], "query_strategy": keys[1], "WSI_metric": keys[2],
                "correlation_method": method, "best_lag_weeks": int(selected.lag_weeks),
                "correlation": selected[method], "n_pairs": int(selected.n_pairs),
                "tied_best_lags": ",".join(str(k) for k in sorted(tied.lag_weeks))})
    return pd.DataFrame(rows)
