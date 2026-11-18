"""Weekly descriptive statistics and the two approved WSI specifications only."""
from __future__ import annotations

import numpy as np
import pandas as pd

FLAGS = {
    "broad_business_control": "in_broad_business_control",
    "anchors_stock_market_tag": "in_anchors_stock_market_tag",
}
CLASSES = ("positive", "neutral", "negative")
SENTIMENT_COLUMNS = [
    "positive_share", "neutral_share", "negative_share", "mean_compound",
    "median_compound", "std_compound", "sentiment_balance", "WSI_count",
    "WSI_continuous", "se_balance", "se_WSI_count", "WSI_count_ci95_low",
    "WSI_count_ci95_high",
]


def classify(compound):
    """Inclusive standard VADER thresholds; no manual overrides."""
    return "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"


def weekly_sentiment(articles, calendar, strategy):
    """Reindex onto the existing NY Monday–Sunday calendar, including empty weeks."""
    selected = articles.loc[articles[FLAGS[strategy]]].copy()
    selected["week_start"] = pd.to_datetime(selected.week_start)
    result = calendar.loc[calendar.query_strategy.eq(strategy), [
        "query_strategy", "week_id", "week_start", "week_end", "days_observed", "is_complete_week",
        "article_count",
    ]].copy()
    result["week_start"] = pd.to_datetime(result.week_start)
    result = result.set_index("week_start").sort_index()
    groups = selected.groupby("week_start").vader_compound
    counts = groups.size().reindex(result.index, fill_value=0)
    assert np.array_equal(result.article_count.to_numpy(), counts.to_numpy())
    for label in CLASSES:
        count = selected.loc[selected.sentiment_class.eq(label)].groupby("week_start").size()
        result[label + "_count"] = count.reindex(result.index, fill_value=0).astype("int64")
        result[label + "_share"] = result[label + "_count"] / counts.replace(0, np.nan)
    for field, values in [("mean_compound", groups.mean()), ("median_compound", groups.median()),
                          ("std_compound", groups.std(ddof=1))]:
        result[field] = values.reindex(result.index)
    balance = result.positive_share - result.negative_share
    result["sentiment_balance"] = balance
    result["WSI_count"] = 50 * (1 + balance)
    result["WSI_continuous"] = 50 * (1 + result.mean_compound)
    variance = (result.positive_share + result.negative_share - balance ** 2) / counts.replace(0, np.nan)
    result["se_balance"] = np.sqrt(variance.clip(lower=0))
    result["se_WSI_count"] = 50 * result.se_balance
    # Keep the requested untruncated normal approximation, even outside [0, 100].
    result["WSI_count_ci95_low"] = result.WSI_count - 1.96 * result.se_WSI_count
    result["WSI_count_ci95_high"] = result.WSI_count + 1.96 * result.se_WSI_count
    result["small_sample_lt3"] = counts.lt(3)
    result["small_sample_lt5"] = counts.lt(5)
    result["unanimous_sentiment_week"] = result[[c + "_count" for c in CLASSES]].max(axis=1).eq(counts) & counts.gt(0)
    result["approx_interval_degenerate"] = counts.gt(0) & result.se_WSI_count.eq(0)
    result["interval_method"] = "multinomial plug-in normal approximation; 1.96; no independence guarantee"
    assert result.loc[counts.eq(0), SENTIMENT_COLUMNS].isna().all().all()
    assert result[[c + "_count" for c in CLASSES]].sum(axis=1).eq(counts).all()
    return result.reset_index()


def descriptive_tables(articles, weekly):
    article_rows, class_rows, week_rows, comparisons = [], [], [], []
    for strategy, flag in FLAGS.items():
        subset = articles.loc[articles[flag]]
        values = subset.vader_compound
        row = {"query_strategy": strategy, "article_count": len(subset), "mean_compound": values.mean(),
               "median_compound": values.median(), "std_compound": values.std(ddof=1),
               "min_compound": values.min(), "max_compound": values.max(),
               "skewness": values.skew(), "excess_kurtosis": values.kurt(),
               "pearson_kurtosis": values.kurt() + 3}
        for label in CLASSES:
            n = int(subset.sentiment_class.eq(label).sum())
            row[label + "_count"] = n
            row[label + "_share"] = n / len(subset)
            class_rows.append({"query_strategy": strategy, "sentiment_class": label, "count": n, "share": n / len(subset)})
        article_rows.append(row)
    for scope in ("complete_weeks", "all_calendar_weeks"):
        weeks = weekly.loc[weekly.is_complete_week] if scope == "complete_weeks" else weekly
        for strategy in FLAGS:
            subset = weeks.loc[weeks.query_strategy.eq(strategy)]
            for metric in ("WSI_count", "WSI_continuous"):
                values = subset[metric].dropna()
                week_rows.append({"scope": scope, "query_strategy": strategy, "metric": metric,
                    "calendar_weeks": len(subset), "valid_weeks": len(values), "missing_weeks": int(subset[metric].isna().sum()),
                    "mean": values.mean(), "median": values.median(), "std": values.std(ddof=1),
                    "min": values.min(), "max": values.max(),
                    "weeks_lt3_articles": int(subset.article_count.lt(3).sum()),
                    "weeks_lt5_articles": int(subset.article_count.lt(5).sum()),
                    "degenerate_intervals": int(subset.approx_interval_degenerate.sum()),
                    "intervals_outside_0_100": int((subset.WSI_count_ci95_low.lt(0) | subset.WSI_count_ci95_high.gt(100)).sum())})
        for metric in ("WSI_count", "WSI_continuous"):
            paired = weeks.pivot(index="week_start", columns="query_strategy", values=metric).dropna()
            a, b = FLAGS
            comparisons.append({"scope": scope, "metric": metric, "paired_valid_weeks": len(paired),
                                "broad_strict_pearson_correlation": paired[a].corr(paired[b])})
    return tuple(pd.DataFrame(rows) for rows in (article_rows, class_rows, week_rows, comparisons))
