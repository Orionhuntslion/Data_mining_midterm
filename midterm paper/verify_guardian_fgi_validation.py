"""Independent date-key, missingness, rank-correlation and visual-fit verification."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from functools import partial

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".packages"))
import numpy as np
import pandas as pd
read_csv = partial(pd.read_csv, float_precision="round_trip")
from guardian_fgi_stats import correlations, lead_lag_tables, shift_by_calendar_week

FOLDER = ROOT / "results/guardian_fgi_validation"
STRATEGIES = {"broad": "broad_business_control", "strict": "anchors_stock_market_tag"}


def close(a, b):
    assert np.isclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True), (a, b)


def pearson_manual(x, y):
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    covariance = math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    return covariance / math.sqrt(math.fsum((a - mean_x) ** 2 for a in x) * math.fsum((b - mean_y) ** 2 for b in y))


def average_ranks(values):
    # Independent average-rank implementation; does not use pandas.rank or production helper.
    groups = {}
    for rank, (_, value) in enumerate(sorted(enumerate(values), key=lambda p: p[1]), 1):
        groups.setdefault(value, []).append(rank)
    return [statistics.mean(groups[v]) for v in values]


def assert_correlations(row, pairs):
    assert row.n_pairs == len(pairs), (row, len(pairs))
    x, y = zip(*pairs)
    close(row.pearson, pearson_manual(x, y))
    close(row.spearman, pearson_manual(average_ranks(x), average_ranks(y)))


def synthetic_checks():
    index = pd.date_range("2020-01-06", periods=14, freq="7D")
    x = pd.Series([8, 1, 5, 3, 9, 0, 7, 2, 6, 4, 11, 10, 14, 12], index=index, dtype=float)
    y = x.shift(1)
    frame = pd.DataFrame({"week_start": index, "is_complete_week": True, "WSI_count": x.values,
        "WSI_continuous": x.values, "FGI_weekly_mean": y.values})
    table, _ = lead_lag_tables({"synthetic": frame})
    best = table.loc[table.WSI_metric.eq("WSI_count")].sort_values("pearson", ascending=False).iloc[0]
    assert best.lag_weeks == 1
    close(best.pearson, 1)
    reverse, _ = lead_lag_tables({"synthetic": frame.assign(WSI_count=y.values, WSI_continuous=y.values, FGI_weekly_mean=x.values)})
    assert reverse.loc[reverse.WSI_metric.eq("WSI_count")].sort_values("pearson", ascending=False).iloc[0].lag_weeks == -1
    y.iloc[5] = np.nan
    moved = shift_by_calendar_week(y, index, 1)
    assert pd.isna(moved.iloc[4]) and moved.iloc[5] == y.iloc[6]
    frame["FGI_weekly_mean"] = y.values
    _, deltas = lead_lag_tables({"synthetic": frame}, differences=True)
    row = deltas.loc[deltas.WSI_metric.eq("WSI_count")].set_index("week_start")
    assert pd.isna(row.loc[index[5], "delta_FGI_weekly_mean"])
    assert pd.isna(row.loc[index[6], "delta_FGI_weekly_mean"])
    pair = correlations(pd.Series([1, 1, 2, 3]), pd.Series([2, 2, 1, 4]))
    close(pair["spearman"], pearson_manual([1.5, 1.5, 3, 4], [2.5, 2.5, 1, 4]))
    assert math.isnan(correlations(pd.Series([1, 1, 1]), pd.Series([1, 2, 3]))["pearson"])


def main():
    synthetic_checks()
    validation = json.loads((FOLDER / "validation_summary.json").read_text(encoding="utf-8"))
    for path, digest in validation["input_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    source = read_csv(ROOT / "data/external/fear_greed_2017-01-01_2026-10-25.csv", parse_dates=["date"])
    assert len(source) == 2443 and source.date.is_unique and not source[["date", "fgi"]].isna().any().any()
    daily = pd.read_parquet(FOLDER / "fgi_daily_validated.parquet")
    pd.testing.assert_frame_equal(source.sort_values("date").reset_index(drop=True),
        daily[source.columns].reset_index(drop=True))
    week_groups = {}
    for record in source.sort_values("date").itertuples():
        day = record.date.date()
        monday = day - dt.timedelta(days=day.weekday())
        week_groups.setdefault(monday, []).append((day, record.fgi))
    fgi = read_csv(FOLDER / "fgi_weekly_validated.csv", parse_dates=["week_start", "week_end"])
    assert len(fgi) == 513 and fgi.week_start.is_unique
    assert fgi.FGI_observation_count.sum() == 2443
    lookup = {}
    for row in fgi.itertuples():
        week = row.week_start.date()
        group = week_groups.get(week, [])
        assert row.FGI_observation_count == len(group)
        if group:
            close(row.FGI_weekly_mean, statistics.mean(v for _, v in group))
            close(row.FGI_weekly_last, group[-1][1])
            assert str(row.FGI_last_observed_date)[:10] == group[-1][0].isoformat()
            lookup[week] = row.FGI_weekly_mean
        else:
            assert pd.isna(row.FGI_weekly_mean) and pd.isna(row.FGI_weekly_last)
    calendar = read_csv(FOLDER / "fgi_calendar_date_audit.csv")
    assert len(calendar) == 3585 and calendar.has_valid_fgi.sum() == 2443
    assert calendar.is_XNYS_session.sum() == 2466
    assert (calendar.is_XNYS_session & ~calendar.has_valid_fgi).sum() == 23
    assert ((~calendar.is_XNYS_session) & calendar.has_valid_fgi).sum() == 0
    audit = json.loads((FOLDER / "fgi_daily_audit.json").read_text(encoding="utf-8"))
    assert audit["XNYS_sessions_represented"] == 2443 and audit["XNYS_sessions_without_value"] == 23
    merged = {}
    merge_stats = read_csv(FOLDER / "merge_diagnostics.csv").set_index("query_strategy")
    for label, strategy in STRATEGIES.items():
        original = read_csv(ROOT / f"results/guardian_sentiment/weekly_{label}_wsi.csv", parse_dates=["week_start", "week_end"])
        frame = read_csv(FOLDER / f"merged_{label}_FGI_weekly.csv", parse_dates=["week_start", "week_end"])
        assert len(frame) == 513
        pd.testing.assert_frame_equal(original, frame[original.columns], check_exact=False, rtol=1e-12, atol=1e-12)
        for column in ["WSI_count", "WSI_continuous"]:
            assert frame[column].isna().equals(original[column].isna())
        fgi_aligned = fgi.set_index("week_start").reindex(frame.week_start)
        for column in ["FGI_weekly_mean", "FGI_weekly_last", "FGI_observation_count"]:
            assert np.allclose(frame[column], fgi_aligned[column], equal_nan=True)
        a, b = frame.WSI_count.notna(), frame.FGI_weekly_mean.notna()
        assert merge_stats.loc[strategy, "matched_value_weeks"] == int((a & b).sum())
        assert merge_stats.loc[strategy, "unmatched_WSI_value_weeks"] == int((a & ~b).sum())
        assert merge_stats.loc[strategy, "unmatched_FGI_value_weeks"] == int((~a & b).sum())
        merged[strategy] = frame
    contemporaneous = read_csv(FOLDER / "contemporaneous_correlations.csv")
    for row in contemporaneous.itertuples():
        subset = merged[row.query_strategy].loc[lambda d: d.is_complete_week, [row.WSI_metric, row.FGI_metric]].dropna()
        assert_correlations(row, list(subset.itertuples(index=False, name=None)))
    # Independent date-key dictionaries: never call the production shift or diff helpers.
    for filename, difference in [("lead_lag_correlations.csv", False), ("first_difference_robustness.csv", True)]:
        table = read_csv(FOLDER / filename)
        assert len(table) == 36
        for row in table.itertuples():
            frame = merged[row.query_strategy].loc[lambda d: d.is_complete_week]
            wsi_values = {r.week_start.date(): getattr(r, row.WSI_metric) for r in frame.itertuples()}
            fgi_values = {r.week_start.date(): r.FGI_weekly_mean for r in frame.itertuples()}
            if difference:
                def differences(values):
                    return {week: value - values.get(week - dt.timedelta(days=7), float("nan")) for week, value in values.items()}
                wsi_values, fgi_values = differences(wsi_values), differences(fgi_values)
            pairs = []
            for week, x in wsi_values.items():
                y = fgi_values.get(week + dt.timedelta(weeks=row.lag_weeks), float("nan"))
                if pd.notna(x) and pd.notna(y):
                    pairs.append((x, y))
            assert_correlations(row, pairs)
    deltas = read_csv(FOLDER / "weekly_first_differences.csv", parse_dates=["week_start"])
    for strategy, frame in merged.items():
        current = frame.set_index("week_start")
        for metric in ["WSI_count", "WSI_continuous"]:
            rows = deltas.loc[deltas.query_strategy.eq(strategy) & deltas.WSI_metric.eq(metric)].set_index("week_start")
            for week, record in rows.iterrows():
                previous = week - pd.Timedelta(days=7)
                for source_column, output_column in [(metric, "delta_WSI"), ("FGI_weekly_mean", "delta_FGI_weekly_mean")]:
                    expected = float("nan")
                    if previous in current.index and current.loc[week, "is_complete_week"] and current.loc[previous, "is_complete_week"]:
                        expected = current.loc[week, source_column] - current.loc[previous, source_column]
                    close(record[output_column], expected)
    levels = read_csv(FOLDER / "lead_lag_correlations.csv")
    changes = read_csv(FOLDER / "first_difference_robustness.csv")
    peaks = read_csv(FOLDER / "strongest_lead_lag_correlations.csv")
    combined = pd.concat([levels, changes])
    for row in peaks.itertuples():
        subset = combined.loc[combined.analysis.eq(row.analysis) & combined.query_strategy.eq(row.query_strategy) & combined.WSI_metric.eq(row.WSI_metric)]
        close(abs(row.correlation), subset[row.correlation_method].abs().max())
        expected = subset.loc[subset.lag_weeks.eq(row.best_lag_weeks)].iloc[0]
        close(row.correlation, expected[row.correlation_method])
        assert row.n_pairs == expected.n_pairs
    broad = merged["broad_business_control"]
    pair = broad.loc[broad.is_complete_week, ["week_start", "WSI_count", "FGI_weekly_mean"]].dropna()
    z = read_csv(FOLDER / "standardized_overlay_data.csv")
    for column in ["WSI_count_z", "FGI_weekly_mean_z"]:
        close(z[column].dropna().mean(), 0); close(z[column].dropna().std(ddof=1), 1)
        assert z[column].notna().sum() == len(pair)
    fit = json.loads((FOLDER / "scatter_visual_fit.json").read_text(encoding="utf-8"))
    close(fit["R_squared"], fit["pearson_r"] ** 2)
    close(fit["pearson_r"], pearson_manual(pair.WSI_count.tolist(), pair.FGI_weekly_mean.tolist()))
    residuals = pair.WSI_count - (fit["intercept"] + fit["slope"] * pair.FGI_weekly_mean)
    close(residuals.mean(), 0)
    assert fit["n_pairs"] == len(pair) == 508
    from PIL import Image
    for name in validation["figures"]:
        with Image.open(ROOT / "figures/guardian_fgi_validation" / name) as image:
            assert image.width > 1000 and image.height > 500
            image.verify()
    for path, digest in validation["input_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    result = {"status": "passed", "FGI_daily_records_verified": len(source), "weekly_calendar_rows": len(fgi),
        "FGI_weekly_means_last_values_counts_independently_verified": True,
        "original_WSI_columns_values_and_missing_weeks_preserved": True,
        "correlation_rows_verified": len(contemporaneous), "level_lead_lag_rows_verified": len(levels),
        "difference_lead_lag_rows_verified": len(changes), "tie_aware_rank_correlations_independently_verified": True,
        "positive_negative_lag_sign_and_missing_gap_synthetic_tests": "passed",
        "differences_do_not_bridge_missing_calendar_weeks": True, "scatter_fit_R_squared_and_z_scores_verified": True,
        "all_protected_input_hashes_unchanged": True, "market_returns_or_forecasting_used": False}
    (FOLDER / "independent_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
