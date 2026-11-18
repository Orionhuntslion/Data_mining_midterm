"""Verify the final market stage independently of its aggregation and OLS formulas."""
import datetime as dt
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".packages"))
import numpy as np
import pandas as pd
from scipy import stats, special
from guardian_market_stats import simple_ols, consecutive_week_returns, PREDICTORS, TARGETS

FOLDER = ROOT / "results/guardian_market_analysis"
read_csv = partial(pd.read_csv, float_precision="round_trip")


def close(a, b):
    assert np.isclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True), (a, b)


def synthetic_checks():
    weeks = pd.date_range("2026-01-05", periods=5, freq="7D")
    prices = pd.Series([100, 110, np.nan, 121, 133.1], index=weeks)
    returns = consecutive_week_returns(prices)
    assert pd.isna(returns.iloc[0]) and pd.isna(returns.iloc[2]) and pd.isna(returns.iloc[3])
    close(returns.iloc[1], math.log(1.1)); close(returns.iloc[4], math.log(1.1))
    # Exact next calendar week cannot skip the missing week at index 2.
    ahead = returns.reindex(weeks + pd.Timedelta(days=7)).to_numpy()
    assert pd.isna(ahead[1]) and pd.isna(ahead[4])
    close(ahead[0], returns.iloc[1]); close(ahead[3], returns.iloc[4])
    x = np.arange(1, 10, dtype=float)
    y = 1 + 0.3 * x + np.array([0.2, -0.1, 0.1, -0.2, 0, 0.3, -0.2, 0.1, -0.1])
    ours, independent = simple_ols(x, y), stats.linregress(x, y)
    close(ours["beta"], independent.slope); close(ours["alpha"], independent.intercept)
    close(ours["se_beta"], independent.stderr); close(ours["p_value_beta_two_sided"], independent.pvalue)
    close(ours["R_squared"], independent.rvalue ** 2)


def main():
    synthetic_checks()
    validation = json.loads((FOLDER / "validation_summary.json").read_text(encoding="utf-8"))
    for path, digest in validation["input_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    audit = json.loads((FOLDER / "market_daily_audit.json").read_text(encoding="utf-8"))
    raw_path = ROOT / "data/raw/market/fred_sp500_2017-01-01_2026-10-25.csv"
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == audit["raw_sha256"]
    raw = read_csv(raw_path, na_values=["."], parse_dates=["observation_date"])
    source = raw.loc[raw.observation_date.between("2017-01-01", "2026-10-25") & raw.SP500.notna()]
    price_map = {row.observation_date.date(): row.SP500 for row in source.itertuples()}
    daily = pd.read_parquet(FOLDER / "sp500_daily_validated.parquet")
    assert len(daily) == len(price_map) == 2466 and daily.date.is_unique
    assert {row.date.date(): row.sp500_close for row in daily.itertuples()} == price_map
    assert daily.is_XNYS_session.all() and not daily.sp500_close.isna().any()
    calendar = read_csv(FOLDER / "market_calendar_audit.csv")
    assert len(calendar) == 3585 and calendar.is_XNYS_session.sum() == 2466
    assert not (calendar.is_XNYS_session & ~calendar.has_valid_close).any()
    assert audit["first_available_date"] == "2017-01-03" and audit["last_available_date"] == "2026-10-23"
    assert audit["duplicate_dates_in_period_excess"] == 0 and audit["missing_XNYS_session_prices"] == 0
    groups = {}
    for day, value in price_map.items():
        monday = day - dt.timedelta(days=day.weekday())
        groups.setdefault(monday, []).append((day, value))
    last_prices = {week: max(values, key=lambda x: x[0]) for week, values in groups.items()}
    weekly = read_csv(FOLDER / "sp500_weekly_returns.csv", parse_dates=["week_start", "week_end"])
    assert len(weekly) == 513 and weekly.week_start.is_unique and weekly.is_complete_week.sum() == 512
    assert weekly.week_start.diff().dropna().eq(pd.Timedelta(days=7)).all()
    for row in weekly.itertuples():
        week = row.week_start.date()
        current, previous = last_prices.get(week), last_prices.get(week - dt.timedelta(days=7))
        assert row.price_observation_count == len(groups.get(week, []))
        if current:
            close(row.weekly_close, current[1]); assert row.last_price_date[:10] == current[0].isoformat()
        else:
            assert pd.isna(row.weekly_close)
        if current and previous:
            close(row.weekly_log_return, math.log(current[1] / previous[1]))
        else:
            assert pd.isna(row.weekly_log_return)
    assert weekly.weekly_log_return.notna().sum() == 511
    available = weekly.loc[weekly.weekly_log_return.notna()]
    assert available.week_start.min() == pd.Timestamp("2017-01-09")
    assert available.week_start.max() == pd.Timestamp("2026-10-19")
    merged = read_csv(FOLDER / "merged_market_WSI_weekly.csv", parse_dates=["week_start", "week_end", "next_return_week_start"])
    market_map = weekly.set_index("week_start").weekly_log_return.to_dict()
    for row in merged.itertuples():
        close(row.weekly_log_return, market_map[row.week_start])
        assert row.next_return_week_start == row.week_start + pd.Timedelta(days=7)
        close(row.next_week_log_return, market_map.get(row.week_start + pd.Timedelta(days=7), float("nan")))
    for label in ["broad", "strict"]:
        original = read_csv(ROOT / f"results/guardian_sentiment/weekly_{label}_wsi.csv")
        for metric in ["WSI_count", "WSI_continuous"]:
            assert np.array_equal(original[metric], merged[label + "_" + metric], equal_nan=True)
        assert merged[label + "_article_count"].equals(original.article_count)
    fgi = read_csv(ROOT / "results/guardian_fgi_validation/fgi_weekly_validated.csv")
    assert np.array_equal(fgi.FGI_weekly_mean, merged.FGI_weekly_mean, equal_nan=True)
    # The machine-readable Parquet exports must match their inspection CSV counterparts.
    for stem, columns in [("sp500_weekly_returns", ["weekly_close", "previous_week_close", "weekly_log_return"]),
        ("merged_market_WSI_weekly", [*PREDICTORS, "weekly_log_return", "next_week_log_return"])]:
        parquet, csv = pd.read_parquet(FOLDER / (stem + ".parquet")), read_csv(FOLDER / (stem + ".csv"))
        for column in columns:
            assert np.array_equal(parquet[column], csv[column], equal_nan=True)
    results = read_csv(FOLDER / "market_OLS_results.csv")
    correlations = read_csv(FOLDER / "market_correlations.csv")
    diagnostics = read_csv(FOLDER / "market_merge_diagnostics.csv")
    pair_status = read_csv(FOLDER / "market_pair_status.csv")
    assert len(results) == len(correlations) == len(diagnostics) == 10
    assert set(zip(results.predictor, results.relationship)) == {(p, h) for p in PREDICTORS for h in TARGETS}
    for row in results.itertuples():
        pair = merged.loc[merged.is_complete_week, ["week_start", row.predictor, row.return_variable]].dropna()
        x, y = pair[row.predictor].to_numpy(), pair[row.return_variable].to_numpy()
        independent = stats.linregress(x, y)
        assert row.n_observations == len(pair)
        close(row.alpha, independent.intercept); close(row.beta, independent.slope)
        close(row.se_beta, independent.stderr); close(row.t_beta, independent.slope / independent.stderr)
        close(row.p_value_beta_two_sided, independent.pvalue)
        # Independent special-function expression for the two-sided t p-value.
        df = len(pair) - 2
        close(row.p_value_beta_two_sided, special.betainc(df / 2, 0.5, df / (df + row.t_beta ** 2)))
        close(row.R_squared, independent.rvalue ** 2)
        close(row.adjusted_R_squared, 1 - (1 - independent.rvalue ** 2) * (len(pair) - 1) / df)
        corr = correlations.loc[correlations.predictor.eq(row.predictor) & correlations.relationship.eq(row.relationship)].iloc[0]
        close(corr.pearson, stats.pearsonr(x, y).statistic)
        close(corr.spearman, stats.spearmanr(x, y).statistic)
        assert corr.n_pairs == len(pair)
        diag = diagnostics.loc[diagnostics.predictor.eq(row.predictor) & diagnostics.relationship.eq(row.relationship)].iloc[0]
        assert diag.valid_paired_weeks == len(pair)
        assert diag.first_matched_predictor_week == pair.week_start.min().date().isoformat()
        assert diag.last_matched_predictor_week == pair.week_start.max().date().isoformat()
        frame = merged.loc[merged.is_complete_week]
        a, b = frame[row.predictor].notna(), frame[row.return_variable].notna()
        assert diag.predictor_only_weeks == (a & ~b).sum()
        assert diag.return_only_weeks == (~a & b).sum()
        status = pair_status.loc[pair_status.predictor.eq(row.predictor) & pair_status.relationship.eq(row.relationship)]
        assert status.pair_status.eq("matched").sum() == len(pair)
    compact = read_csv(FOLDER / "market_benchmark_comparison.csv")
    assert len(compact) == 6 and set(compact.predictor) == {"broad_WSI_count", "broad_WSI_continuous", "FGI_weekly_mean"}
    plot = read_csv(FOLDER / "market_timeseries_plot_data.csv")
    close(plot.broad_WSI_count_z.dropna().mean(), 0)
    close(plot.broad_WSI_count_z.dropna().std(ddof=1), 1)
    from PIL import Image
    for name in validation["figures"]:
        path = ROOT / "figures/guardian_market_analysis" / name
        if path.suffix == ".png":
            with Image.open(path) as image:
                assert image.width >= 1500 and image.height >= 1000
                image.verify()
        else:
            assert ET.parse(path).getroot().tag.endswith("svg")
    for path, digest in validation["input_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    report = {"status": "passed", "daily_prices_verified_against_raw_FRED": len(daily),
        "daily_duplicates": 0, "missing_XNYS_prices": 0, "weekly_calendar_rows": 513, "weekly_returns_verified": 511,
        "last_prices_and_consecutive_week_log_formula_independently_verified": True,
        "no_fill_and_no_gap_skipping_synthetic_tests": "passed", "next_week_date_alignment_verified": True,
        "original_WSI_and_FGI_values_and_missingness_exactly_preserved": True,
        "univariate_OLS_models_verified_against_SciPy": len(results), "correlation_rows_verified_against_SciPy": len(correlations),
        "conventional_SE_t_p_R2_adjusted_R2_verified": True, "p_values_also_verified_via_incomplete_beta_function": True,
        "merge_diagnostics_and_benchmark_comparison_verified": True, "figures_verified": len(validation["figures"]),
        "all_upstream_file_hashes_unchanged": True, "forecasting_ML_multivariate_or_additional_models": False}
    (FOLDER / "independent_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
