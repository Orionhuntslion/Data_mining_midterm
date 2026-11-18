"""Final approved exploratory S&P 500 return analysis; no upstream result changes."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".packages"))
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from guardian_market_data import obtain_market_csv, SOURCE_PAGE
from guardian_market_stats import consecutive_week_returns, analyze_market

START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2026-10-25")
RESULTS = ROOT / "results/guardian_market_analysis"
FIGURES = ROOT / "figures/guardian_market_analysis"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_file(frame, name):
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def dataset_file(frame, name):
    csv_file(frame, name + ".csv")
    frame.to_parquet(RESULTS / (name + ".parquet"), index=False)


def protected_inputs():
    folders = ["data/processed/guardian_text", "data/processed/guardian_text_final", "data/processed/guardian_vader",
        "data/external", "results/guardian_text_preprocessing", "results/guardian_text_final", "results/guardian_sentiment",
        "results/guardian_fgi_validation", "figures/guardian_text_preprocessing", "figures/guardian_text_final",
        "figures/guardian_sentiment", "figures/guardian_fgi_validation"]
    files = [ROOT / "FINAL_PREPROCESSING_POLICY.json", ROOT / "FINAL_PREPROCESSING_POLICY.md"]
    for folder in folders:
        files.extend(sorted(p for p in (ROOT / folder).rglob("*") if p.is_file()))
    return files


def market_datasets(calendar):
    path, metadata = obtain_market_csv()
    raw = pd.read_csv(path, na_values=["."], float_precision="round_trip")
    date_column = "observation_date" if "observation_date" in raw else "DATE"
    dates = pd.to_datetime(raw[date_column], format="%Y-%m-%d", errors="coerce")
    prices = pd.to_numeric(raw.SP500, errors="coerce")
    full = pd.DataFrame({"date": dates, "sp500_close": prices})
    in_period = full.loc[full.date.between(START, END)].copy().sort_values("date")
    audit = {"source": "FRED SP500", "source_page": SOURCE_PAGE, "original_provider": metadata["original_provider"],
        "download_url": metadata["request_url"], "raw_sha256": metadata["sha256"],
        "raw_download_rows": len(raw), "in_period_source_rows": len(in_period),
        "outside_period_rows_excluded": len(raw) - len(in_period),
        "invalid_dates_in_download": int(dates.isna().sum()),
        "invalid_nonmissing_prices_in_download": int((prices.isna() & raw.SP500.notna()).sum()),
        "duplicate_dates_in_download_excess": int(dates.duplicated().sum()),
        "duplicate_dates_in_period_excess": int(in_period.date.duplicated().sum()),
        "missing_prices_in_period_source_rows": int(in_period.sp500_close.isna().sum()),
        "nonpositive_prices_in_period": int(in_period.sp500_close.le(0).sum()),
        "price_field_selected": "SP500 daily index close", "adjusted_close_available": False,
        "price_basis": "price index, excluding dividends; not total return; no artificial adjusted series",
        "missing_price_handling": "source holiday blanks audited separately; no interpolation/forward filling",
        "date_handling": "source observation date is the US session label; not a UTC-midnight timestamp"}
    json_file(audit, RESULTS / "market_daily_audit.json")
    assert not any(audit[k] for k in ["invalid_dates_in_download", "invalid_nonmissing_prices_in_download",
        "duplicate_dates_in_download_excess", "duplicate_dates_in_period_excess", "nonpositive_prices_in_period"])
    exchange = xcals.get_calendar("XNYS", start=START - pd.Timedelta(days=14), end=END + pd.Timedelta(days=14))
    sessions = exchange.sessions_in_range(START, END)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    valid = in_period.loc[in_period.sp500_close.notna()].copy()
    valid["is_XNYS_session"] = valid.date.isin(sessions)
    assert valid.is_XNYS_session.all(), "Unexpected non-session close; inspect data before analysis"
    all_dates = pd.date_range(START, END, freq="D")
    grid = pd.DataFrame({"date": all_dates, "is_XNYS_session": all_dates.isin(sessions)})
    grid = grid.merge(in_period, on="date", how="left", validate="one_to_one")
    grid["has_valid_close"] = grid.sp500_close.notna()
    missing_sessions = grid.loc[grid.is_XNYS_session & ~grid.has_valid_close]
    audit.update({"first_available_date": valid.date.min().date().isoformat(), "last_available_date": valid.date.max().date().isoformat(),
        "valid_daily_observations": len(valid), "missing_prices_in_validated_dataset": int(valid.sp500_close.isna().sum()),
        "expected_XNYS_sessions": len(sessions), "missing_XNYS_session_prices": len(missing_sessions),
        "holiday_blank_source_rows": int((~in_period.date.isin(sessions) & in_period.sp500_close.isna()).sum()),
        "no_preperiod_seed_price": True})
    json_file(audit, RESULTS / "market_daily_audit.json")
    csv_file(in_period.loc[in_period.sp500_close.isna()], "market_missing_source_rows.csv")
    csv_file(grid, "market_calendar_audit.csv")
    csv_file(missing_sessions, "market_missing_XNYS_sessions.csv")
    valid["source"] = "FRED SP500 / S&P Dow Jones Indices LLC"
    valid["price_basis"] = "price_index_close_excluding_dividends"
    valid["week_start"] = valid.date - pd.to_timedelta(valid.date.dt.weekday, unit="D")
    dataset_file(valid, "sp500_daily_validated")
    by_week = valid.groupby("week_start").agg(weekly_close=("sp500_close", "last"),
        last_price_date=("date", "max"), first_price_date=("date", "min"), price_observation_count=("sp500_close", "count"))
    session_frame = pd.DataFrame({"date": sessions, "week_start": sessions - pd.to_timedelta(sessions.dayofweek, unit="D")})
    expected = session_frame.groupby("week_start").agg(expected_XNYS_sessions=("date", "size"), expected_last_session=("date", "max"))
    weekly = calendar.set_index("week_start").join(by_week).join(expected)
    for column in ["price_observation_count", "expected_XNYS_sessions"]:
        weekly[column] = weekly[column].fillna(0).astype("int64")
    weekly["missing_session_prices"] = weekly.expected_XNYS_sessions - weekly.price_observation_count
    weekly["last_price_matches_expected_last_session"] = weekly.last_price_date.eq(weekly.expected_last_session)
    weekly["previous_week_close"] = weekly.weekly_close.shift(1)
    weekly["previous_price_date"] = weekly.last_price_date.shift(1)
    weekly["weekly_log_return"] = consecutive_week_returns(weekly.weekly_close)
    weekly["return_price_span_calendar_days"] = (weekly.last_price_date - weekly.previous_price_date).dt.days
    weekly["missing_return_reason"] = np.select([weekly.weekly_close.isna(), weekly.previous_week_close.isna()],
        ["no_current_week_close", "no_previous_calendar_week_close"], default="")
    weekly["price_basis"] = "price_index_close_excluding_dividends"
    weekly = weekly.reset_index()
    assert weekly.price_observation_count.sum() == len(valid)
    dataset_file(weekly, "sp500_weekly_returns")
    return weekly, audit


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected_inputs()}
    upstream = json.loads((ROOT / "results/guardian_fgi_validation/independent_validation.json").read_text(encoding="utf-8"))
    assert upstream["status"] == "passed"
    broad = pd.read_csv(ROOT / "results/guardian_sentiment/weekly_broad_wsi.csv", parse_dates=["week_start", "week_end"], float_precision="round_trip")
    strict = pd.read_csv(ROOT / "results/guardian_sentiment/weekly_strict_wsi.csv", parse_dates=["week_start"], float_precision="round_trip")
    fgi = pd.read_csv(ROOT / "results/guardian_fgi_validation/fgi_weekly_validated.csv", parse_dates=["week_start"], float_precision="round_trip")
    calendar = broad[["week_id", "week_start", "week_end", "days_observed", "is_complete_week"]]
    weekly, audit = market_datasets(calendar)
    merged = weekly.copy()
    for label, frame in [("broad", broad), ("strict", strict)]:
        keep = frame[["week_start", "WSI_count", "WSI_continuous", "article_count"]].rename(columns={
            "WSI_count": label + "_WSI_count", "WSI_continuous": label + "_WSI_continuous", "article_count": label + "_article_count"})
        merged = merged.merge(keep, on="week_start", how="left", validate="one_to_one")
    merged = merged.merge(fgi[["week_start", "FGI_weekly_mean", "FGI_observation_count"]], on="week_start", how="left", validate="one_to_one")
    merged = merged.set_index("week_start").sort_index()
    assert len(merged) == 513 and merged.index.equals(pd.date_range(merged.index.min(), merged.index.max(), freq="7D"))
    merged["next_return_week_start"] = merged.index + pd.Timedelta(days=7)
    merged["next_week_log_return"] = merged.weekly_log_return.reindex(merged.index + pd.Timedelta(days=7)).to_numpy()
    merged = merged.reset_index()
    dataset_file(merged, "merged_market_WSI_weekly")
    correlations, ols, diagnostics, pair_status, compact = analyze_market(merged)
    for frame, name in [(correlations, "market_correlations.csv"), (ols, "market_OLS_results.csv"),
        (diagnostics, "market_merge_diagnostics.csv"), (pair_status, "market_pair_status.csv"),
        (compact, "market_benchmark_comparison.csv")]:
        csv_file(frame, name)
    from guardian_market_outputs import make_figures, write_handoff
    figures = make_figures(merged, correlations, ols, RESULTS, FIGURES)
    method = {"analysis": "final exploratory univariate return analysis", "period": ["2017-01-01", "2026-10-25"],
        "week": "existing America/New_York Monday-Sunday; complete weeks used for analysis; partial boundary retained",
        "weekly_price": "last valid close in each calendar week", "weekly_return": "ln(P_w / P_(w-1)); decimal log units",
        "initial_week": "no price outside target period used; first complete-week return missing without previous-week close",
        "gaps": "full calendar retained; no fill; missing week close makes that week and next return missing",
        "next_week_alignment": "target return keyed to week_start + 7 days before dropping missing pairs",
        "OLS": "r_(w+h)=alpha+beta*x_w+epsilon, h=0 or 1; separate single predictor plus intercept for every fit",
        "coefficient_unit": "decimal weekly log return per one 0-100 index point; not a regression on price levels",
        "SE_beta": "sqrt([sum(residual^2)/(n-2)]/sum((x-xbar)^2)); conventional homoskedastic iid OLS",
        "test": "t=beta/SE_beta; two-sided p=2*Student_t_sf(abs(t),df=n-2)",
        "R_squared": "1-SSE/SST", "adjusted_R_squared": "1-(1-R_squared)*(n-1)/(n-2)",
        "correlations": "Pearson and tie-aware average-rank Spearman on each model's valid pair set",
        "CSV_precision": "round_trip; no changes or rounding to original WSI/FGI",
        "FGI_comparison": "optional context benchmark; pairwise sample sizes reported, not necessarily common dates; not ground truth",
        "limitations": ["exploratory associations only", "conventional SE/p-values can be misleading under serial dependence or heteroskedasticity",
            "multiple specifications without multiplicity adjustment", "WSI measurement error, variable article density and overlapping strategies",
            "FGI archived source eras and missing weeks retained", "price return excludes dividends", "no causal, predictive, trading or market-efficiency claims"],
        "upstream_results_modified": False, "forecasting_ML_multivariate": False,
        "versions": {name: importlib.metadata.version(name) for name in ["requests", "pandas", "numpy", "pyarrow", "scipy", "matplotlib", "exchange_calendars"]},
        "python_version": sys.version}
    json_file(method, RESULTS / "market_method.json")
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in protected_inputs()}
    outputs = [p for p in RESULTS.iterdir() if p.is_file() and p.name != "validation_summary.json"]
    validation = {"status": "pipeline_checks_passed", "input_sha256": hashes, "all_upstream_data_results_figures_and_policy_unchanged": True,
        "market_daily_observations": audit["valid_daily_observations"], "weekly_calendar_rows": len(weekly),
        "valid_weekly_returns": int(weekly.weekly_log_return.notna().sum()), "univariate_OLS_models": len(ols),
        "figures": figures, "output_files": [str(p.relative_to(ROOT)) for p in sorted(outputs)],
        "forecasting_ML_multivariate_models": False, "additional_analysis_authorized_or_started": False}
    json_file(validation, RESULTS / "validation_summary.json")
    write_handoff(ROOT, audit, method, diagnostics, correlations, ols, compact, figures)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(diagnostics.to_string(index=False))
    print(compact.to_string(index=False))
    print("Final exploratory stage complete. No forecasts, ML, multivariate or additional analyses.", flush=True)


if __name__ == "__main__":
    main()
