"""Validate cached FGI and existing weekly WSI; stop after level/change associations."""
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
from guardian_fgi_stats import contemporaneous, lead_lag_tables, strongest_lags

START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2026-10-25")
FGI_PATH = ROOT / "data/external/fear_greed_2017-01-01_2026-10-25.csv"
WSI_FOLDER = ROOT / "results/guardian_sentiment"
RESULTS = ROOT / "results/guardian_fgi_validation"
FIGURES = ROOT / "figures/guardian_fgi_validation"
STRATEGIES = {"broad": "broad_business_control", "strict": "anchors_stock_market_tag"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(frame, name):
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def save_dataset(frame, name):
    save_csv(frame, name + ".csv")
    frame.to_parquet(RESULTS / (name + ".parquet"), index=False)


def audit_fgi(calendar):
    original = pd.read_csv(FGI_PATH, float_precision="round_trip")
    frame = original.copy()
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="coerce")
    numeric = pd.to_numeric(frame.fgi, errors="coerce")
    audit = {"source": str(FGI_PATH.relative_to(ROOT)), "sha256": sha(FGI_PATH), "daily_rows": len(frame),
        "unique_dates": int(frame.date.nunique()), "duplicate_date_rows_excess": int(frame.date.duplicated().sum()),
        "duplicate_date_rows_all": int(frame.date.duplicated(keep=False).sum()),
        "missing_fields": {c: int(n) for c, n in original.isna().sum().items()},
        "invalid_or_missing_dates": int(frame.date.isna().sum()),
        "missing_fgi_values": int(original.fgi.isna().sum()),
        "invalid_nonmissing_fgi": int((numeric.isna() & original.fgi.notna()).sum()),
        "out_of_range_values": int((numeric.notna() & ~numeric.between(0, 100)).sum()),
        "date_interpretation": "existing daily date label, assigned to NY calendar week; not treated as UTC midnight",
        "source_era_counts": {str(k): int(v) for k, v in frame.source_era.value_counts(dropna=False).items()},
        "source_urls": sorted(frame.source_url.dropna().unique().tolist()), "network_refresh": False,
        "interpolation_or_forward_fill": False}
    save_json(audit, RESULTS / "fgi_daily_audit.json")
    if audit["invalid_or_missing_dates"] or audit["duplicate_date_rows_excess"] or audit["invalid_nonmissing_fgi"] or audit["out_of_range_values"]:
        raise ValueError("FGI validation failed; see fgi_daily_audit.json; no silent deduplication or repair")
    frame["fgi"] = numeric
    assert frame.date.between(START, END).all(), "FGI dates outside approved target"
    frame = frame.sort_values("date")
    # Calendar computation is local; no market prices or new FGI values are fetched.
    # Pad construction so a holiday/weekend endpoint is inside calendar bounds.
    exchange = xcals.get_calendar("XNYS", start=START - pd.Timedelta(days=14), end=END + pd.Timedelta(days=14))
    sessions = exchange.sessions_in_range(START, END)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    dates = pd.date_range(START, END, freq="D")
    daily_calendar = pd.DataFrame({"date": dates, "is_weekday": dates.dayofweek < 5, "is_XNYS_session": dates.isin(sessions)})
    daily_calendar = daily_calendar.merge(frame[["date", "fgi", "source_era"]], on="date", how="left", validate="one_to_one", indicator="record_match")
    daily_calendar["has_daily_record"] = daily_calendar.record_match.eq("both")
    daily_calendar["has_valid_fgi"] = daily_calendar.fgi.notna()
    daily_calendar = daily_calendar.drop(columns="record_match")
    frame["is_XNYS_session"] = frame.date.isin(sessions)
    frame["week_start"] = frame.date - pd.to_timedelta(frame.date.dt.weekday, unit="D")
    valid = frame.loc[frame.fgi.notna()]
    represented_sessions = valid.date.isin(sessions)
    audit.update({"earliest_date": frame.date.min().date().isoformat(), "latest_date": frame.date.max().date().isoformat(),
        "valid_daily_observations": len(valid), "calendar_dates_in_target": len(dates),
        "calendar_dates_represented": int(valid.date.nunique()), "calendar_dates_without_value": int(len(dates) - valid.date.nunique()),
        "weekday_dates_in_target": int((dates.dayofweek < 5).sum()), "weekday_dates_represented": int((valid.date.dt.weekday < 5).sum()),
        "weekend_dates_represented": int((valid.date.dt.weekday >= 5).sum()),
        "XNYS_sessions_in_target": len(sessions), "XNYS_sessions_represented": int(represented_sessions.sum()),
        "XNYS_sessions_without_value": int((daily_calendar.is_XNYS_session & ~daily_calendar.has_valid_fgi).sum()),
        "non_XNYS_dates_with_value": int((~represented_sessions).sum()),
        "calendar_library": "exchange_calendars", "calendar_library_version": importlib.metadata.version("exchange_calendars"),
        "calendar_caveat": "XNYS reference calendar is not an assertion of FGI publication schedule; all available FGI observations retained"})
    save_json(audit, RESULTS / "fgi_daily_audit.json")
    save_csv(daily_calendar, "fgi_calendar_date_audit.csv")
    save_csv(daily_calendar.loc[daily_calendar.is_XNYS_session & ~daily_calendar.has_valid_fgi], "fgi_missing_XNYS_sessions.csv")
    save_csv(frame.loc[~frame.is_XNYS_session], "fgi_non_XNYS_observations.csv")
    save_dataset(frame, "fgi_daily_validated")
    grouped = valid.groupby("week_start", sort=True)
    summary = grouped.agg(FGI_weekly_mean=("fgi", "mean"), FGI_weekly_last=("fgi", "last"),
        FGI_observation_count=("fgi", "count"), FGI_first_observed_date=("date", "min"),
        FGI_last_observed_date=("date", "max"), FGI_weekly_min=("fgi", "min"), FGI_weekly_max=("fgi", "max"),
        FGI_session_observation_count=("is_XNYS_session", "sum"),
        FGI_source_eras=("source_era", lambda x: " | ".join(sorted(set(x)))))
    expected = pd.Series(sessions - pd.to_timedelta(sessions.dayofweek, unit="D")).value_counts()
    weekly = calendar.copy().set_index("week_start").join(summary)
    weekly["FGI_observation_count"] = weekly.FGI_observation_count.fillna(0).astype("int64")
    weekly["FGI_session_observation_count"] = weekly.FGI_session_observation_count.fillna(0).astype("int64")
    weekly["FGI_expected_XNYS_sessions"] = expected.reindex(weekly.index, fill_value=0).astype("int64")
    weekly["FGI_missing_XNYS_sessions"] = weekly.FGI_expected_XNYS_sessions - weekly.FGI_session_observation_count
    weekly["FGI_non_XNYS_observation_count"] = weekly.FGI_observation_count - weekly.FGI_session_observation_count
    weekly["FGI_source_eras"] = weekly.FGI_source_eras.fillna("")
    weekly = weekly.reset_index()
    assert weekly.FGI_observation_count.sum() == len(valid)
    assert weekly.loc[weekly.FGI_observation_count.eq(0), ["FGI_weekly_mean", "FGI_weekly_last"]].isna().all().all()
    save_dataset(weekly, "fgi_weekly_validated")
    save_csv(weekly.loc[weekly.FGI_observation_count.eq(0)], "fgi_weeks_without_observations.csv")
    annual = daily_calendar.assign(year=daily_calendar.date.dt.year).groupby("year").agg(
        calendar_dates=("date", "size"), XNYS_sessions=("is_XNYS_session", "sum"), represented_dates=("has_valid_fgi", "sum"))
    annual["represented_XNYS_sessions"] = daily_calendar.loc[daily_calendar.is_XNYS_session].groupby(daily_calendar.date.dt.year).has_valid_fgi.sum()
    annual["missing_XNYS_sessions"] = annual.XNYS_sessions - annual.represented_XNYS_sessions
    save_csv(annual.reset_index(), "fgi_annual_coverage.csv")
    save_csv(weekly.FGI_observation_count.value_counts().sort_index().rename_axis("observations_per_week").reset_index(name="weeks"),
             "fgi_weekly_observation_distribution.csv")
    return weekly, audit


def merge_diagnostics(wsi, fgi, strategy):
    fgi_columns = ["week_start"] + [c for c in fgi if c.startswith("FGI_")]
    merged = wsi.merge(fgi[fgi_columns], on="week_start", how="outer", validate="one_to_one", indicator="calendar_key_match")
    assert merged.calendar_key_match.eq("both").all()
    a, b = merged.WSI_count.notna(), merged.FGI_weekly_mean.notna()
    merged["validation_pair_status"] = np.select([a & b, a & ~b, ~a & b], ["matched", "WSI_only", "FGI_only"], default="neither_valid")
    paired = merged.loc[a & b]
    row = {"query_strategy": strategy, "common_calendar_week_keys": int(merged.calendar_key_match.eq("both").sum()),
        "unmatched_WSI_calendar_keys": int(merged.calendar_key_match.eq("left_only").sum()),
        "unmatched_FGI_calendar_keys": int(merged.calendar_key_match.eq("right_only").sum()),
        "valid_WSI_weeks": int(a.sum()), "valid_FGI_weeks": int(b.sum()), "matched_value_weeks": len(paired),
        "unmatched_WSI_value_weeks": int((a & ~b).sum()), "unmatched_FGI_value_weeks": int((~a & b).sum()),
        "neither_valid_weeks": int((~a & ~b).sum()),
        "first_matched_week": paired.week_start.min().date().isoformat(),
        "last_matched_week": paired.week_start.max().date().isoformat()}
    merged["calendar_key_match"] = merged.calendar_key_match.astype(str)
    return merged.sort_values("week_start"), row


def protected_inputs():
    return [FGI_PATH, ROOT / "FINAL_PREPROCESSING_POLICY.json", ROOT / "FINAL_PREPROCESSING_POLICY.md",
        ROOT / "results/guardian_text_final/final_preprocessing_policy.json", WSI_FOLDER / "sentiment_model.json",
        *[WSI_FOLDER / f"weekly_{label}_wsi.{extension}" for label in STRATEGIES for extension in ("csv", "parquet")],
        *sorted((ROOT / "data/processed/guardian_text_final/articles").rglob("*.parquet")),
        *sorted((ROOT / "data/processed/guardian_vader/articles").rglob("*.parquet"))]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected_inputs()}
    # The earlier validation is read, not re-executed; no VADER or preprocessing imports.
    previous = json.loads((WSI_FOLDER / "independent_validation.json").read_text(encoding="utf-8"))
    assert previous["status"] == "passed"
    wsi = {strategy: pd.read_csv(WSI_FOLDER / f"weekly_{label}_wsi.csv", parse_dates=["week_start", "week_end"], float_precision="round_trip")
        for label, strategy in STRATEGIES.items()}
    for frame in wsi.values():
        assert len(frame) == 513 and frame.week_start.is_unique and frame.is_complete_week.sum() == 512
        assert frame.week_start.diff().dropna().eq(pd.Timedelta(days=7)).all()
    calendar = next(iter(wsi.values()))[["week_id", "week_start", "week_end", "days_observed", "is_complete_week"]]
    fgi, audit = audit_fgi(calendar)
    merged, diagnostics = {}, []
    for label, strategy in STRATEGIES.items():
        frame, diag = merge_diagnostics(wsi[strategy], fgi, strategy)
        merged[strategy] = frame
        diagnostics.append(diag)
        save_dataset(frame, f"merged_{label}_FGI_weekly")
    merge_summary = pd.DataFrame(diagnostics)
    save_csv(merge_summary, "merge_diagnostics.csv")
    save_csv(pd.concat([frame[["query_strategy", "week_start", "article_count", "FGI_observation_count", "calendar_key_match", "validation_pair_status"]]
                       for frame in merged.values()], ignore_index=True), "merge_week_status.csv")
    correlations = contemporaneous(merged)
    levels, _ = lead_lag_tables(merged)
    changes, delta = lead_lag_tables(merged, differences=True)
    peaks = strongest_lags(pd.concat([levels, changes], ignore_index=True))
    for frame, name in [(correlations, "contemporaneous_correlations.csv"), (levels, "lead_lag_correlations.csv"),
        (changes, "first_difference_robustness.csv"), (changes.loc[changes.lag_weeks.eq(0)], "first_difference_contemporaneous.csv"),
        (delta, "weekly_first_differences.csv"), (peaks, "strongest_lead_lag_correlations.csv")]:
        save_csv(frame, name)
    from guardian_fgi_outputs import make_figures, write_review
    figures, fit = make_figures(merged["broad_business_control"], levels, RESULTS, FIGURES)
    save_json(fit, RESULTS / "scatter_visual_fit.json")
    method = {"period": [START.date().isoformat(), END.date().isoformat()], "week": "America/New_York Monday-Sunday",
        "analysis_scope": "complete weeks; partial boundary retained in files, excluded from analysis",
        "daily_FGI_policy": "existing date labels; all valid observations retained, including non-XNYS dates if any; no interpolation or fill",
        "FGI_primary": "available-observation weekly mean", "FGI_robustness": "last available daily observation by date",
        "correlations": "pairwise complete Pearson; Spearman = Pearson of average ranks within valid pairs",
        "CSV_numeric_precision": "round_trip parser; no rounding or normalization of existing WSI values; preserve exact float ties",
        "lag_definition": "rho(k) = Corr(WSI_w, FGI_(w+k)); positive k compares WSI with future FGI",
        "lag_alignment": "exact 7*k calendar-day date-key match on uncompressed weekly grid; drop missing pairs afterwards",
        "first_differences": "x_w-x_(w-1) on full calendar; missing either adjacent week => missing difference; no fill",
        "standardization": "each series separately, ddof=1, same primary matched complete-week sample; full calendar retained",
        "scatter_fit": "univariate contemporaneous least squares with intercept, visualization only; R-squared is not forecast performance",
        "best_lag_selection": "highest absolute coefficient per series and method; ties prefer smallest abs(k), then k; all ties saved",
        "inference": "descriptive coefficients only; no iid p-values; persistence and multiple lag comparisons limit interpretation",
        "source_provenance_limit": "FGI is a third-party archive with pre-2021 and post-2021 source eras; not ground truth",
        "versions": {p: importlib.metadata.version(p) for p in ("pandas", "numpy", "pyarrow", "matplotlib", "exchange_calendars", "pyluach", "toolz", "korean_lunar_calendar", "tzdata")},
        "network_requests_during_analysis": 0, "VADER_or_preprocessing_rerun": False,
        "market_return_analysis_regression_forecasting_multivariate": False}
    save_json(method, RESULTS / "validation_method.json")
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in protected_inputs()}
    save_json({"input_sha256": hashes, "all_protected_inputs_unchanged": True,
        "status": "pipeline_checks_passed", "figures": figures, "network_requests_during_analysis": 0,
        "market_data_used": False}, RESULTS / "validation_summary.json")
    write_review(ROOT, audit, merge_summary, correlations, levels, changes, peaks, fit, method)
    status_path = ROOT / "results/execution_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["code_status"] = "WSI-FGI validation and calendar-aligned lead-lag / change diagnostics complete; no market-return analysis"
    status["guardian_WSI_FGI_validation"] = {"status": "diagnostics_complete", "FGI_daily_observations": audit["valid_daily_observations"],
        "lags": list(range(-4, 5)), "weekly_FGI": ["mean", "last"], "first_difference_robustness": True,
        "preprocessing_and_WSI_unchanged": True, "market_returns_regression_forecasting_started": False}
    save_json(status, status_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(merge_summary.to_string(index=False))
    print(correlations.to_string(index=False))
    print(peaks.loc[peaks.analysis.eq("levels")].to_string(index=False))
    print("Stopped after WSI-FGI validation; no market returns, market-return regression or forecasting.", flush=True)


if __name__ == "__main__":
    main()
