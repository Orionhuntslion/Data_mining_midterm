"""07. Pearson/Spearman 및 달력 날짜 기준 선행·후행 관계. 인과 추정 아님."""
import numpy as np
import pandas as pd

from midterm_common import INDEX_COLUMNS, read_csv, save_csv, save_json


def pair_stats(x, y):
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).replace([np.inf,-np.inf], np.nan).dropna()
    row = {"n_overlap": len(pair), "pearson": np.nan, "spearman": np.nan, "status": "ok"}
    if len(pair) < 3:
        row["status"] = "fewer_than_3_pairs"
    elif pair.x.nunique() < 2 or pair.y.nunique() < 2:
        row["status"] = "constant_series"
    else:
        row["pearson"] = pair.x.corr(pair.y, method="pearson")
        row["spearman"] = pair.x.corr(pair.y, method="spearman")
    return row


def lag_table(x, fgi, max_lag, **extra):
    """rho(k)=Corr(x_t, FGI_{t+k}); 결측을 버리기 전에 날짜를 이동한다."""
    rows = []
    for lag in range(-max_lag, max_lag+1):
        future = fgi.reindex(x.index + pd.Timedelta(days=lag))
        future.index = x.index
        rows.append({**extra, "lag": lag, **pair_stats(x, future)})
    return pd.DataFrame(rows)


def run(cfg):
    daily = read_csv(cfg.processed / "merged_daily_data.csv", ["date"]).set_index("date").reindex(cfg.calendar)
    mask = daily.n_docs.ge(cfg.min_daily_docs)
    contemporaneous = []
    sensitivity = []
    lags = []
    # 1일 차분은 실제로 인접한 달력 날짜만. 금요일→월요일을 1일로 취급하지 않는다.
    fgi_diff = daily.fgi.diff()
    for col in INDEX_COLUMNS:
        x = daily[col].where(mask)
        contemporaneous.append({"index": col, **pair_stats(x, daily.fgi)})
        lags.append(lag_table(x, daily.fgi, cfg.max_lag, index=col, specification="levels"))
        lags.append(lag_table(x.diff(), fgi_diff, cfg.max_lag, index=col, specification="calendar_first_difference"))
        for threshold in sorted({1, 10, cfg.min_daily_docs, 50}):
            sensitivity.append({"index": col, "min_daily_docs": threshold,
                **pair_stats(daily[col].where(daily.n_docs.ge(threshold)), daily.fgi)})
    lag_result = pd.concat(lags, ignore_index=True)
    save_csv(pd.DataFrame(contemporaneous), cfg.results / "correlations.csv")
    save_csv(pd.DataFrame(sensitivity), cfg.results / "correlation_sensitivity.csv")
    save_csv(lag_result, cfg.results / "lag_correlations.csv")

    words = read_csv(cfg.results / "word_daily_patterns.csv", ["date"])
    word_lags = []
    for term, group in words.groupby("term", sort=True):
        group = group.set_index("date").reindex(cfg.calendar)
        x = group.document_share.where(group.n_docs.ge(cfg.min_daily_docs))
        word_lags.append(lag_table(x, daily.fgi, cfg.max_lag, term=term))
    save_csv(pd.concat(word_lags, ignore_index=True), cfg.results / "word_lag_correlations.csv")
    save_json({"formula": "rho(k) = Corr(WSI_t, FGI_(t+k))", "unit": "calendar days",
        "positive_k": "web sentiment compared with future FGI; possible web lead association",
        "negative_k": "web sentiment compared with past FGI; possible FGI lead association",
        "min_daily_docs": cfg.min_daily_docs, "missing_dates": "not interpolated; exact date matches only",
        "statistics": "descriptive associations; no IID p-values, predictive or causal claims",
        "cautions": ["serial dependence", "multiple lag/word comparisons", "FGI close vs full-day headlines",
                     "date-varying source mix", "GDELT first-seen time differs from publication",
                     "capped date-desc sample is not random and may favor late-day news"]}, cfg.results / "association_method.json")
    print(pd.DataFrame(contemporaneous).to_string(index=False))
    if pd.DataFrame(contemporaneous).n_overlap.max() < 10:
        print("주의: 비교 가능한 날짜가 10일 미만입니다. 수집 범위/누락/표본 기준을 확인하세요.")
    return lag_result
