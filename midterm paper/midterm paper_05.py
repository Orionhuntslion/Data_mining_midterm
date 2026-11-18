"""05. 세 가지 일별 지수와 결합 진단. 무관측일/FGI 휴장일은 보간하지 않는다."""
import numpy as np
import pandas as pd

from midterm_common import INDEX_COLUMNS, read_csv, save_csv, save_json


def build_daily(frame, cfg):
    work = frame.assign(is_positive=frame.sentiment_label.eq("positive").astype(int),
                        is_negative=frame.sentiment_label.eq("negative").astype(int))
    daily = work.groupby("date").agg(n_docs=("compound", "size"), mean_sentiment=("compound", "mean"),
        median_sentiment=("compound", "median"), std_sentiment=("compound", "std"),
        n_positive=("is_positive", "sum"), n_negative=("is_negative", "sum"))
    daily = daily.reindex(cfg.calendar)
    for col in ["n_docs", "n_positive", "n_negative"]:
        daily[col] = daily[col].fillna(0).astype(int)
    daily["n_neutral"] = daily.n_docs - daily.n_positive - daily.n_negative
    total = daily.n_docs.replace(0, np.nan)
    daily["positive_share"] = daily.n_positive / total
    daily["negative_share"] = daily.n_negative / total
    daily["neutral_share"] = daily.n_neutral / total
    daily["WSI_continuous"] = 50 * (daily.mean_sentiment + 1)
    daily["WSI_count"] = 50 * (1 + (daily.n_positive - daily.n_negative) / total)
    # Bernoulli의 분모는 양성+음성 문서. 중립 문서는 제외한다.
    daily["n_binary"] = daily.n_positive + daily.n_negative
    n_binary = daily.n_binary.replace(0, np.nan)
    daily["p_hat"] = daily.n_positive / n_binary
    daily["WSI_binary"] = 100 * daily.p_hat
    daily["binary_se_iid"] = np.sqrt(daily.p_hat * (1 - daily.p_hat) / n_binary)
    daily["normal_approx_ok"] = daily.n_positive.ge(10) & daily.n_negative.ge(10)
    # Wilson 95% 구간. 문서들이 독립 Bernoulli라는 작업 가정 아래에서만 유효하다.
    z = 1.959963984540054
    center = (daily.p_hat + z*z/(2*n_binary)) / (1 + z*z/n_binary)
    half = z * np.sqrt(daily.p_hat*(1-daily.p_hat)/n_binary + z*z/(4*n_binary*n_binary)) / (1+z*z/n_binary)
    daily["binary_wilson_low_iid"] = 100 * (center-half).clip(0, 1)
    daily["binary_wilson_high_iid"] = 100 * (center+half).clip(0, 1)
    daily["eligible_day"] = daily.n_docs.ge(cfg.min_daily_docs)
    for col in INDEX_COLUMNS:
        assert daily[col].dropna().between(0, 100).all()
    assert daily.n_docs.sum() == len(frame)
    return daily.reset_index()


def run(cfg):
    scored = read_csv(cfg.processed / "scored_text_data.csv", ["date"])
    fgi = read_csv(cfg.processed / "fear_greed_data.csv", ["date"])
    daily = build_daily(scored, cfg)
    save_csv(daily, cfg.processed / "daily_web_sentiment.csv")
    merged = daily.merge(fgi, how="left", on="date", validate="one_to_one", indicator=True)
    merged["has_fgi"] = merged._merge.eq("both")
    merged = merged.drop(columns="_merge")
    merged["analysis_eligible"] = merged.eligible_day & merged.has_fgi
    save_csv(merged, cfg.processed / "merged_daily_data.csv")
    docs = scored.merge(fgi[["date", "fgi", "regime"]], how="left", on="date", validate="many_to_one")
    save_csv(docs, cfg.processed / "regime_text_data.csv")
    stats = docs.groupby("regime", dropna=False).agg(n_docs=("doc_id", "size"),
        mean_sentiment=("compound", "mean"), median_sentiment=("compound", "median"),
        std_sentiment=("compound", "std"), skewness=("compound", "skew"),
        excess_kurtosis=("compound", lambda x: x.kurt()),
        positive_share=("sentiment_label", lambda x: x.eq("positive").mean()),
        negative_share=("sentiment_label", lambda x: x.eq("negative").mean())).reset_index()
    save_csv(stats, cfg.results / "regime_summary.csv")
    diagnostics = {"calendar_days": len(daily), "days_with_text": int(daily.n_docs.gt(0).sum()),
        "low_sample_days": int((daily.n_docs.gt(0) & ~daily.eligible_day).sum()),
        "fgi_days": len(fgi), "matched_days_with_text": int((merged.has_fgi & merged.n_docs.gt(0)).sum()),
        "eligible_matched_days": int(merged.analysis_eligible.sum()),
        "texts_without_fgi_regime": int(docs.regime.isna().sum()),
        "missing_fgi_dates": merged.loc[~merged.has_fgi, "date"].dt.strftime("%Y-%m-%d").tolist(),
        "no_text_dates": daily.loc[daily.n_docs.eq(0), "date"].dt.strftime("%Y-%m-%d").tolist(),
        "binary_note": "Neutrals excluded; IID Bernoulli uncertainty is illustrative. Syndication/dependence can invalidate it.",
        "alignment_note": "Daily labels in America/New_York; news may appear after FGI close. No intraday predictive claim."}
    save_json(diagnostics, cfg.results / "merge_diagnostics.json")
    print(pd.Series({k: v for k,v in diagnostics.items() if not isinstance(v,list)}).to_string())
    return merged
