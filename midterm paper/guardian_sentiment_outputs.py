"""Saved VADER results: plots, transparent manual-inspection candidates and review."""
import re

import numpy as np
import pandas as pd
from guardian_weekly_sentiment import FLAGS, CLASSES

LABELS = {"broad_business_control": "Broad", "anchors_stock_market_tag": "Strict"}


def inspection_sample(articles, folder):
    frame = articles.sort_values("guardian_id").copy()
    text = frame.text_sentiment
    slang_pattern = r"\b(?:fuck(?:ing|ed)?|shit|bullshit|bastard|damn|crap|fomo|yolo|stonks|hodl|tendies|rekt)\b|diamond hands|to the moon"
    bullish_pattern = r"\b(?:rall(?:y|ies|ied)|surge[sd]?|soar(?:s|ed|ing)?|bullish|rebound(?:s|ed)?|record high)\b"
    bearish_pattern = r"\b(?:plummet(?:s|ed|ing)?|slump(?:s|ed)?|crash(?:es|ed)?|bearish|sell[- ]off|tumble[sd]?)\b"
    bullish = text.str.contains(bullish_pattern, flags=re.I, regex=True)
    bearish = text.str.contains(bearish_pattern, flags=re.I, regex=True)
    slang = text.str.contains(slang_pattern, flags=re.I, regex=True)
    suspicious = (frame.vader_compound.eq(0) & (bullish | bearish)) | (bullish & frame.sentiment_class.eq("negative")) | (bearish & frame.sentiment_class.eq("positive"))
    cohorts = [
        ("strongly_positive", frame.vader_compound.ge(0.75), "compound >= 0.75", 12),
        ("strongly_negative", frame.vader_compound.le(-0.75), "compound <= -0.75", 12),
        ("near_neutral", frame.vader_compound.abs().le(0.10), "abs(compound) <= 0.10, including threshold-adjacent labels", 12),
        ("profanity_or_slang", slang, slang_pattern, 15),
        ("potential_suspicious_classification", suspicious,
         "direction-word heuristic vs zero/opposite compound; candidate only, not a verified error", 20),
    ]
    reasons, inventory = {}, []
    for i, (name, mask, rule, limit) in enumerate(cohorts):
        pool = frame.loc[mask]
        chosen = pool.sample(n=min(limit, len(pool)), random_state=42 + i)
        for identifier in chosen.guardian_id:
            reasons.setdefault(identifier, []).append(name)
        inventory.append({"inspection_category": name, "candidate_articles": len(pool), "sampled_articles": len(chosen),
                          "selection_rule": rule, "random_seed": 42 + i})
    result = frame.loc[frame.guardian_id.isin(reasons)].copy()
    result["inspection_reasons"] = result.guardian_id.map(lambda x: " | ".join(reasons[x]))
    result["title"] = result.headline_for_text
    result["compound"] = result.vader_compound
    result["publication_date"] = result.date_et
    result["manual_error_assessment"] = ""
    columns = ["guardian_id", "title", "description", "text_sentiment", "compound", "sentiment_class",
        "published_at_utc", "published_at_et", "publication_date", *FLAGS.values(), "inspection_reasons", "manual_error_assessment"]
    result[columns].to_csv(folder / "vader_inspection_sample.csv", index=False, encoding="utf-8-sig")
    # Preserve all unusual-text candidates, even if the small inspection sample omits some.
    unusual = frame.loc[slang | suspicious].copy()
    unusual["profanity_or_slang_candidate"] = slang.loc[unusual.index]
    unusual["direction_heuristic_candidate"] = suspicious.loc[unusual.index]
    unusual[["guardian_id", "headline_for_text", "description", "text_sentiment", "vader_compound", "sentiment_class",
        "date_et", *FLAGS.values(), "profanity_or_slang_candidate", "direction_heuristic_candidate"]].to_csv(
            folder / "unusual_text_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inventory).to_csv(folder / "inspection_candidate_inventory.csv", index=False, encoding="utf-8-sig")
    return {"sample_unique_articles": len(result), "all_unusual_candidates": len(unusual), "cohorts": inventory,
            "automatic_relabeling": False, "heuristics_are_not_ground_truth": True}


def make_figures(articles, weekly, class_stats, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for ax, (strategy, flag) in zip(axes, FLAGS.items()):
        subset = articles.loc[articles[flag]]
        ax.hist(subset.vader_compound, bins=np.linspace(-1, 1, 41), color="#3274a1", edgecolor="white")
        ax.axvline(-0.05, color="#b33638", linestyle="--", linewidth=1)
        ax.axvline(0.05, color="#438752", linestyle="--", linewidth=1)
        ax.set(title=f"{LABELS[strategy]} · n={len(subset):,}", xlabel="VADER compound", ylabel="Articles", xlim=(-1, 1))
    fig.tight_layout()
    path = folder / "compound_score_histogram.png"
    fig.savefig(path, dpi=170); plt.close(fig); paths.append(path.name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, strategy in zip(axes, FLAGS):
        rows = class_stats.loc[class_stats.query_strategy.eq(strategy)].set_index("sentiment_class").loc[list(CLASSES)]
        bars = ax.bar(rows.index, rows["count"], color=["#438752", "#969696", "#b33638"])
        ax.bar_label(bars, labels=[f"{n:,}\n({s:.1%})" for n, s in zip(rows["count"], rows.share)], padding=4)
        ax.set(title=LABELS[strategy], ylabel="Articles", ylim=(0, rows["count"].max() * 1.25))
    fig.tight_layout()
    path = folder / "sentiment_class_distribution.png"
    fig.savefig(path, dpi=170); plt.close(fig); paths.append(path.name)
    for metric in ("WSI_count", "WSI_continuous"):
        fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
        for ax, strategy in zip(axes, FLAGS):
            subset = weekly.loc[weekly.query_strategy.eq(strategy)].sort_values("week_start")
            dates = pd.to_datetime(subset.week_start)
            ax.plot(dates, subset[metric], linewidth=0.9, color="#21618c", label=metric)
            if metric == "WSI_count":
                ax.fill_between(dates, subset.WSI_count_ci95_low, subset.WSI_count_ci95_high,
                    alpha=0.16, color="#21618c", label="Approx. 95% plug-in sampling interval")
            else:
                ax.set_ylim(-3, 103)
            ax.axhline(50, color="gray", linewidth=0.8, linestyle="--")
            ax.set(title=LABELS[strategy], ylabel="WSI", xlabel="NY Monday week start")
            ax.legend(loc="lower left", fontsize=8)
            ax.grid(alpha=0.15)
        fig.suptitle(f"{metric} · missing weeks kept as gaps; no interpolation", fontsize=12)
        fig.tight_layout()
        path = folder / f"weekly_{metric.lower()}.png"
        fig.savefig(path, dpi=170); plt.close(fig); paths.append(path.name)
    return paths


def markdown_table(frame, columns):
    def format_cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}" if pd.notna(v) else "NA"
        return str(v)
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"] +
        ["| " + " | ".join(format_cell(row[c]) for c in columns) + " |" for _, row in frame.iterrows()])


def write_report(root, model, article_stats, weekly_stats, comparison, validation):
    lines = ["# Guardian VADER 및 주간 WSI 진단", "",
        "VADER 3.3.2를 확정된 `guardian_text_v2_final`의 `text_sentiment` 전체에 기사별로 한 번씩 적용했다.",
        "전처리 정책, text_raw/text_sentiment/text_clean, 원본 메타데이터와 소속 플래그는 수정하지 않았다.",
        "동일 기사에는 한 점수만 부여하고 두 전략의 소속을 유지한다. 파일럿 relevance 라벨은 사용하지 않았다.", "",
        f"물리적 기사 {validation['physical_articles']:,}건; broad {validation['articles_by_strategy']['broad_business_control']:,}건; "
        f"strict {validation['articles_by_strategy']['anchors_stock_market_tag']:,}건; 중첩 {validation['overlap_articles']:,}건.", "",
        "## 기사별 분포", "",
        "positive: compound >= 0.05; negative: compound <= -0.05; 나머지는 neutral. 수동 보정은 없다.",
        "표준편차는 ddof=1, 왜도는 pandas 보정 표본 왜도, 첨도는 Fisher 초과첨도(정규분포 0)이다. Pearson 첨도도 CSV에 병기했다.", "",
        markdown_table(article_stats, ["query_strategy", "article_count", "mean_compound", "median_compound", "std_compound",
            "min_compound", "max_compound", "skewness", "excess_kurtosis"]), "",
        markdown_table(article_stats, ["query_strategy", "positive_count", "positive_share", "neutral_count", "neutral_share", "negative_count", "negative_share"]), "",
        "## 주간 구조와 두 WSI", "",
        "America/New_York 월요일~일요일 주를 그대로 사용한다. 513개 경계 포함 주 중 완전한 주는 512개이다.",
        "2016-12-26~2017-01-01 경계 주는 연구기간 내 하루만 포함하는 부분 주이며 기사 0건이다. 삭제하지 않고 표시했다.",
        "기본 기술통계는 complete_weeks 기준이며 all_calendar_weeks 결과도 저장했다.",
        "기사 0건인 주는 count=0, 비율/평균/중앙값/표준편차/WSI/SE/구간은 NA이다. 단일 기사 주의 표준편차도 NA이다.", "",
        "주 지표: B=(P-N)/(P+N+U), WSI_count=50*(1+B). 강건성 지표: WSI_continuous=50*(1+mean_compound).",
        "두 지표 모두 0~100이다. count 지표에서 50은 긍정·부정 수가 균형이거나 전부 중립인 경우이다.", "",
        markdown_table(weekly_stats.loc[weekly_stats.scope.eq("complete_weeks")],
            ["query_strategy", "metric", "calendar_weeks", "valid_weeks", "missing_weeks", "mean", "median", "std", "min", "max"]), "",
        "## Broad와 strict의 기술적 비교", "",
        "두 전략을 하나의 지표로 합치지 않았다. 아래 상관은 두 Guardian WSI 사이에서만 계산했으며 FGI와의 상관이 아니다.",
        "두 표본에 중첩 기사가 있으므로 독립 표본 비교로 해석하지 않는다.", "",
        markdown_table(comparison, ["scope", "metric", "paired_valid_weeks", "broad_strict_pearson_correlation"]), "",
        "## 근사 표본 불확실성", "",
        "SE(B)=sqrt([p_positive+p_negative-(p_positive-p_negative)^2]/n); SE(WSI)=50*SE(B).",
        "구간은 WSI_count ± 1.96*SE(WSI)인 **근사 표본 구간**이다. 요청된 식 그대로 저장해 0~100 밖으로 나갈 수도 있다.",
        "기사 간 독립성은 보장하지 않는다. 동일 사건·반복 설명·live-blog 등으로 실제 불확실성을 과소평가할 수 있다.",
        "적은 기사 수와 한 범주만 관찰된 주에서는 정규근사가 부적절하거나 SE=0이 될 수 있다. 0폭 구간은 확실성의 증거가 아니다.",
        "이 구간은 VADER 오분류, 표본 관련성, 기사 의존성을 포함한 전체 오차구간이 아니다. lt3/lt5 및 degenerate 플래그를 저장했다.", "",
        markdown_table(weekly_stats.loc[weekly_stats.scope.eq("complete_weeks") & weekly_stats.metric.eq("WSI_count")],
            ["query_strategy", "weeks_lt3_articles", "weeks_lt5_articles", "degenerate_intervals", "intervals_outside_0_100"]), "",
        "## 수동 오류 검사 표본", "",
        f"고유 기사 {validation['inspection_inventory']['sample_unique_articles']}건을 `vader_inspection_sample.csv`로 저장했다.",
        "강한 긍정/부정(±0.75), 중립 근처(|compound|<=0.10), 욕설·슬랭, 방향 단어와 점수의 의심 조합을 고정 시드로 추출했다.",
        "의심 조합은 검사 후보를 찾는 휴리스틱이며 오류 판정이나 정답 라벨이 아니다. 원 점수·분류는 그대로 보존했다.",
        "모든 특이 텍스트 후보는 `unusual_text_candidates.csv`에 추가 보존했다. 각 후보 규칙·전체 수·추출 수는 inventory CSV에 있다.",
        "VADER는 일반 어휘·규칙 기반 모형이다. 금융 약어, 상승 맥락의 욕설, 풍자, 밈과 도메인 표현을 완벽히 이해한다고 가정하지 않는다.", "",
        "## 결과물과 실행", "",
        "- 기사 Parquet: `data/processed/guardian_vader/articles/year=YYYY/part-00000.parquet`",
        "- 주간 broad/strict 감성·WSI: `results/guardian_sentiment/weekly_{broad,strict}_{sentiment,wsi}.{csv,parquet}`",
        "- 기사·주간 통계, 클래스 분포, 전략 간 비교, 모델·입출력 해시: `results/guardian_sentiment/`",
        "- 점수 히스토그램·클래스 분포·주간 WSI 두 그림: `figures/guardian_sentiment/`",
        "- 검토 노트북: `guardian_sentiment_review.ipynb`", "",
        "```powershell", "python guardian_sentiment_analysis.py", "python verify_guardian_sentiment.py",
        "python build_guardian_sentiment_notebook.py --execute", "```", "",
        "원자료와 정책 해시, 모든 기존 열과 점수·분류, 집계·수식·결측·표본 보존을 별도 검증 스크립트로 확인한다.",
        "FGI를 읽거나 병합하지 않았고 WSI-FGI 상관, 시차, 시장 수익률, 회귀 및 예측은 수행하지 않았다. 경제적 의미도 해석하지 않았다.", "",
        "모형 및 임계값 근거: [VADER 공식 저장소](https://github.com/cjhutto/vaderSentiment/blob/master/README.rst).", ""]
    (root / "GUARDIAN_SENTIMENT_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
