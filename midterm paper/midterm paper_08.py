"""08. 과목 필수 그림과 RQ별 비교 그림을 PNG로 저장한다."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from wordcloud import WordCloud

from midterm_common import INDEX_COLUMNS, REGIMES, read_csv, save_csv

COLORS = ["#c44747", "#dd8c52", "#b2a55a", "#4e9b80", "#347ca2"]


def run(cfg):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.figsize": (11,4.5), "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 160, "font.size": 10})
    cfg.figures.mkdir(exist_ok=True)
    artifacts = []

    def finish(fig, name, caption):
        fig.tight_layout()
        path = cfg.figures / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        artifacts.append({"file": path.name, "caption": caption})

    daily = read_csv(cfg.processed / "merged_daily_data.csv", ["date"])
    docs = read_csv(cfg.processed / "regime_text_data.csv", ["date"])
    freq = read_csv(cfg.results / "word_frequencies.csv")
    regime_freq = read_csv(cfg.results / "regime_word_frequencies.csv")
    compare = read_csv(cfg.results / "fear_greed_word_comparison.csv")
    tfidf = read_csv(cfg.results / "regime_tfidf.csv")
    lags = read_csv(cfg.results / "lag_correlations.csv")

    fig, ax = plt.subplots()
    ax.bar(daily.date, daily.n_docs, color="#347ca2")
    ax.axhline(cfg.min_daily_docs, color="#c44747", ls="--", label="Analysis sample threshold")
    ax.set(xlabel=f"Date ({cfg.timezone})", ylabel="Deduplicated documents", title="Daily document coverage")
    ax.legend()
    finish(fig, "01_daily_document_count", "중복 제거 후 날짜별 문서 수; 0은 뉴스 없는 날짜")

    fig, ax = plt.subplots()
    top = freq.head(cfg.top_n).iloc[::-1]
    ax.barh(top.term, top["count"], color="#347ca2")
    ax.set(xlabel="Token count", title="Most frequent words")
    fig.set_size_inches(10, max(5, len(top)*0.24))
    finish(fig, "02_top_words", "전 기간 상위 단어; 은어·욕설을 자동 삭제하지 않음")

    def cloud(series, title, name):
        fig, ax = plt.subplots(figsize=(11,5))
        if len(series) and series.sum() > 0:
            wc = WordCloud(width=1400, height=650, background_color="white", collocations=False,
                           random_state=cfg.seed, stopwords=set()).generate_from_frequencies(series.to_dict())
            ax.imshow(wc, interpolation="bilinear")
        else:
            ax.text(0.5,0.5,"No observations in this regime",ha="center",transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(title)
        finish(fig, name, title)

    cloud(freq.set_index("term")["count"], "Overall word cloud", "03_wordcloud_overall")
    for i, regime in enumerate(REGIMES):
        table = regime_freq.loc[regime_freq.regime.eq(regime)]
        cloud(table.set_index("term")["count"], f"Word cloud: {regime}", f"04_wordcloud_{i+1}_{regime.lower().replace(' ','_')}")

    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].hist(docs.compound, bins=np.linspace(-1,1,42), color="#347ca2", alpha=0.8)
    axes[0].set(xlabel="VADER compound", ylabel="Documents", title="Sentiment distribution")
    if len(docs) > 2 and docs.compound.nunique() > 1:
        grid = np.linspace(-1,1,250)
        axes[1].plot(grid, gaussian_kde(docs.compound)(grid), color="#347ca2")
    axes[1].set(xlabel="VADER compound", ylabel="Density", title="KDE (smoothed, boundary bias possible)")
    finish(fig, "05_sentiment_distribution", "문서별 compound 분포; 왜도를 행동편향의 증거로 해석하지 않음")

    fig, ax = plt.subplots()
    for regime, color in zip(REGIMES, COLORS):
        values = docs.loc[docs.regime.eq(regime), "compound"]
        if len(values):
            ax.hist(values,bins=np.linspace(-1,1,25),density=True,histtype="step",label=f"{regime} (n={len(values)})",color=color)
    ax.set(xlabel="VADER compound",ylabel="Density",title="Sentiment by FGI regime")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    finish(fig, "06_regime_distributions", "관측된 FGI 국면별 문서 감성 분포")

    fig, ax = plt.subplots()
    for col in INDEX_COLUMNS:
        ax.plot(daily.date,daily[col],label=col,alpha=0.85)
    ax.set(ylim=(0,100),xlabel="Date",ylabel="Web sentiment (0–100)",title="Daily Web Sentiment Indices")
    ax.legend()
    finish(fig, "07_sentiment_trend", "전체 일별 지수; 분석 표본 기준 미달일도 표시")

    fig, ax = plt.subplots()
    ax.plot(daily.date,daily.WSI_count,label="WSI count (primary)",color="#347ca2")
    ax.plot(daily.date,daily.fgi,label="CNN FGI",color="#dd8c52")
    ax.set(ylim=(0,100),xlabel="Date",ylabel="Index (0–100)",title="Web sentiment and Fear & Greed")
    ax.legend()
    finish(fig,"08_wsi_fgi_timeseries","같은 날짜에 정렬; 누락/휴장일을 연결 값으로 대체하지 않음")

    eligible = daily.loc[daily.analysis_eligible].copy()
    fig, ax = plt.subplots()
    for col,color in [("WSI_count","#347ca2"),("fgi","#dd8c52")]:
        std = eligible[col].std()
        if len(eligible)>1 and std>0:
            z=(daily[col]-eligible[col].mean())/std
            ax.plot(daily.date,z,label=col,color=color)
    ax.set(xlabel="Date",ylabel="Z-score",title="Standardized series (moments from eligible matched days)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    finish(fig,"09_standardized_timeseries","적격 공통 날짜의 평균/표준편차로 각각 표준화")

    fig, ax = plt.subplots()
    ax.scatter(eligible.fgi,eligible.WSI_count,c="#347ca2",alpha=0.7)
    ax.set(xlim=(0,100),ylim=(0,100),xlabel="CNN FGI",ylabel="WSI count (primary)",title=f"Same-date association (eligible days n={len(eligible)})")
    finish(fig,"10_scatter_wsi_fgi","같은 날 연관성; 인과/예측 성능이 아님")

    fig, axes = plt.subplots(1,2,figsize=(13,4.5))
    for ax,spec in zip(axes,["levels","calendar_first_difference"]):
        for col in INDEX_COLUMNS:
            group=lags.loc[lags["index"].eq(col)&lags.specification.eq(spec)]
            ax.plot(group.lag,group.pearson,marker="o",label=col)
        ax.axvline(0,color="gray",ls="--")
        ax.axhline(0,color="gray",lw=0.8)
        ax.set(xlabel="k (calendar days); +k = web vs future FGI",ylabel="Pearson correlation",ylim=(-1,1),title=spec)
    axes[0].legend(fontsize=8)
    finish(fig,"11_lag_correlations","rho(k)=Corr(WSI_t,FGI_(t+k)); 수준/달력 1일 차분 비교")

    fig, axes = plt.subplots(1,2,figsize=(13,6))
    for ax, group,color in zip(axes,["Fear","Greed"],[COLORS[0],COLORS[-1]]):
        table=compare.loc[compare.group.eq(group)].head(20).iloc[::-1]
        ax.barh(table.term,table.per_1000_tokens,color=color)
        ax.set(title=f"{group} (including extreme)",xlabel="Occurrences per 1,000 tokens")
    finish(fig,"12_fear_vs_greed_words","표본 크기 차이를 완화하기 위해 토큰 1,000개당 빈도로 표시")

    fig, axes = plt.subplots(1,5,figsize=(20,6))
    for ax,regime,color in zip(axes,REGIMES,COLORS):
        table=tfidf.loc[tfidf.regime.eq(regime)].head(12).iloc[::-1]
        ax.barh(table.term,table.mean_tfidf,color=color)
        ax.set(title=regime,xlabel="Mean document TF-IDF")
    finish(fig,"13_regime_tfidf","공통 어휘집으로 계산한 국면별 평균 문서 TF-IDF")

    words=read_csv(cfg.results / "word_daily_patterns.csv",["date"])
    selected=read_csv(cfg.results / "selected_timing_terms.csv").term.head(6).tolist()
    fig, axes=plt.subplots(3,2,figsize=(14,9),sharex=True)
    for ax,term in zip(axes.flat,selected):
        group=words.loc[words.term.eq(term)]
        ax.plot(group.date,group.document_share,color="#347ca2")
        ax.set(title=term,ylabel="Share of documents",ylim=(0,1))
        second=ax.twinx()
        second.plot(daily.date,daily.fgi,color="#dd8c52",alpha=0.35)
        second.set_ylim(0,100)
        second.set_ylabel("FGI",color="#dd8c52")
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    fig.autofmt_xdate()
    finish(fig,"14_word_timing","문서빈도로 미리 선택한 상위 단어; 왼쪽 문서비율, 오른쪽 FGI")
    save_csv(pd.DataFrame(artifacts),cfg.results / "figure_manifest.csv")
    print(f"그림 {len(artifacts)}개 저장: {cfg.figures}")
    return artifacts
