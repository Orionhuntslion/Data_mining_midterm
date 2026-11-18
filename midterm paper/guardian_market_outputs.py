"""Two report-ready figure designs and a numerical/technical handoff only."""
import numpy as np
import pandas as pd


def make_figures(merged, correlations, ols, results, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    primary = merged.loc[merged.is_complete_week & merged.broad_WSI_count.notna() & merged.weekly_log_return.notna()]
    fit = ols.loc[ols.predictor.eq("broad_WSI_count") & ols.relationship.eq("contemporaneous")].iloc[0]
    corr = correlations.loc[correlations.predictor.eq("broad_WSI_count") & correlations.relationship.eq("contemporaneous")].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(primary.broad_WSI_count, 100 * primary.weekly_log_return, s=24, alpha=0.48, color="#21618c", edgecolors="none")
    x = np.linspace(primary.broad_WSI_count.min(), primary.broad_WSI_count.max(), 100)
    ax.plot(x, 100 * (fit.alpha + fit.beta * x), color="#a34b00", linewidth=1.8, label="Exploratory contemporaneous OLS")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.text(0.04, 0.96, f"n = {int(fit.n_observations)}\nPearson r = {corr.pearson:.3f}\nR² = {fit.R_squared:.3f}",
        transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.set(title="Broad count WSI and S&P 500 weekly log return", xlabel="Broad WSI_count (0–100)",
        ylabel="Weekly log return × 100", xlim=(0, 100))
    ax.grid(alpha=0.15); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    paths = []
    for extension in ("png", "svg"):
        name = "market_broad_count_scatter." + extension
        fig.savefig(folder / name, dpi=200); paths.append(name)
    plt.close(fig)
    mean, std = primary.broad_WSI_count.mean(), primary.broad_WSI_count.std(ddof=1)
    plot = merged[["week_start", "is_complete_week", "weekly_log_return", "broad_WSI_count"]].copy()
    mask = plot.is_complete_week & plot.weekly_log_return.notna() & plot.broad_WSI_count.notna()
    plot["broad_WSI_count_z"] = ((plot.broad_WSI_count - mean) / std).where(mask)
    plot["weekly_return_x100"] = (plot.weekly_log_return * 100).where(plot.is_complete_week)
    plot.to_csv(results / "market_timeseries_plot_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"series": "broad_WSI_count", "standardization_sample": "contemporaneous matched complete weeks",
        "n": len(primary), "mean": mean, "std_ddof1": std}]).to_csv(results / "market_plot_standardization.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    axes[0].plot(plot.week_start, plot.weekly_return_x100, color="#21618c", linewidth=0.9)
    axes[0].set(title="S&P 500 weekly log return", ylabel="Log return × 100")
    axes[1].plot(plot.week_start, plot.broad_WSI_count_z, color="#a34b00", linewidth=0.9)
    axes[1].set(title="Broad count WSI · standardized separately", ylabel="WSI z-score", xlabel="America/New_York Monday week start")
    for ax in axes:
        ax.axhline(0, color="gray", linewidth=0.7); ax.grid(alpha=0.15)
    fig.tight_layout()
    for extension in ("png", "svg"):
        name = "market_weekly_returns_wsi." + extension
        fig.savefig(folder / name, dpi=200); paths.append(name)
    plt.close(fig)
    return paths


def table(frame, columns):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.7g}" if pd.notna(v) else "NA"
        return str(v)
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"] +
        ["| " + " | ".join(cell(row[c]) for c in columns) + " |" for _, row in frame.iterrows()])


def write_handoff(root, audit, method, diagnostics, correlations, ols, compact, figures):
    # Deliberately restricted to the seven handoff categories requested by the user.
    lines = ["# 최종 탐색 분석 · 기술 인계", "", "## 데이터 출처", "",
        "- FRED `SP500`, 원출처 S&P Dow Jones Indices LLC: [자료 설명](https://fred.stlouisfed.org/series/SP500).",
        f"- 다운로드: {audit['download_url']}; 원응답 SHA-256 `{audit['raw_sha256']}`.",
        "- 일별 종가 가격지수. 배당 포함 총수익지수가 아니며 별도 adjusted close 필드는 없다. 임의 수정종가를 만들지 않았다.",
        "- FRED 기본 공개 CSV를 보관하고 2017-01-01~2026-10-25를 로컬 선택했다. 다른 가격 출처를 이어 붙이지 않았다.", "",
        "## 방법·수식", "",
        "- 뉴욕 월요일~일요일 달력 유지. 분석은 완전한 주만 사용하며, 부분 경계 주는 파일에 유지한다.",
        "- P_w = 주 안의 마지막 유효 거래일 종가; r_w = ln(P_w/P_(w-1)). 단위는 소수 로그수익률이다.",
        "- 직전 **달력 주** 종가가 없으면 수익률은 NA. 기간 밖 초기 가격·보간·전방 채움은 사용하지 않는다.",
        "- 동시점: r_w = alpha + beta*x_w + epsilon_w. 탐색적 다음 주: r_(w+1) = alpha + beta*x_w + epsilon_w.",
        "- 다음 주는 정확한 week_start+7일로 연결하고 그 후 결측 쌍을 제외한다. 각 적합은 설명변수 하나와 절편만 사용한다.",
        "- x는 broad/strict count·continuous WSI 또는 맥락 비교 FGI_weekly_mean. 기존 값·생성 규칙은 변경하지 않는다.",
        "- SE(beta)=sqrt([SSE/(n-2)]/Sxx); t=beta/SE(beta); p=2*t_df=n-2.sf(|t|). 모두 conventional OLS 기준이다.",
        "- R²=1-SSE/SST; adjusted R²=1-(1-R²)*(n-1)/(n-2). beta는 지수 1포인트당 소수 로그수익률이다.",
        "- Pearson 및 동률 평균 순위 Spearman을 각 명세의 유효 쌍에서 계산한다. CSV round_trip 정밀도를 사용한다.",
        "- 그림만 수익률에 100을 곱한다. 시계열 WSI z=(x-mean)/std(ddof=1), 동시점 유효 쌍 표본 기준이다.", "",
        "## 사용 패키지", "",
        "- " + "; ".join(f"{name} {version}" for name, version in method["versions"].items()) + ".",
        "- 노트북: nbformat, nbclient, ipykernel, jupyter_client, IPython. 검증 이미지 읽기: Pillow.",
        "- 표준 라이브러리: pathlib, hashlib, json, datetime, sys, importlib.metadata; 검증용 statistics/math.",
        "- 통계 함수 근거: [SciPy 단순 선형 적합](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html).", "",
        "## 표본 크기·자료 진단", "",
        f"- 유효 일별 종가 {audit['valid_daily_observations']:,}개; {audit['first_available_date']}~{audit['last_available_date']}.",
        f"- 기간 내 날짜 중복 {audit['duplicate_dates_in_period_excess']}; 유효 가격 자료의 결측 {audit['missing_prices_in_validated_dataset']}.",
        f"- 원응답의 기간 내 행 {audit['in_period_source_rows']:,}개 중 가격 공란 {audit['missing_prices_in_period_source_rows']}개 "
        f"(비거래일 공란 {audit['holiday_blank_source_rows']}). 예상 XNYS 거래일 {audit['expected_XNYS_sessions']:,}일; 누락 거래일 가격 {audit['missing_XNYS_session_prices']}개.",
        "- 달력 513주, 완전한 주 512주. 첫 완전 주는 기간 내 직전 주 종가가 없어 수익률이 결측이다.", "",
        table(diagnostics, ["predictor", "relationship", "valid_paired_weeks", "predictor_only_weeks", "return_only_weeks",
            "first_matched_predictor_week", "last_matched_predictor_week"]), "",
        "## 수치 결과", "",
        table(correlations, ["predictor", "relationship", "n_pairs", "pearson", "spearman"]), "",
        table(ols, ["predictor", "relationship", "n_observations", "alpha", "beta", "se_beta", "t_beta", "p_value_beta_two_sided", "R_squared", "adjusted_R_squared"]), "",
        "### Broad·FGI 비교표", "",
        table(compact, ["predictor", "relationship", "n_pairs", "pearson", "spearman", "beta", "p_value_beta_two_sided", "R_squared"]), "",
        "## 생성 파일", "",
        "- `data/raw/market/fred_sp500_2017-01-01_2026-10-25.csv`, `.metadata.json`, `source_access_log.json`: 원응답·출처·접근 기록.",
        "- `results/guardian_market_analysis/sp500_daily_validated.{csv,parquet}`: 유효 일별 종가.",
        "- `results/guardian_market_analysis/sp500_weekly_returns.{csv,parquet}`: 주간 종가·로그수익률·관측 수.",
        "- `results/guardian_market_analysis/merged_market_WSI_weekly.{csv,parquet}`: 분리된 WSI 열·FGI·동시/다음 주 수익률.",
        "- `market_correlations.csv`, `market_OLS_results.csv`, `market_benchmark_comparison.csv`: 상관·OLS·맥락 비교.",
        "- `market_merge_diagnostics.csv`, `market_pair_status.csv`: 유효/비매칭 표본과 날짜.",
        "- `market_daily_audit.json`, `market_calendar_audit.csv`, `market_missing_source_rows.csv`, `market_missing_XNYS_sessions.csv`: 자료 진단.",
        "- `market_timeseries_plot_data.csv`, `market_plot_standardization.csv`: 그림 자료와 표준화 기준.",
        "- `market_method.json`, `validation_summary.json`, `independent_validation.json`: 방법·패키지·입력 해시·독립 검증.",
        "- 위 확장자 없는 결과 파일명 기준 폴더: `results/guardian_market_analysis/`.",
        *["- `figures/guardian_market_analysis/" + name + "`." for name in figures],
        "- `guardian_market_analysis_review.ipynb`: 실행된 저장 결과 검토 노트북.",
        "- 코드: `guardian_market_data.py`, `guardian_market_analysis.py`, `guardian_market_stats.py`, `guardian_market_outputs.py`, `verify_guardian_market_analysis.py`, `build_guardian_market_notebook.py`.", "",
        "## 한계·주의", "",
        "- 탐색적 동시점/다음 주 연관성만 계산했다. 인과·예측 능력·거래 수익성·시장 효율성 위반의 증거가 아니다.",
        "- Conventional 표준오차·p값은 자기상관·이분산에 강건하지 않다. 여러 명세를 비교했으며 다중검정 보정은 하지 않았다.",
        "- 통계적 유의성과 경제적 중요성은 같지 않다. 작은 R²를 예측력으로 해석하지 않는다.",
        "- WSI 측정오차·기사 밀도 차이·두 전략의 중첩이 있다. FGI는 정답이 아니며 기존 자료 계보·누락을 유지했다.",
        "- 모델마다 결측 제거 후 표본 날짜·수가 다를 수 있다. 위 비교표는 공통 표본으로 강제하지 않았으며 n을 병기했다.",
        "- 가격지수 수익률은 배당 재투자를 포함하지 않는다. 원자료 재배포 조건은 FRED 출처 페이지를 확인한다.",
        "- 전처리·VADER·WSI·FGI 자료와 이전 결과를 수정하지 않았다. 예측·ML·다변량·추가 모형은 수행하지 않았다.", ""]
    (root / "GUARDIAN_MARKET_HANDOFF.md").write_text("\n".join(lines), encoding="utf-8")
