"""WSI-FGI comparison figures and a descriptive review, without causal claims."""
import numpy as np
import pandas as pd
from guardian_fgi_stats import correlations


def make_figures(broad, lags, results, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    source = broad.set_index("week_start").sort_index()
    observed = source[["WSI_count", "FGI_weekly_mean"]].where(source.is_complete_week, np.nan)
    paired = observed.dropna()
    means, stds = paired.mean(), paired.std(ddof=1)
    standardized = ((observed - means) / stds).where(observed.notna().all(axis=1), np.nan)
    standardized.rename(columns={c: c + "_z" for c in standardized}).reset_index().to_csv(
        results / "standardized_overlay_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"series": means.index, "mean": means.values, "std_ddof1": stds.values,
                  "paired_complete_weeks": len(paired)}).to_csv(results / "standardization_parameters.csv", index=False, encoding="utf-8-sig")
    paths = []
    for frame, title, ylabel, filename in [
        (observed, "Broad count WSI and weekly mean FGI · observed levels", "Index (0–100)", "wsi_fgi_overlay_0_100.png"),
        (standardized, "Broad count WSI and weekly mean FGI · separate z-scores", "z-score (sample std)", "wsi_fgi_overlay_standardized.png"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(frame.index, frame.WSI_count, label="Broad WSI_count", color="#2471a3", linewidth=1.05)
        ax.plot(frame.index, frame.FGI_weekly_mean, label="FGI_weekly_mean", color="#b45f06", alpha=0.9, linewidth=1.05)
        ax.set(title=title, xlabel="NY Monday week start", ylabel=ylabel)
        if frame is observed:
            ax.set_ylim(0, 100)
        else:
            ax.axhline(0, color="gray", linewidth=0.7)
        ax.grid(alpha=0.2); ax.legend(loc="upper right")
        fig.tight_layout(); fig.savefig(figures / filename, dpi=170); plt.close(fig); paths.append(filename)
    x, y = paired.FGI_weekly_mean.to_numpy(), paired.WSI_count.to_numpy()
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    intercept = float(y.mean() - slope * x.mean())
    fitted = intercept + slope * x
    r_squared = float(1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
    corr = correlations(paired.WSI_count, paired.FGI_weekly_mean)
    fit = {"n_pairs": len(paired), "x": "FGI_weekly_mean", "y": "broad WSI_count", "slope": slope,
        "intercept": intercept, "pearson_r": corr["pearson"], "R_squared": r_squared,
        "purpose": "contemporaneous scatter visualization only; not predictive power"}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(x, y, s=22, alpha=0.45, color="#2471a3", edgecolors="none")
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, intercept + slope * grid, color="#b45f06", label="Contemporaneous linear fit (visual only)")
    ax.text(0.04, 0.96, f"n={len(paired)}\nPearson r={corr['pearson']:.3f}\nR²={r_squared:.3f}",
        transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.set(title="Broad count WSI vs weekly mean FGI", xlabel="FGI_weekly_mean", ylabel="Broad WSI_count", xlim=(0, 100), ylim=(0, 100))
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.15)
    fig.tight_layout(); name = "wsi_fgi_scatter.png"; fig.savefig(figures / name, dpi=170); plt.close(fig); paths.append(name)
    primary = lags.loc[lags.query_strategy.eq("broad_business_control") & lags.WSI_metric.eq("WSI_count")].sort_values("lag_weeks")
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, color in [("pearson", "#2471a3"), ("spearman", "#b45f06")]:
        ax.plot(primary.lag_weeks, primary[method], marker="o", label=method.title(), color=color)
    ax.axhline(0, color="gray", linewidth=0.8); ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set(title="Broad count WSI vs weekly mean FGI · level lead-lag association", xticks=range(-4, 5),
        xlabel="k in Corr(WSI[w], FGI[w+k]) · k > 0 compares WSI with future FGI", ylabel="Correlation")
    ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout(); name = "wsi_fgi_lead_lag.png"; fig.savefig(figures / name, dpi=170); plt.close(fig); paths.append(name)
    return paths, fit


def table(frame, columns):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}" if pd.notna(v) else "NA"
        return str(v)
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"] +
        ["| " + " | ".join(cell(row[c]) for c in columns) + " |" for _, row in frame.iterrows()])


def write_review(root, audit, merge, contemporaneous, lags, changes, peaks, fit, method):
    lines = ["# Guardian WSI–FGI 주간 검증 진단", "",
        "기존 broad(주 자료)·strict(강건성)의 count/continuous WSI를 수정 없이 사용했다. VADER 및 전처리를 재실행하지 않았다.",
        "이 단계는 텍스트 기반 지표와 시장 기반 FGI의 동시적·시간적 연관성을 기술한다. FGI를 정답으로 가정하지 않는다.", "",
        "## FGI 날짜·자료 검증", "",
        f"- 기존 역사 CSV: {audit['earliest_date']}~{audit['latest_date']}, {audit['daily_rows']:,}행 / 유효 관측 {audit['valid_daily_observations']:,}개.",
        f"- 날짜 중복 초과 행 {audit['duplicate_date_rows_excess']}, 결측 FGI {audit['missing_fgi_values']}, 잘못된 날짜 {audit['invalid_or_missing_dates']}, 0~100 범위 밖 값 {audit['out_of_range_values']}.",
        f"- 연구기간 전체 달력 날짜 {audit['calendar_dates_in_target']:,}일 중 관측 날짜 {audit['calendar_dates_represented']:,}일; 값 없는 날짜 {audit['calendar_dates_without_value']:,}일.",
        f"- 평일 {audit['weekday_dates_in_target']:,}일 중 {audit['weekday_dates_represented']:,}일 관측; 주말 관측 {audit['weekend_dates_represented']}일.",
        f"- exchange_calendars {audit['calendar_library_version']} XNYS 기준 거래 세션 {audit['XNYS_sessions_in_target']:,}일 중 {audit['XNYS_sessions_represented']:,}일 관측; 값 없는 세션 {audit['XNYS_sessions_without_value']}일.",
        f"- XNYS 비거래일 관측 {audit['non_XNYS_dates_with_value']}일. 거래 달력은 참고 기준이며 FGI 발행 달력과 동일하다고 가정하지 않는다. 관측값을 달력 때문에 삭제하지 않았다.", "",
        "입력은 이미 날짜 라벨이므로 UTC 자정으로 간주해 전날로 이동시키지 않았다. 뉴욕 월~일 주에 배정했다.",
        "이전 `results/fgi_manifest.json`은 초기 수집 파일의 기록이다. 이번 검증의 행 수·날짜·해시는 실제 사용한 역사 CSV에서 새로 산출했다.",
        f"FGI 입력 SHA-256: `{audit['sha256']}`.", "",
        "자료는 제3자 아카이브이며 source_era 계보를 그대로 보존했다. 2021년 전후 계보 구분은 공식 지수 방법론 변경의 확정이 아니다.",
        "source_era별 행 수: " + "; ".join(f"{k}: {v:,}" for k, v in audit["source_era_counts"].items()) + ".", "",
        "## 주간 FGI와 병합", "",
        "주 지표는 주 안의 모든 유효 관측값 평균(FGI_weekly_mean), 강건성은 날짜순 마지막 유효 값(FGI_weekly_last)이다.",
        "주별 관측 수·처음/마지막 날짜·거래 세션 누락 수·자료 계보를 저장했다. 일별 값 보간, 주간 값 전방 채움은 없다.",
        "FGI가 통째로 비어 있는 완전한 주의 시작일: " + ", ".join(pd.read_csv(root / "results/guardian_fgi_validation/fgi_weeks_without_observations.csv").loc[lambda d: d.is_complete_week, "week_start"]) + ".",
        "2020년 누락 거래 세션은 22일이고 2026년은 1일이다. 누락 일자와 주별 관측 수는 별도 CSV에 보존했다.",
        "513개 달력 주와 경계 부분 주를 파일에 유지했다. 기술분석은 512개 완전한 주를 기준으로 유효 쌍만 사용했다.",
        "아래 matched는 양쪽에 유효 값이 있는 주이다. 공통 달력 키 수는 전략별 513개이며, 값이 없는 주와 키가 없는 주를 구분했다.", "",
        table(merge, ["query_strategy", "matched_value_weeks", "unmatched_WSI_value_weeks", "unmatched_FGI_value_weeks", "neither_valid_weeks", "first_matched_week", "last_matched_week"]), "",
        "Strict의 빈 주와 FGI 누락 주는 그대로 남아 있다. unmatched 목록은 merge_week_status.csv에서 확인할 수 있다.", "",
        "## 동시적 상관", "",
        "Pearson은 값의 상관, Spearman은 유효 쌍 안에서 동률에 평균 순위를 부여한 순위 상관이다.",
        "CSV는 round_trip 정밀도로 읽어 저장된 실수·동률을 보존했다. 기존 WSI 값을 반올림하거나 규칙을 변경하지 않았다.",
        "아래 계수는 기술통계이다. 시계열 의존성을 무시한 iid p-value는 보고하지 않았다.", "",
        table(contemporaneous, ["query_strategy", "WSI_metric", "FGI_metric", "n_pairs", "pearson", "spearman"]), "",
        f"주 산점도의 단순 동시적 직선 적합: n={fit['n_pairs']}, Pearson r={fit['pearson_r']:.4f}, R²={fit['R_squared']:.4f}.",
        "이 직선은 시각화 목적이며 R²를 예측력으로 해석하지 않는다.",
        "표준화 그림은 동일한 주 분석 유효 쌍에서 각 시계열을 자체 평균·표본 표준편차(ddof=1)로 z 변환했다. 결측 주를 연결·보간하지 않았다.", "",
        "## 시차 상관", "",
        "정의: rho(k)=Corr(WSI_w, FGI_(w+k)), k=-4,...,+4.",
        "k>0은 현재 WSI와 이후 FGI, k<0은 현재 WSI와 이전 FGI를 비교한다.",
        "완전한 주간 달력의 정확한 7*k일 날짜 키를 맞춘 후 결측 쌍을 제외했다. 결측을 먼저 삭제한 뒤 행을 이동하지 않았다.",
        "모든 36개 시차·전략·WSI 조합의 유효 쌍 수와 두 계수는 lead_lag_correlations.csv에 저장했다.",
        "각 계수의 절댓값이 가장 큰 시차는 다음과 같다. 동률은 |k|가 작은 시차, 그 다음 k가 작은 시차를 표시하며 전체 동률 목록도 저장했다.", "",
        table(peaks.loc[peaks.analysis.eq("levels")], ["query_strategy", "WSI_metric", "correlation_method", "best_lag_weeks", "correlation", "n_pairs"]), "",
        "최대 상관 시차는 시간적 연관성의 기술 결과이다. 비영 시차만으로 인과관계·정보전달·예측 가능성을 입증하지 않는다.",
        "자기상관·공통 추세·관측 수 차이와 여러 시차의 비교를 고려해야 하며, 단일 최고점의 의미를 과장하지 않는다.", "",
        "## 1차 차분 강건성", "",
        "Delta x_w=x_w-x_(w-1)을 원래 주간 달력 위에서 계산했다. 이웃한 어느 주라도 결측이면 차분도 결측이며 간격을 뛰어넘지 않았다.",
        "FGI_weekly_mean에 대한 차분의 동시적 상관은 다음과 같다. 수준 분석을 대체하지 않는다.", "",
        table(changes.loc[changes.lag_weeks.eq(0)], ["query_strategy", "WSI_metric", "n_pairs", "pearson", "spearman"]), "",
        "차분의 전체 -4~+4 시차 결과는 first_difference_robustness.csv, 최대 절대상관은 strongest_lead_lag_correlations.csv에 있다.", "",
        "## 파일과 중단 지점", "",
        "- results/guardian_fgi_validation/: 검증 일별·주별 FGI, broad/strict 병합 CSV·Parquet, 병합/달력 진단, 수준/차분 상관, 모델 방법·해시.",
        "- figures/guardian_fgi_validation/: 0~100 중첩, z-score 중첩, 산점도, 주 명세 시차 그림.",
        "- guardian_fgi_validation_review.ipynb: 저장 결과만 읽는 검토 노트북.", "",
        "```powershell", "python guardian_fgi_validation.py", "python verify_guardian_fgi_validation.py",
        "python build_guardian_fgi_notebook.py --execute", "```", "",
        "기존 FGI, WSI와 확정 전처리·채점 자료는 덮어쓰지 않았다. 시장수익률 분석·시장수익률 회귀·예측·다변량 모형은 수행하지 않았다.",
        "외부 서버에서 FGI를 다시 받지 않았으며, 이번 결과는 보관한 자료 스냅샷에 대한 것이다.", "",
        "계산 근거: [동률 평균 순위](https://pandas.pydata.org/docs/reference/api/pandas.Series.rank.html), "
        "[XNYS 세션 달력](https://github.com/gerrymanoim/exchange_calendars).", ""]
    (root / "GUARDIAN_WSI_FGI_VALIDATION_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
