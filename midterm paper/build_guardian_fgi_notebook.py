"""Saved-results-only notebook for WSI-FGI validation and lead-lag diagnostics."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".packages"))
import nbformat


def main():
    m, c = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
    nb.cells = [
        m("""# Guardian WSI–FGI · 주간 검증
기존 broad/strict count·continuous WSI와 보관한 FGI를 사용했다.
**전처리·VADER·WSI 규칙은 보존한다.** 이 노트북은 저장 결과만 읽는다.
FGI를 정답으로 보거나, 상관을 인과·예측력으로 해석하지 않는다. 시장수익률 분석은 수행하지 않는다."""),
        c('''from pathlib import Path
import sys,json
BASE=Path.cwd().resolve()
if not (BASE/"guardian_fgi_validation.py").exists():
    BASE=BASE/"midterm paper"
sys.path.insert(0,str(BASE/".packages"))
import pandas as pd
from IPython.display import display,Markdown,Image
RESULTS=BASE/"results/guardian_fgi_validation"
FIGURES=BASE/"figures/guardian_fgi_validation"
pd.set_option("display.max_colwidth",140)
method=json.loads((RESULTS/"validation_method.json").read_text(encoding="utf-8"))
print(method["analysis_scope"])
print(method["lag_definition"])'''),
        m("## 1. FGI 날짜·누락 진단\n날짜 라벨은 UTC 자정으로 재해석하지 않았다. 평일과 XNYS 거래일을 구분하고 값은 보간하지 않았다."),
        c('''audit=json.loads((RESULTS/"fgi_daily_audit.json").read_text(encoding="utf-8"))
display(pd.DataFrame([{k:audit[k] for k in ["earliest_date","latest_date","daily_rows","duplicate_date_rows_excess",
    "missing_fgi_values","calendar_dates_in_target","calendar_dates_represented","XNYS_sessions_in_target",
    "XNYS_sessions_represented","XNYS_sessions_without_value"]}]))
display(pd.read_csv(RESULTS/"fgi_annual_coverage.csv"))
display(pd.read_csv(RESULTS/"fgi_weeks_without_observations.csv"))
display(pd.read_csv(RESULTS/"fgi_weekly_observation_distribution.csv"))'''),
        m("## 2. 주간 FGI와 병합\n주 지표는 관측값 평균, 강건성은 마지막 관측값이다. 경계 부분 주·빈 strict 주·FGI 누락 주를 유지했다. 분석은 완전한 주의 유효 쌍을 사용한다."),
        c('''weekly_fgi=pd.read_csv(RESULTS/"fgi_weekly_validated.csv")
display(weekly_fgi.head(10))
display(pd.read_csv(RESULTS/"merge_diagnostics.csv"))
unmatched=pd.read_csv(RESULTS/"merge_week_status.csv")
display(unmatched.loc[~unmatched.validation_pair_status.eq("matched")])'''),
        m("## 3. 동시적 상관과 시각화\nSpearman은 유효 쌍에서 동률 평균 순위를 사용한다. 두 지표를 각각 표준화하며 ddof=1이다. 산점도 직선은 시각화용이고 R²는 예측력이 아니다."),
        c('''display(pd.read_csv(RESULTS/"contemporaneous_correlations.csv"))
for name in ["wsi_fgi_overlay_0_100.png","wsi_fgi_overlay_standardized.png","wsi_fgi_scatter.png"]:
    display(Image(filename=str(FIGURES/name),width=1150 if "overlay" in name else 800))
print((RESULTS/"scatter_visual_fit.json").read_text(encoding="utf-8"))'''),
        m("""## 4. 시차 상관
`rho(k) = Corr(WSI_w, FGI_(w+k))`, k = −4,…,+4.
양수는 현재 WSI와 미래 FGI, 음수는 현재 WSI와 이전 FGI를 비교한다.
결측 주를 삭제해 시간축을 압축하지 않고 정확히 7×k일 날짜 키를 맞춘다.
최대 절대상관 시차는 인과·정보전달·예측 가능성을 입증하지 않는다."""),
        c('''lags=pd.read_csv(RESULTS/"lead_lag_correlations.csv")
display(lags.loc[lags.query_strategy.eq("broad_business_control") & lags.WSI_metric.eq("WSI_count")])
display(Image(filename=str(FIGURES/"wsi_fgi_lead_lag.png"),width=1000))
peaks=pd.read_csv(RESULTS/"strongest_lead_lag_correlations.csv")
display(peaks.loc[peaks.analysis.eq("levels")])'''),
        m("## 5. 1차 차분 강건성\n수준 분석을 대체하지 않는다. 현재 주 또는 직전 달력 주가 결측이면 차분도 결측이다."),
        c('''display(pd.read_csv(RESULTS/"first_difference_contemporaneous.csv"))
display(pd.read_csv(RESULTS/"first_difference_robustness.csv"))
display(peaks.loc[peaks.analysis.eq("first_differences")])'''),
        m("## 6. 독립 검증과 중단 지점\n자료 계보 차이·누락·시계열 의존성과 다중 시차 비교를 보존해 보고한다. 시장수익률 회귀·예측·다변량 모형은 수행하지 않았다."),
        c('''print((RESULTS/"independent_validation.json").read_text(encoding="utf-8"))
display(Markdown((BASE/"GUARDIAN_WSI_FGI_VALIDATION_REVIEW.md").read_text(encoding="utf-8")))'''),
    ]
    nbformat.validate(nb)
    path = ROOT / "guardian_fgi_validation_review.ipynb"
    nbformat.write(nb, path)
    print(f"Created {path.name}: {len(nb.cells)} cells", flush=True)
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(nb, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}})
        client.on_cell_start = lambda cell, cell_index, **kwargs: print(f"WSI-FGI review {cell_index + 1}/{len(nb.cells)}", flush=True)
        client.execute()
        nbformat.write(nb, path)
        print("Executed and saved read-only WSI-FGI review notebook", flush=True)


if __name__ == "__main__":
    main()
