"""Read-only final quantitative-stage review; all tables and plots are saved results."""
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
        m("""# 최종 탐색 분석 · WSI와 S&P 500 주간 수익률
저장된 수치·그림 검토 전용이다. 전처리·VADER·WSI·FGI를 변경하거나 재실행하지 않는다.
모든 OLS는 설명변수 하나와 절편만 사용한다. 예측·ML·다변량·추가 분석은 수행하지 않는다."""),
        c('''from pathlib import Path
import json,sys
BASE=Path.cwd().resolve()
if not (BASE/"guardian_market_analysis.py").exists():
    BASE=BASE/"midterm paper"
sys.path.insert(0,str(BASE/".packages"))
import pandas as pd
from IPython.display import display,Image,Markdown
RESULTS=BASE/"results/guardian_market_analysis"
FIGURES=BASE/"figures/guardian_market_analysis"
pd.set_option("display.max_colwidth",130)
pd.set_option("display.float_format",lambda x:f"{x:.7g}")
method=json.loads((RESULTS/"market_method.json").read_text(encoding="utf-8"))
print(method["weekly_return"])
print(method["OLS"])'''),
        m("## 1. 출처·가격 검증\nFRED SP500 / S&P Dow Jones Indices. 배당 미포함 가격지수이며 별도 수정종가는 없다. 원응답 공휴일 공란과 실제 거래일 누락을 구분한다."),
        c('''audit=json.loads((RESULTS/"market_daily_audit.json").read_text(encoding="utf-8"))
display(pd.DataFrame([{k:audit[k] for k in ["source","first_available_date","last_available_date","valid_daily_observations",
    "duplicate_dates_in_period_excess","missing_prices_in_validated_dataset","missing_prices_in_period_source_rows",
    "holiday_blank_source_rows","missing_XNYS_session_prices"]}]))
display(pd.read_csv(RESULTS/"sp500_daily_validated.csv").head())
display(pd.read_csv(RESULTS/"market_missing_XNYS_sessions.csv"))'''),
        m("## 2. 주간 수익률·병합 진단\n월~일 마지막 종가의 연속 달력 주 로그비율. 첫 주의 이전 종가와 마지막 주의 다음 수익률은 만들지 않는다. Strict 빈 주도 유지한다."),
        c('''weekly=pd.read_csv(RESULTS/"sp500_weekly_returns.csv",float_precision="round_trip")
display(weekly.head(8))
display(pd.read_csv(RESULTS/"market_merge_diagnostics.csv"))'''),
        m("## 3. 동시점 상관·OLS\n수익률은 소수 로그 단위, beta는 지수 1포인트당 수익률이다. 표준오차·t·p는 conventional OLS이며 자기상관·이분산에 강건하지 않다."),
        c('''corr=pd.read_csv(RESULTS/"market_correlations.csv",float_precision="round_trip")
ols=pd.read_csv(RESULTS/"market_OLS_results.csv",float_precision="round_trip")
display(corr.loc[corr.relationship.eq("contemporaneous")])
display(ols.loc[ols.relationship.eq("contemporaneous")])
display(Image(filename=str(FIGURES/"market_broad_count_scatter.png"),width=1000))'''),
        m("## 4. 탐색적 다음 주 관계·FGI 맥락 비교\nWSI_w와 r_(w+1)을 정확한 다음 달력 주로 연결한다. 예측 모형이나 예측 능력의 증거로 부르지 않는다. FGI는 정답이 아니며 각 명세의 n이 다를 수 있다."),
        c('''display(corr.loc[corr.relationship.eq("exploratory_next_week")])
display(ols.loc[ols.relationship.eq("exploratory_next_week")])
display(pd.read_csv(RESULTS/"market_benchmark_comparison.csv",float_precision="round_trip"))'''),
        m("## 5. 수익률·표준화 WSI 시계열\n단위를 분리한 두 패널이다. 수익률은 그림에서만 100배, WSI 표준화는 동시점 유효 쌍의 표본 표준편차(ddof=1)를 사용한다."),
        c('''display(Image(filename=str(FIGURES/"market_weekly_returns_wsi.png"),width=1250))
display(pd.read_csv(RESULTS/"market_plot_standardization.csv"))'''),
        m("## 6. 독립 검증·기술 인계\n기존 자료·결과 해시, 원응답 가격, 주간 수식·다음 주 정렬, 모든 단변량 OLS와 상관을 확인한다. 추가 분석은 수행하지 않는다."),
        c('''print((RESULTS/"independent_validation.json").read_text(encoding="utf-8"))
display(Markdown((BASE/"GUARDIAN_MARKET_HANDOFF.md").read_text(encoding="utf-8")))'''),
    ]
    nbformat.validate(nb)
    path = ROOT / "guardian_market_analysis_review.ipynb"
    nbformat.write(nb, path)
    print(f"Created {path.name}: {len(nb.cells)} cells", flush=True)
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(nb, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}})
        client.on_cell_start = lambda cell, cell_index, **kwargs: print(f"Market review {cell_index + 1}/{len(nb.cells)}", flush=True)
        client.execute()
        nbformat.write(nb, path)
        print("Executed and saved final market review notebook; no analysis rerun", flush=True)


if __name__ == "__main__":
    main()
