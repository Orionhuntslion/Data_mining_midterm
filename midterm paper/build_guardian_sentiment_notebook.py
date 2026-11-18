"""Build/execute a read-only review of saved VADER/weekly WSI diagnostics."""
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
        m("""# Guardian VADER · 주간 WSI 검토
확정 전처리 `guardian_text_v2_final`을 보존하고 `text_sentiment`를 VADER 3.3.2로 평가했다.
**Broad = primary, strict = robustness.** 두 표본을 합친 지표는 만들지 않는다.

이 노트북은 저장 결과만 읽는다. 모델 재실행, 전처리 변경, API 요청, FGI 병합·시차·수익률 분석은 수행하지 않는다."""),
        c('''from pathlib import Path
import json,sys
BASE=Path.cwd().resolve()
if not (BASE/"guardian_sentiment_analysis.py").exists():
    BASE=BASE/"midterm paper"
sys.path.insert(0,str(BASE/".packages"))
import pandas as pd
from IPython.display import display,Markdown,Image
RESULTS=BASE/"results/guardian_sentiment"
FIGURES=BASE/"figures/guardian_sentiment"
pd.set_option("display.max_colwidth",160)
model=json.loads((RESULTS/"sentiment_model.json").read_text(encoding="utf-8"))
print(model["package"],model["version"],model["input"],model["input_policy"])
print("Inclusive thresholds:",model["negative_threshold_inclusive"],model["positive_threshold_inclusive"])
print("Frozen policy SHA-256:",model["input_policy_sha256"])'''),
        m("## 1. 기사별 점수·클래스 분포\nstd는 ddof=1, 첨도는 Fisher 초과첨도(정규분포 0)이다. 분류를 수동 보정하지 않았다."),
        c('''display(pd.read_csv(RESULTS/"article_descriptive_statistics.csv"))
display(pd.read_csv(RESULTS/"sentiment_class_distribution.csv"))
display(Image(filename=str(FIGURES/"compound_score_histogram.png"),width=1100))
display(Image(filename=str(FIGURES/"sentiment_class_distribution.png"),width=1100))'''),
        m("""## 2. 주간 감성과 두 WSI
뉴욕 월~일 주를 보존한다. 경계의 불완전 주를 포함하면 513주, 완전한 주는 512주이다.
기사 0건인 주는 비율·감성·지표·구간이 결측이며 보간하지 않았다.

`WSI_count = 50 × [1 + (positive_count − negative_count) / article_count]`가 주 명세다.
`WSI_continuous = 50 × (1 + mean_compound)`는 연속 점수 강건성 명세다."""),
        c('''broad=pd.read_csv(RESULTS/"weekly_broad_sentiment.csv")
strict=pd.read_csv(RESULTS/"weekly_strict_sentiment.csv")
display(broad.head(10))
display(strict.loc[strict.article_count.eq(0)])
stats=pd.read_csv(RESULTS/"weekly_descriptive_statistics.csv")
display(stats.loc[stats.scope.eq("complete_weeks")])
display(pd.read_csv(RESULTS/"broad_strict_wsi_comparison.csv"))'''),
        m("""## 3. 주별 지표와 근사 표본 구간
Count 지표 그림의 음영은 multinomial plug-in SE에 기반한 ±1.96 SE 구간이다.
독립성을 보장하지 않으며, 적은 기사 수·같은 감성 범주만 관찰된 주에서는 과소 추정할 수 있다.
0폭 구간은 확실성의 증거가 아니다. 구간은 식 그대로 저장해 0~100 밖으로 나갈 수 있다.
Broad와 strict에는 중첩 기사가 있다. 두 표본을 독립 표본으로 해석하지 않는다."""),
        c('''display(Image(filename=str(FIGURES/"weekly_wsi_count.png"),width=1200))
display(Image(filename=str(FIGURES/"weekly_wsi_continuous.png"),width=1200))
display(stats.loc[stats.scope.eq("complete_weeks") & stats.metric.eq("WSI_count"),
    ["query_strategy","weeks_lt3_articles","weeks_lt5_articles","degenerate_intervals","intervals_outside_0_100"]])'''),
        m("""## 4. 오류 검사 후보 · 자동 재분류 없음
강한 긍정·부정, 중립 근처, 욕설·슬랭(관측된 경우), 방향 단어와 점수의 의심 조합을 검사한다.
휴리스틱은 오류를 확정하지 않는다. 범용 VADER의 금융 표현·풍자·밈 처리 능력을 가정하지 않는다.
`manual_error_assessment`는 비워 두었다."""),
        c('''display(pd.read_csv(RESULTS/"inspection_candidate_inventory.csv"))
sample=pd.read_csv(RESULTS/"vader_inspection_sample.csv")
print("Inspection articles:",len(sample))
display(sample[["title","text_sentiment","compound","sentiment_class","inspection_reasons","manual_error_assessment"]].head(20))'''),
        m("## 5. 보존·검증·중단 지점\n전체 기사 원본 열·확정 정책 보존, 전 기사 재채점 교차 검증, 집계·수식·결측 확인. FGI·시장 자료는 읽지 않았다."),
        c('''print((RESULTS/"independent_validation.json").read_text(encoding="utf-8"))
display(Markdown((BASE/"GUARDIAN_SENTIMENT_REVIEW.md").read_text(encoding="utf-8")))'''),
    ]
    nbformat.validate(nb)
    path = ROOT / "guardian_sentiment_review.ipynb"
    nbformat.write(nb, path)
    print(f"Created {path.name}: {len(nb.cells)} cells", flush=True)
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(nb, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}})
        client.on_cell_start = lambda cell, cell_index, **kwargs: print(f"Sentiment review {cell_index + 1}/{len(nb.cells)}", flush=True)
        client.execute()
        nbformat.write(nb, path)
        print("Saved executed read-only sentiment review notebook", flush=True)


if __name__ == "__main__":
    main()
