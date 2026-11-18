"""최종 전처리 비교·정책·워드클라우드 검토 노트북. 저장 결과만 읽는다."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_full_collection.py"))
import nbformat


def main():
    m=nbformat.v4.new_markdown_cell;c=nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook()
    nb.metadata.kernelspec={"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"}
    nb.cells=[
        m("""# 최종 전처리 명세 · 동결 결과 검토
**guardian_text_v2_final · Broad/Strict weekly**

추가 불용어는 **happened, latest, coverage, rolling, live, news** 여섯 개뿐이다.
`new`, `year`, `business`, `says`는 유지한다. 표제어화는 비활성화이다.
이 노트북은 저장된 결과만 읽으며 감성모델·WSI·FGI·시차·수익률 분석을 실행하지 않는다."""),
        c('''from pathlib import Path
import runpy,json
BASE=Path.cwd().resolve()
if not (BASE/"guardian_full_collection.py").exists():
    BASE=BASE/"midterm paper"
runpy.run_path(str(BASE/"guardian_full_collection.py"))
import pandas as pd
from IPython.display import display,Markdown,Image
RESULTS=BASE/"results/guardian_text_final"
pd.set_option("display.max_colwidth",100)
STRATEGY="broad_business_control"  # anchors_stock_market_tag로 변경 가능
policy=json.loads((BASE/"FINAL_PREPROCESSING_POLICY.json").read_text(encoding="utf-8"))
print(policy["status"],policy["version"])
print("Policy SHA-256:",policy["policy_sha256"])'''),
        m("## 1. Clean 토큰·고유 어휘 변경 전후\n원자료와 v1 처리 자료는 보존했다. 삭제 수는 두 전략 각각 계산하므로 합집합에 더하지 않는다."),
        c('''display(pd.read_csv(RESULTS/"custom_stopword_before_after.csv"))
display(pd.read_csv(RESULTS/"custom_stopword_removed_counts.csv"))'''),
        m("## 2. Top 30 전후와 신규 진입\n새 단어를 추가한 것이 아니라 여섯 단어 제외 후 기존 단어 순위가 올라간 결과이다."),
        c('''before=pd.read_csv(RESULTS/"top30_before.csv")
after=pd.read_csv(RESULTS/"top30_after.csv")
paired=before.loc[before.query_strategy.eq(STRATEGY)].merge(after.loc[after.query_strategy.eq(STRATEGY)],on=["query_strategy","rank"],suffixes=("_before","_after"))
display(paired)
display(pd.read_csv(RESULTS/"top30_new_entries.csv"))'''),
        m("## 3. 금융 어휘와 잔여 상용구\n경제 어휘의 빈도는 그대로 보존한다. 반복 템플릿에 남은 경제 단어도 추가 삭제 없이 보고한다."),
        c('''financial=pd.read_csv(RESULTS/"financial_vocabulary_preservation.csv")
display(financial.loc[financial.query_strategy.eq(STRATEGY)])
display(pd.read_csv(RESULTS/"phrase_artifact_review.csv"))
display(pd.read_csv(RESULTS/"repeated_description_templates.csv"))
display(pd.read_csv(RESULTS/"common_bigrams.csv").loc[lambda t:t.query_strategy.eq(STRATEGY)].head(30))'''),
        m("## 4. 전체 기간과 대표 연도 워드클라우드\n2017·2020·2026 패널과 모든 연도의 개별 결과를 저장했다. 2026년은 10월 25일까지이다."),
        c('''for label in ["broad","strict"]:
    display(Markdown(f"**{label} · full period**"))
    display(Image(filename=str(BASE/f"figures/guardian_text_final/wordcloud_{label}_full_period.png"),width=1100))
display(Image(filename=str(BASE/"figures/guardian_text_final/wordcloud_representative_years.png"),width=1300))
display(pd.read_csv(RESULTS/"representative_year_word_frequencies.csv").loc[lambda t:t.query_strategy.eq(STRATEGY)].groupby("year").head(10))'''),
        m("## 5. 최종 Parquet·주간 기술통계·검증\ntext_raw/text_sentiment는 v1과 동일하다. 기사 수와 주간 구조도 그대로 두고 clean 토큰 통계만 갱신했다."),
        c('''display(pd.read_csv(RESULTS/"custom_stopword_inspection.csv").head(6))
display(pd.read_csv(RESULTS/"weekly_article_token_counts.csv").loc[lambda t:t.query_strategy.eq(STRATEGY)].head(12))
print((RESULTS/"independent_validation.json").read_text(encoding="utf-8"))
display(Markdown((BASE/"FINAL_PREPROCESSING_REVIEW.md").read_text(encoding="utf-8")))'''),
    ]
    path=ROOT/"guardian_text_preprocessing_final.ipynb"
    nbformat.write(nb,path)
    print(f"Created {path.name}: {len(nb.cells)} cells")
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        manager=KernelManager(kernel_name="python3")
        manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        client=NotebookClient(nb,km=manager,timeout=180,resources={"metadata":{"path":str(ROOT)}})
        client.on_cell_start=lambda cell,cell_index,**kwargs: print(f"Final preprocessing notebook {cell_index+1}/{len(nb.cells)}",flush=True)
        client.execute()
        nbformat.write(nb,path)
        print("Executed saved-results review only; no models or network requests")


if __name__=="__main__":
    main()
