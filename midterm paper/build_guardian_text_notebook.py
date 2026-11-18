"""전처리 결과 검토 노트북 생성. 기본 동작은 저장 결과 읽기만 수행한다."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_full_collection.py"))
import nbformat


def main():
    m=nbformat.v4.new_markdown_cell
    c=nbformat.v4.new_code_cell
    notebook=nbformat.v4.new_notebook()
    notebook.metadata.kernelspec={"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"}
    notebook.cells=[
        m("""# Guardian 텍스트 전처리 · 검토용
**Primary: broad / Strict robustness: stock-market tag · Weekly**

원자료 5,265건을 한 번씩 보존하고 `text_raw`, `text_sentiment`, `text_clean`을 별도로 만들었다.
이 노트북은 기본적으로 저장 결과만 읽는다. **VADER 점수·WSI·FGI·시차·수익률은 계산하지 않는다.**
추가 불용어와 표제어화는 미적용 상태이다. 아래 결과를 보고 채택 여부를 결정한다."""),
        c('''from pathlib import Path
import runpy,json
BASE=Path.cwd().resolve()
if not (BASE/"guardian_text_preprocess.py").exists():
    BASE=BASE/"midterm paper"
pipeline=runpy.run_path(str(BASE/"guardian_text_preprocess.py"))
import pandas as pd
from IPython.display import display,Markdown,Image
RESULTS=BASE/"results/guardian_text_preprocessing"
pd.set_option("display.max_colwidth",110)
diagnostics=json.loads((RESULTS/"preprocessing_diagnostics.json").read_text(encoding="utf-8"))
STRATEGY="broad_business_control"  # anchors_stock_market_tag로 변경 가능
TOP_N=20  # 파일에는 각 전략 상위 100개를 저장했다.
print("Physical articles:",diagnostics["physical_article_records"])'''),
        m("""## 1. 규칙과 중복 진단
`headline`을 우선 사용하며 `web_title`도 원본 그대로 보존한다. 표시 텍스트의 정확한 중복만 결합 시 한 번 제외한다.
near-duplicate 후보는 표시만 한다. HTML 제거는 감성용/빈도용 새 열에서만 수행한다."""),
        c('''display(pd.DataFrame(diagnostics["strategy_token_counts"]))
print({name:diagnostics[name] for name in ["headline_differs_from_web_title","headline_description_exact_raw","headline_description_exact_visible","headline_description_near_duplicate","articles_with_html","articles_with_html_entities","articles_with_urls","empty_text_clean"]})
display(pd.read_csv(RESULTS/"headline_description_duplicates.csv").head(20))
inspection=pd.read_csv(RESULTS/"preprocessing_inspection_sample.csv",keep_default_na=False)
display(inspection[["guardian_id","headline_for_text","description","text_raw","text_sentiment","text_clean"]].head(8))'''),
        m("""## 2. Raw / clean 상위 토큰
raw는 HTML·URL 제거 전, 원래 대소문자의 토큰이다. clean은 별도 빈도 입력이며 시장 표기 정규화와 기본 불용어 제거를 포함한다.
기사 수나 토큰 수가 다른 두 전략의 빈도를 단순 합쳐 해석하지 않는다."""),
        c('''for filename in ["top_100_raw_tokens.csv","top_100_cleaned_tokens.csv","common_bigrams.csv"]:
    table=pd.read_csv(RESULTS/filename)
    display(Markdown(f"**{filename} · {STRATEGY}**"))
    display(table.loc[table.query_strategy.eq(STRATEGY)].head(TOP_N))'''),
        m("""## 3. 웹 아티팩트·상용구·추가 불용어 후보
추가 불용어는 아직 적용하지 않았다. 후보의 문서 빈도·문맥을 확인한다.
amp 등 짧은 문자열은 실제 회사/약어일 수 있으므로 자동 삭제하지 않는다."""),
        c('''display(pd.read_csv(RESULTS/"artifact_token_counts.csv"))
display(pd.read_csv(RESULTS/"boilerplate_phrase_counts.csv"))
candidates=pd.read_csv(RESULTS/"candidate_custom_stopwords.csv")
display(candidates.loc[candidates.query_strategy.eq(STRATEGY)].sort_values("count",ascending=False))
display(pd.read_csv(RESULTS/"html_web_artifact_examples.csv").head(5))'''),
        m("""## 4. 표제어화 비교 · 기본 텍스트에는 미적용
어휘 감소가 의미 보존을 입증하지는 않는다. noun-only WordNet의 원래 변환과 보호 규칙을 함께 검토한다.
특히 국가/시장 약어를 일반 명사·동사처럼 처리하지 않는다."""),
        c('''lemma=json.loads((RESULTS/"lemmatization_evaluation.json").read_text(encoding="utf-8"))
print(json.dumps(lemma,ensure_ascii=False,indent=2))
display(pd.read_csv(RESULTS/"lemmatization_risk_probes.csv"))
mapping=pd.read_csv(RESULTS/"lemmatization_mapping.csv")
display(mapping.loc[mapping.would_change].head(30))'''),
        m("""## 5. 주간 기술통계
월~일의 NY 날짜 기준이다. 불완전한 시작 경계 주와 strict 0건 주도 보존한다.
token_count는 clean 토큰 수이며 unique_token_count는 주간 어휘 합집합 크기이다. 감성 평균/점수는 없다."""),
        c('''weekly=pd.read_csv(RESULTS/"weekly_article_token_counts.csv")
display(weekly.loc[weekly.query_strategy.eq(STRATEGY)].head(12))
display(weekly.groupby("query_strategy")[["article_count","token_count"]].sum())
print("Weekly rows:",len(weekly),"| complete rows:",int(weekly.is_complete_week.sum()))'''),
        m("""## 6. 연도별 빈도와 워드클라우드
모든 연도 데이터를 생성했다. 보고서용 연도 선택은 표시 설정일 뿐 원자료/빈도표 생성 범위를 바꾸지 않는다.
2026년은 10월 25일까지의 부분 연도다. 아래는 해석 없이 기술 결과만 표시한다."""),
        c('''yearly=pd.read_csv(RESULTS/"yearly_top_word_frequencies.csv")
display(yearly.loc[yearly.query_strategy.eq(STRATEGY)].groupby("year").head(5))
for label in ["broad","strict"]:
    display(Markdown(f"**{label} · all years**"))
    display(Image(filename=str(BASE/f"figures/guardian_text_preprocessing/wordcloud_{label}_all_years.png"),width=1100))
DISPLAY_YEARS=[]  # 예: [2017,2020,2026]. 데이터/개별 그림은 모든 연도로 이미 저장됨.
for year in DISPLAY_YEARS:
    for label in ["broad","strict"]:
        display(Image(filename=str(BASE/f"figures/guardian_text_preprocessing/wordcloud_{label}_{year}.png"),width=1000))'''),
        m("""## 7. 검증과 재실행
원본 열·전략 플래그·ID·원본 Parquet 해시를 비교하고 주간 토큰 집계도 재계산했다.
필요할 때만 전처리를 다시 실행한다. 실행해도 모델 점수나 외부 자료 결합을 수행하지 않는다."""),
        c('''print((RESULTS/"validation_summary.json").read_text(encoding="utf-8"))
RERUN_PREPROCESS=False
if RERUN_PREPROCESS:
    pipeline["main"]()
display(Markdown(f"진단 보고서: `{BASE / 'GUARDIAN_TEXT_PREPROCESSING_REVIEW.md'}`"))'''),
    ]
    path=ROOT/"guardian_text_preprocessing.ipynb"
    nbformat.write(notebook,path)
    print(f"Created {path.name}: {len(notebook.cells)} cells")
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        manager=KernelManager(kernel_name="python3")
        manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        client=NotebookClient(notebook,km=manager,timeout=180,resources={"metadata":{"path":str(ROOT)}})
        client.on_cell_start=lambda cell,cell_index,**kwargs: print(f"Preprocessing notebook {cell_index+1}/{len(notebook.cells)}",flush=True)
        client.execute()
        nbformat.write(notebook,path)
        print("Executed saved-output review only; no scores or network requests")


if __name__=="__main__":
    main()
