"""Guardian API 2차 파일럿 노트북 생성. 네트워크 요청 없음."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_api_pilot.py"))
import nbformat


def main():
    notebook=nbformat.v4.new_notebook()
    notebook.metadata.kernelspec={"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"}
    m=nbformat.v4.new_markdown_cell
    c=nbformat.v4.new_code_cell
    notebook.cells=[
        m("""# Guardian API · 미국 주식시장 관련성 파일럿
목표 기간 **2017-01-01~2026-10-25**. 세 시기의 일주일 구간에서 6개 전략을 비교한다.
이 노트북은 Guardian 공식 API만 사용한다. 이전 HTML 표본이나 외부 자료를 합치지 않는다.
기본 동작은 저장된 결과 조회이다. **API 키가 없으면 검색 건수·정밀도·규모를 추정하지 않는다.**"""),
        c('''from pathlib import Path
import runpy
BASE=Path.cwd().resolve()
if not (BASE/"guardian_api_pilot.py").exists():
    BASE=BASE/"midterm paper"
api=runpy.run_path(str(BASE/"guardian_api_pilot.py"))
from guardian_api_config import load_key,STRATEGIES,STRATEGY_NOTES,PILOT_WINDOWS
import pandas as pd
import json
from IPython.display import display,Markdown
RESULTS=BASE/"results"/"guardian_api_pilot"
key,key_source=load_key(BASE)
print("API key configured:",bool(key),"| source:",key_source)
del key
print("Configure:",BASE/".env","→ GUARDIAN_API_KEY= 뒤에 키 입력")'''),
        m("""## 1. 키 설정
같은 폴더의 `.env` 파일에서 **`GUARDIAN_API_KEY=` 뒤에 발급받은 키**를 넣고 저장한다.
환경변수도 지원하며 환경변수가 우선이다. 파일은 git에서 제외한다. 키를 셀이나 출력에 넣지 않는다.
[개발자 키 발급](https://open-platform.theguardian.com/access/).

일주일 범위는 아래와 같다. UTC 검색의 경계 버퍼를 추가한 뒤 실제 뉴욕 날짜로 자른다.
API total과 ET 구간 다운로드 수는 다를 수 있다."""),
        c('''display(pd.DataFrame([{"year":year,"start_et":start,"end_et":end} for year,(start,end) in PILOT_WINDOWS.items()]))
display(pd.DataFrame([{"query_strategy":name,"parameters":json.dumps(params),"rationale":STRATEGY_NOTES[name]} for name,params in STRATEGIES.items()]))'''),
        m("""## 2. 필요할 때만 파일럿 실행
키 설정 후 아래 `RUN_PILOT=True`로 바꾸거나 터미널에서 `python guardian_api_pilot.py`를 실행한다.
성공한 페이지는 재사용한다. 기본은 200건/페이지, 주간·전략당 최대 3페이지이다.
전체 기간은 전략당 1건짜리 **건수 메타데이터 요청**만 허용하고 전체 페이지를 순회하지 않는다."""),
        c('''RUN_PILOT=False
if RUN_PILOT:
    result=api["run"]()
else:
    print("Saved-results review only; no network requests.")'''),
        m("## 3. 상태와 검색 결과\n미시도는 0건 검색과 다르다. `partial_page_cap`은 주간 전체 결과를 받지 못한 상태이다."),
        c('''status=json.loads((RESULTS/"run_status.json").read_text(encoding="utf-8"))
print(json.dumps(status,ensure_ascii=False,indent=2))
comparison=pd.read_csv(RESULTS/"strategy_comparison.csv")
display(comparison)'''),
        m("## 4. 중복과 메타데이터\n전략별 소속은 보존하고 Guardian ID로 고유 기사를 센다. 에디션/제작 오피스는 시장 관련성 라벨이 아니다."),
        c('''print((RESULTS/"deduplication_summary.json").read_text(encoding="utf-8"))
display(pd.read_csv(RESULTS/"section_office_counts.csv"))
display(pd.read_csv(RESULTS/"tag_counts.csv").head(30))
catalog=RESULTS/"catalog_inspection.json"
if catalog.exists():
    print("Sections/editions/tag catalog:",catalog)
else:
    print("Catalog not queried; no API metadata available.")'''),
        m("""## 5. 연도별 무작위 수동 검토 표본
`manual_review_sample.csv`에서 `relevant_us_equity_market`에 1/0을 입력한다.
연도별 고유 기사 합집합에서 최대 50건을 고정 시드로 추출한다. 충분하지 않으면 있는 기사만 쓴다.
전략별 소속은 `query_strategy`에 함께 표시한다. 입력한 라벨은 재실행 시 보존된다.
키가 없는 초기 상태의 파일은 **헤더만 있는 템플릿**이다."""),
        c('''review=pd.read_csv(RESULTS/"manual_review_sample.csv",keep_default_na=False)
print("Actual sampled articles:",len(review))
display(review)
display(pd.read_csv(RESULTS/"manual_precision.csv"))'''),
        m("## 6. 전체 기간 수집량·비용 추정\nAPI 메타데이터와 관측 JSON 크기로 계산한다. 버퍼 구간·일일 한도·통신 지연 한계를 함께 읽는다. 조회하지 못한 숫자는 비워 둔다."),
        c('''display(pd.read_csv(RESULTS/"volume_estimates.csv"))
display(Markdown((BASE/"GUARDIAN_API_PILOT_REVIEW.md").read_text(encoding="utf-8")))'''),
        m("""## 7. 제공 라벨 반영 · 기사 수 × precision
최종 전략은 보류한다. 전체 API 기사 수에 파일럿 precision을 곱한 값은 **같은 precision이 전체 기간에 유지된다는 조건부 시나리오**이다.
작은 표본과 연도별 차이를 함께 확인한다. 참고 구간은 이항 작업모형의 Wilson 구간이며 시간 외삽 불확실성을 포함하지 않는다.
자료 재계산은 `python guardian_strategy_yield.py`로 한다. 이 과정은 네트워크 요청을 보내지 않는다."""),
        c('''comparison_path=RESULTS/"strategy_yield_comparison.csv"
if comparison_path.exists():
    comparison=pd.read_csv(comparison_path)
    display(comparison[["query_strategy","reviewed","precision","precision_wilson_95_low","precision_wilson_95_high","full_period_api_count_with_utc_buffer","scenario_relevant_count","scenario_noise_count","decision_status"]].round(4))
    display(pd.read_csv(RESULTS/"strategy_yield_by_year.csv")[["query_strategy","year","unique_articles","n_reviewed","precision","pilot_review_is_census"]])
    from IPython.display import Image
    display(Image(filename=str(BASE/"figures/guardian_strategy_yield.png"),width=1100))
else:
    print("Provided-label yield comparison has not been generated.")'''),
    ]
    path=ROOT/"guardian_api_pilot.ipynb"
    nbformat.write(notebook,path)
    print(f"Created {path.name}: {len(notebook.cells)} cells")
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        manager=KernelManager(kernel_name="python3")
        manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        client=NotebookClient(notebook,km=manager,timeout=120,resources={"metadata":{"path":str(ROOT)}})
        client.on_cell_start=lambda cell,cell_index,**kwargs: print(f"Guardian API notebook {cell_index+1}/{len(notebook.cells)}",flush=True)
        client.execute()
        nbformat.write(notebook,path)
        print("Saved executed review notebook; RUN_PILOT=False; no HTTP requests")


if __name__=="__main__":
    main()
