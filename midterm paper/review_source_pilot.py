"""저장된 파일럿만 점검·문서별 VADER 확인·노트북 정리. 네트워크 요청 없음."""
from __future__ import annotations

import json
import re
import runpy
import sys
from pathlib import Path

BASE=Path(__file__).resolve().parent
pipeline=runpy.run_path(str(BASE/"midterm paper.py"))
common=pipeline["common"]
cleaner=pipeline["load_module"](3,"midterm_stage_03")

import nbformat
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def inspect_saved_samples():
    analyzer=SentimentIntensityAnalyzer()
    all_rows=[]
    for file in sorted((BASE/"data"/"processed"/"pilot"/"news").glob("*/*/*/pilot_*.jsonl")):
        frame=pd.read_json(file,orient="records",lines=True)
        for record in frame.to_dict("records"):
            record["sentiment_text"]=cleaner.sentiment_text(record["raw_text"])
            record["frequency_text"]=" ".join(cleaner.tokens_for_frequency(record["sentiment_text"]))
            record.update(analyzer.polarity_scores(record["sentiment_text"]))
            record["sentiment_label"]="positive" if record["compound"]>=0.05 else "negative" if record["compound"]<=-0.05 else "neutral"
            record["us_equity_keyword_hit"]=bool(re.search(
                r"\b(?:Wall Street|S&P\s?500|Federal Reserve|Nasdaq|Dow Jones|US stocks|US markets|US economy|US interest|Fed)\b|\bU\.S\.",
                record["raw_text"],re.I))
            all_rows.append(record)
        # 소규모 문서별 점수만 기록한다. 이 자료로 일별 WSI/상관을 추정하지 않는다.
    audit=pd.DataFrame(all_rows)
    common.save_csv(audit,BASE/"results"/"source_pilot"/"news_sample_audit.csv")
    if len(audit):
        coverage=audit.groupby(["pilot_source","requested_day"]).agg(
            n_docs=("doc_id","size"),n_us_keyword_hits=("us_equity_keyword_hit","sum"),
            n_missing_descriptions=("description_raw",lambda x:sum(not str(v).strip() for v in x)))
        common.save_csv(coverage.reset_index(),BASE/"results"/"source_pilot"/"news_sample_quality.csv")
        print(coverage.to_string())
    return audit


def update_notebooks():
    old=BASE/"midterm_sentiment.ipynb"
    notebook=nbformat.read(old,as_version=4)
    for cell in notebook.cells:
        cell.source=cell.source.replace("2026-08-01","2017-01-01").replace("2026-10-24","2026-10-25")
        cell.source=cell.source.replace("WSI_continuous (주 지수)","WSI_continuous (강건성)")
        cell.source=cell.source.replace("| WSI_count |","| **WSI_count (주 지수)** |")
        cell.source=cell.source.replace('lag_results["index"].eq("WSI_continuous")','lag_results["index"].eq("WSI_count")')
        if cell.cell_type=="code":
            cell.outputs=[]
            cell.execution_count=None
    notice="""> **수정 설계 · 현재 단계는 출처 파일럿입니다.**
> 목표 기간: **2017-01-01~2026-10-25**. 먼저 `midterm_source_pilot.ipynb`과 `SOURCE_PILOT_REVIEW.md`를 확인하세요.
> 아래는 기존 **뉴스 단독** 분석 경로입니다. Reddit 자료를 합치지 않습니다.
> 주 지수는 `WSI_count`, 연속 지수는 강건성 비교입니다. 대규모 수집은 파일럿 검토 전 중단합니다.

"""
    notebook.cells[0].source=re.sub(r"^> \*\*수정 설계.*?\n\n","",notebook.cells[0].source,flags=re.S)
    notebook.cells[0].source=notice+notebook.cells[0].source
    notebook.cells[0].source=notebook.cells[0].source.replace(
        "# Financial Web Sentiment × Fear & Greed",
        "# Validating the Fear & Greed Index with Web-Based Financial Sentiment\n### Correlation and Lead-Lag Analysis")
    nbformat.write(notebook,old)

    pilot=nbformat.v4.new_notebook()
    pilot.metadata.kernelspec={"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"}
    pilot.cells=[
        nbformat.v4.new_markdown_cell("""# 출처 파일럿 리뷰
**Validating the Fear & Greed Index with Web-Based Financial Sentiment: Correlation and Lead-Lag Analysis**

목표 기간은 **2017-01-01~2026-10-25**, 정렬 기준은 `America/New_York`이다.
2017-01-03, 2020-03-16, 2026-10-23 각각 최대 3문서로 접근과 필드를 점검했다.
이 노트북의 기본 동작은 **저장한 실제 결과 조회**이다. 전체 기간 수집을 실행하지 않는다.
뉴스와 Reddit은 별도 자료와 지수를 사용한다. 주 지수는 count, 연속 지수는 강건성 비교이다.
"""),
        nbformat.v4.new_code_cell('''from pathlib import Path
import runpy
BASE=Path.cwd().resolve()
if not (BASE/"midterm paper.py").exists():
    BASE=BASE/"midterm paper"
pipeline=runpy.run_path(str(BASE/"midterm paper.py"))
read_csv=pipeline["common"].read_csv
import json
from IPython.display import display,Markdown
RESULTS=BASE/"results"/"source_pilot"
print("Target: 2017-01-01 ~ 2026-10-25 | America/New_York")'''),
        nbformat.v4.new_markdown_cell("## 1. 실제 요청 결과\n`failed`, `credentials_missing`, `not_attempted_*`를 자료 0건과 구분한다. HTTP 요청 로그는 `http_log.jsonl`에 있다."),
        nbformat.v4.new_code_cell('''summary=read_csv(RESULTS/"pilot_summary.csv")
display(summary)
print("Actual HTTP requests:",len((RESULTS/"http_log.jsonl").read_text(encoding="utf-8").splitlines()))'''),
        nbformat.v4.new_markdown_cell("## 2. 수집 문서와 필드 점검\n원문 headline+description, 발행 시각, URL, VADER 점수를 확인한다. 표본은 접근 검증용이며 일별 지수나 상관을 계산하기에 충분하지 않다. 미국시장 키워드 검사는 간단한 품질 신호일 뿐 최종 관련성 판정이 아니다."),
        nbformat.v4.new_code_cell('''audit=read_csv(RESULTS/"news_sample_audit.csv")
display(audit.reindex(columns=["pilot_source","requested_day","published_at","date_et","title_raw","description_raw","neg","neu","pos","compound","us_equity_keyword_hit","url"]))
display(read_csv(RESULTS/"news_sample_quality.csv"))'''),
        nbformat.v4.new_markdown_cell("## 3. FGI 전체 기간 점검\n값이 없는 평일에는 거래소 휴일이 포함될 수 있다. 실제 거래일 누락으로 단정하지 않으며 보간하지 않는다. 2021년 이전/이후 자료 계보를 구분한다."),
        nbformat.v4.new_code_cell('''print(json.dumps(json.loads((RESULTS/"fgi_coverage.json").read_text(encoding="utf-8")),ensure_ascii=False,indent=2))
display(read_csv(RESULTS/"fgi_annual_coverage.csv"))'''),
        nbformat.v4.new_markdown_cell("## 4. 출처 리뷰와 확대 전 결정"),
        nbformat.v4.new_code_cell('''review=BASE/"SOURCE_PILOT_REVIEW.md"
display(Markdown(review.read_text(encoding="utf-8")))'''),
        nbformat.v4.new_markdown_cell("## 5. 필요할 때만 파일럿 재실행\n아래 실행 플래그는 기본 False이다. 429/접근 거부 응답에서 해당 호스트 요청을 멈춘다. Guardian API 키는 환경변수로만 읽고 로그에서 숨긴다."),
        nbformat.v4.new_code_cell('''RERUN_PILOT=False
if RERUN_PILOT:
    cfg=pipeline["Config"](root=BASE,start="2017-01-01",end="2026-10-25")
    pipeline["run_stage"](cfg,"pilot")'''),
    ]
    nbformat.write(pilot,BASE/"midterm_source_pilot.ipynb")
    print("Updated notebook + created source-pilot notebook")


def execute_pilot_notebook():
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    path=BASE/"midterm_source_pilot.ipynb"
    notebook=nbformat.read(path,as_version=4)
    manager=KernelManager(kernel_name="python3")
    manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    client=NotebookClient(notebook,km=manager,timeout=120,resources={"metadata":{"path":str(BASE)}})
    client.on_cell_start=lambda cell,cell_index,**kwargs: print(f"Pilot notebook {cell_index+1}/{len(notebook.cells)}",flush=True)
    client.execute()
    nbformat.write(notebook,path)
    print("Saved executed pilot-review notebook; RERUN_PILOT=False; no new HTTP requests")


if __name__=="__main__":
    inspect_saved_samples()
    update_notebooks()
    if "--execute" in sys.argv:
        execute_pilot_notebook()
