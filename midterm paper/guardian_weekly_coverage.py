"""수집 원자료의 주별 표본 수 진단과 사용 라이브러리 목록. API/텍스트 전처리 없음."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import runpy
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_full_collection.py"))
import numpy as np
import pandas as pd
from guardian_api_config import TARGET_START,TARGET_END

STRATEGIES={"broad_business_control":"broad","anchors_stock_market_tag":"strict"}


def weekly_counts(daily):
    """NY 달력 날짜를 월요일~일요일에 배정. 잘린 경계 주를 따로 표시한다."""
    daily=daily.copy()
    daily["week_start"]=daily.date_et-pd.to_timedelta(daily.date_et.dt.weekday,unit="D")
    fields=["unique_articles",*STRATEGIES,"overlap"]
    counts=daily.groupby("week_start")[fields].sum()
    observations=daily.groupby("week_start").agg(observed_start=("date_et","min"),observed_end=("date_et","max"),days_observed=("date_et","size"))
    weekly=counts.join(observations).reset_index()
    weekly["week_end"]=weekly.week_start+pd.Timedelta(days=6)
    weekly["is_complete_week"]=weekly.days_observed.eq(7)
    iso=weekly.week_start.dt.isocalendar()
    weekly["iso_year"]=iso.year.to_numpy()
    weekly["iso_week"]=iso.week.to_numpy()
    assert weekly.days_observed.between(1,7).all()
    assert weekly.days_observed.sum()==len(daily)
    assert weekly.week_start.dt.weekday.eq(0).all() and weekly.week_end.dt.weekday.eq(6).all()
    for name in fields:
        assert weekly[name].sum()==daily[name].sum()
    return weekly[["week_start","week_end","iso_year","iso_week","observed_start","observed_end","days_observed","is_complete_week",*fields]]


def distribution(weekly):
    rows=[]
    for scope,frame in [("complete_weeks",weekly.loc[weekly.is_complete_week]),("all_intersecting_weeks",weekly)]:
        for name,label in STRATEGIES.items():
            x=frame[name]
            if x.empty:
                raise ValueError("집계할 주간 표본이 없습니다.")
            rows.append({"scope":scope,"strategy":name,"label":label,"n_weeks":len(x),"total_articles":int(x.sum()),
                "mean":x.mean(),"median":x.median(),"min":int(x.min()),"max":int(x.max()),
                "p25":x.quantile(.25,interpolation="linear"),"p75":x.quantile(.75,interpolation="linear"),
                "zero_article_weeks":int(x.eq(0).sum()),"lt3_article_weeks":int(x.lt(3).sum()),"lt5_article_weeks":int(x.lt(5).sum()),
                "zero_article_weeks_pct":100*x.eq(0).mean(),"lt3_article_weeks_pct":100*x.lt(3).mean(),"lt5_article_weeks_pct":100*x.lt(5).mean()})
    return pd.DataFrame(rows)


def package_inventory(results):
    primary=[("requests","공식 Guardian API GET, 인증·오류·요청 간격 처리"),
        ("pandas","발행 시각 UTC/뉴욕 변환, 건수 집계, CSV, 품질 진단"),
        ("pyarrow","연도별 Parquet 쓰기·배치 읽기·시간대 스키마 보존"),
        ("numpy","고정 시드 무작위 검사 표본, 수치 계산"),
        ("matplotlib","월별 기사량·주별 분포 그래프")]
    auxiliary=[("beautifulsoup4","초기 공개 HTML 출처 파일럿의 제목/설명 메타데이터 추출. 전체 수집은 공식 API라 미사용"),
        ("nbformat","주피터 노트북 생성·구조 검증"),("nbclient","노트북 자동 실행·검증"),
        ("ipykernel","Python 주피터 실행 커널"),("jupyter_client","노트북 커널 실행·관리"),("ipython","노트북 표·그림·Markdown 표시")]
    recorded=json.loads((results/"validation_summary.json").read_text(encoding="utf-8"))["versions"]
    rows=[]
    for group,packages in [("full_collection_and_quality",primary),("initial_pilot_or_notebook",auxiliary)]:
        for name,purpose in packages:
            try:
                version=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version="metadata_unavailable"
            rows.append({"group":group,"library":name,"current_environment_version":version,
                "version_recorded_at_full_validation":recorded.get(name,""),"purpose":purpose,"installation":"external package"})
    builtins={"sqlite3":f"원자료 ID 중복 병합, 페이지별 트랜잭션, 재개 기록; SQLite 엔진 {sqlite3.sqlite_version}",
        "pathlib":"상대/절대 파일 경로와 디렉터리 관리","json":"원응답과 메타데이터·진단 JSON 보존",
        "hashlib":"요청 조건·원자료 SHA-256 검증","datetime / time":"시각·요청 간격·24시간 예산 관리",
        "urllib.robotparser / urllib.parse":"robots 규칙 확인과 공식 API URL 검사",
        "os":"환경변수에서 API 키 읽기; .env는 직접 파싱하며 python-dotenv는 사용하지 않음",
        "argparse / dataclasses":"CLI 옵션과 설정 관리","sys / runpy":"실행 환경 경로와 기존 공통 모듈 재사용",
        "math":"파일럿 요청량 추정과 precision 참고 구간 계산",
        "tempfile / unittest.mock":"임시 합성 자료와 네트워크 없는 회귀 검증",
        "collections.Counter":"Parquet 날짜와 소속 플래그의 주간 건수 독립 교차 검증"}
    python=sys.version.split()[0]
    for name,purpose in builtins.items():
        rows.append({"group":"python_standard_library","library":name,"current_environment_version":f"Python {python}",
            "version_recorded_at_full_validation":"","purpose":purpose,"installation":"built in; no pip install"})
    inventory=pd.DataFrame(rows)
    inventory.to_csv(results/"collection_library_inventory.csv",index=False,encoding="utf-8-sig")
    text=["# 데이터 수집·품질 진단에 사용한 라이브러리","",f"실행 환경: **Python {python}**, SQLite 엔진 **{sqlite3.sqlite_version}**.",
        "전체 원자료 수집은 Guardian 공식 Content API만 사용했다. 일별·주별 건수 진단은 수집 자료의 표본 규모 확인이다.","",
        "## 전체 수집·저장·품질 진단", "", "| 패키지 | 확인 버전 | 용도 |", "|---|---|---|"]
    for row in inventory.loc[inventory.group.eq("full_collection_and_quality")].itertuples():
        text.append(f"| {row.library} | {row.current_environment_version} | {row.purpose} |")
    text.extend(["", "`sqlite3`는 외부 설치 패키지가 아니라 Python 표준 라이브러리다. 페이지별 SQLite와 최종 Parquet는 역할이 다르다.",
        "requests/pandas/pyarrow 버전은 전체 원자료 검증 당시 기록과 함께 CSV에 저장했다. 나머지 버전은 같은 실행 환경에서 현재 조회한 값이다.","",
        "## 초기 출처 탐색·노트북 도구", "", "| 패키지 | 확인 버전 | 사용 범위 |", "|---|---|---|"])
    for row in inventory.loc[inventory.group.eq("initial_pilot_or_notebook")].itertuples():
        text.append(f"| {row.library} | {row.current_environment_version} | {row.purpose} |")
    text.extend(["", "BeautifulSoup를 전체 Guardian API 크롤링에 사용했다고 기술하지 않는다. HTML 출처 파일럿과 공식 API 전체 수집을 구분한다.","",
        "## 표준 라이브러리", ""])
    text.extend(f"- `{name}`: {purpose}." for name,purpose in builtins.items())
    text.extend(["", "## 분석용 의존성과의 구분", "",
        "저장소 requirements.txt에는 향후 분석용 VADER·SciPy·scikit-learn·wordcloud도 포함된다.",
        "이들은 이번 전체 원자료 수집·주간 건수 진단에 사용한 도구 목록에 포함하지 않는다.",
        "초기 별도 HTML 소표본의 VADER 시험과 달리, 최종 원자료 5,265건에는 감성점수·WSI·FGI·relevance 라벨을 붙이지 않았다.",
        "별도 pip 패키지인 python-dotenv는 사용하지 않았다. .env는 표준 라이브러리로 읽고 비밀값을 출력하지 않는다."])
    (ROOT/"COLLECTION_LIBRARIES.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    snapshot=[f"# Python {python}; current environment snapshot; standard-library modules are not pip packages"]
    snapshot.extend(f"{r.library}=={r.current_environment_version}" for r in inventory.loc[~inventory.group.eq("python_standard_library")].itertuples() if r.current_environment_version!="metadata_unavailable")
    (results/"collection_requirements_snapshot.txt").write_text("\n".join(snapshot)+"\n",encoding="utf-8")
    return inventory


def main():
    results=ROOT/"results"/"guardian_full"
    source=results/"daily_article_counts.csv"
    daily=pd.read_csv(source,parse_dates=["date_et"])
    expected=pd.date_range(TARGET_START,TARGET_END,name="date_et")
    if daily.date_et.duplicated().any() or not pd.DatetimeIndex(daily.date_et).equals(expected):
        raise ValueError("일별 원자료 집계에 날짜 중복/누락 또는 기간 불일치가 있습니다.")
    fields=["unique_articles",*STRATEGIES,"overlap"]
    if daily[fields].isna().any().any() or (daily[fields]<0).any().any() or not np.equal(daily[fields],np.floor(daily[fields])).all().all():
        raise ValueError("기사 수는 결측 없는 0 이상의 정수여야 합니다.")
    weekly=weekly_counts(daily)
    stats=distribution(weekly)
    quality=json.loads((results/"quality_summary.json").read_text(encoding="utf-8"))
    for row in quality["strategy_counts"]:
        assert weekly[row["query_strategy"]].sum()==row["unique_articles_in_et_target"]
    assert weekly.unique_articles.sum()==quality["final_unique_article_count"]
    assert weekly.overlap.sum()==quality["overlap_between_strategies"]
    # 기존 일별 CSV와 독립적으로, Parquet에서 날짜/플래그만 배치로 읽어 ISO 주별 건수를 대조한다.
    import pyarrow.dataset as ds
    dataset=ds.dataset(ROOT/"data/raw/news/guardian_full/articles",format="parquet",partitioning="hive")
    counters={name:Counter() for name in STRATEGIES}
    for batch in dataset.scanner(columns=["date_et","in_broad_business_control","in_anchors_stock_market_tag"],batch_size=200).to_batches():
        for record in batch.to_pylist():
            iso=record["date_et"].isocalendar()
            for name in STRATEGIES:
                counters[name][(iso.year,iso.week)]+=int(record["in_"+name])
    for row in weekly.to_dict("records"):
        for name in STRATEGIES:
            assert row[name]==counters[name][(int(row["iso_year"]),int(row["iso_week"]))]
    weekly.to_csv(results/"weekly_article_counts.csv",index=False,encoding="utf-8-sig")
    stats.to_csv(results/"weekly_distribution_summary.csv",index=False,encoding="utf-8-sig")
    full=weekly.loc[weekly.is_complete_week]
    for suffix,test in [("zero",lambda x:x.eq(0)),("lt3",lambda x:x.lt(3)),("lt5",lambda x:x.lt(5))]:
        tables=[]
        for strategy,label in STRATEGIES.items():
            table=full.loc[test(full[strategy]),["week_start","week_end",strategy]].rename(columns={strategy:"article_count"})
            table.insert(0,"strategy",strategy)
            tables.append(table)
        pd.concat(tables,ignore_index=True).to_csv(results/f"weekly_{suffix}_article_weeks.csv",index=False,encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,(name,label),color in zip(axes,STRATEGIES.items(),["#286c98","#d88736"]):
        x=full[name]
        ax.hist(x,bins=np.arange(-.5,max(full[list(STRATEGIES)].max())+1.5,1),color=color,alpha=.85)
        ax.axvline(x.median(),color="#333333",ls="--",label=f"Median = {x.median():g}")
        ax.set(title=f"{label}: {len(x)} complete weeks",xlabel="Articles per Mon–Sun week (New York)",ylabel="Number of weeks")
        ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"guardian_weekly_article_distribution.png",dpi=160)
    plt.close(fig)
    package_inventory(results)
    lines=["# 주별 기사 수 분포 · 데이터 수집 단계 최종 진단", "",
        "목표 기간: **2017-01-01~2026-10-25**, 뉴욕 시간 기준 **월요일~일요일**.",
        f"목표 기간과 겹치는 주는 {len(weekly)}개, 그중 7일 모두 관측되는 완전한 주는 **{len(full)}개**이다.",
        "시작 경계의 2016-12-26~2017-01-01 주는 연구 기간에 2017-01-01 하루만 포함되어 불완전한 주로 별도 표시했다.",
        "주 분석용 표는 이 경계 주를 제외한다. 날짜 범위를 늘리거나 원자료를 바꾸지 않았다.","",
        "| 항목 | broad | strict |", "|---|---:|---:|"]
    primary=stats.loc[stats.scope.eq("complete_weeks")].set_index("label")
    metrics=[("완전한 주 수","n_weeks"),("총 기사 수","total_articles"),("평균","mean"),("중앙값","median"),
        ("최소","min"),("최대","max"),("25 percentile","p25"),("75 percentile","p75"),
        ("기사 0건인 주","zero_article_weeks"),("기사 <3건인 주","lt3_article_weeks"),("기사 <5건인 주","lt5_article_weeks")]
    for title,field in metrics:
        values=[f"{primary.loc[label,field]:,.2f}" if field=="mean" else f"{primary.loc[label,field]:,.0f}" for label in ["broad","strict"]]
        lines.append(f"| {title} | {' | '.join(values)} |")
    lines.extend(["", "`<3`과 `<5`에는 0건인 주가 포함되고 서로 배타적인 구간이 아니다. 분위수는 pandas의 선형 보간 기준이다.",
        "", "## 경계 주를 포함한 참고 결과", "", "| 전략 | 주 수 | 평균 | 중앙값 | 최소/최대 | 25/75% | 0건 | <3건 | <5건 |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|"])
    for row in stats.loc[stats.scope.eq("all_intersecting_weeks")].itertuples():
        lines.append(f"| {row.label} | {row.n_weeks} | {row.mean:.2f} | {row.median:g} | {row.min}/{row.max} | {row.p25:g}/{row.p75:g} | {row.zero_article_weeks} | {row.lt3_article_weeks} | {row.lt5_article_weeks} |")
    lines.extend(["", "## 저장 결과", "", "- `weekly_article_counts.csv`: 모든 경계 주를 포함한 주별 수와 완전한 주 플래그.",
        "- `weekly_distribution_summary.csv`: 완전한 주/전체 겹치는 주의 통계와 희소 주 비율.",
        "- `weekly_zero_article_weeks.csv`, `weekly_lt3_article_weeks.csv`, `weekly_lt5_article_weeks.csv`: 완전한 주 기준 실제 해당 주 목록.",
        "- `figures/guardian_weekly_article_distribution.png`: 완전한 주 기준 분포.",
        "- `COLLECTION_LIBRARIES.md`: 수집·저장·품질 진단과 초기 탐색/노트북 패키지 구분.","",
        "새 API 요청·텍스트 전처리·VADER·WSI·FGI 결합은 수행하지 않았다. 주간 감성지수 채택도 아직 확정하지 않는다."])
    (ROOT/"GUARDIAN_WEEKLY_COVERAGE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest={"source_daily_counts_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "week_definition":"Monday through Sunday; America/New_York calendar date", "complete_weeks":len(full),
        "all_intersecting_weeks":len(weekly),"partial_weeks":int((~weekly.is_complete_week).sum()),
        "weekly_totals_match_daily_and_collection":True,"independent_parquet_iso_week_crosscheck":True,
        "network_requests":0,"text_preprocessing":False,"sentiment_scoring":False}
    (results/"weekly_coverage_validation.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(stats.to_string(index=False))


if __name__=="__main__":
    main()
