"""Guardian 공식 API 2차 파일럿. 세 주간의 페이지와 전체 기간 건수 메타데이터만 조회."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
if (ROOT/".packages").is_dir():
    sys.path.insert(0,str(ROOT/".packages"))
if "midterm_common" not in sys.modules:
    runpy.run_path(str(ROOT/"midterm paper.py"))

import pandas as pd
from guardian_api_config import (PilotSettings,PILOT_WINDOWS,STRATEGIES,STRATEGY_NOTES,TARGET_START,TARGET_END,load_key,window_params)
from guardian_api_client import GuardianClient,PilotStopped,QueryRejected,write_json
from guardian_api_analysis import deduplicate,normalise,manual_sample,save_diagnostics


def search_params(strategy,start,end,page_size,page):
    return {**STRATEGIES[strategy],**window_params(start,end),"page-size":page_size,"page":page,
        "order-by":"oldest","use-date":"published","show-fields":"headline,trailText,productionOffice","show-tags":"all"}


def page_path(settings,strategy,year,params):
    signature=hashlib.sha256(json.dumps({k:v for k,v in params.items() if k!="page"},sort_keys=True).encode()).hexdigest()[:16]
    return settings.raw/"pilot_pages"/str(year)/strategy/signature/f"page_{params['page']:05d}.json"


def download_window(client,settings,strategy,year,records,page_observations):
    start,end=PILOT_WINDOWS[year]
    row={"query_strategy":strategy,"year":year,"status":"pending",**{f"api_{k.replace('-','_')}":v for k,v in window_params(start,end).items()},
        "downloaded_records_with_utc_buffer":0,"records_in_et_window":0,"duplicates_within_strategy":0,"unique_articles":0,"pages_downloaded":0}
    ids=[]
    totals=[]
    try:
        for page in range(1,settings.max_pages_per_window+1):
            params=search_params(strategy,start,end,settings.page_size,page)
            path=page_path(settings,strategy,year,params)
            envelope,cached=client.fetch("/search",params,"pilot",path)
            response=envelope["response"]
            if int(response.get("currentPage",page))!=page:
                raise QueryRejected("Guardian currentPage가 요청 페이지와 다릅니다.")
            items=response["results"]
            total=int(response["total"])
            totals.append(total)
            pages=int(response["pages"])
            row.update(total_search_results=totals[0],pages_reported=pages,
                page_size_observed=int(response.get("pageSize",settings.page_size)),pages_downloaded=page)
            row["downloaded_records_with_utc_buffer"]+=len(items)
            page_observations.append({"strategy":strategy,"response":response,"saved_bytes":path.stat().st_size,"cache_used":cached})
            for item in items:
                article=normalise(item,year)
                if start<=article["date_et"]<=end:
                    records.append({**article,"query_strategy":strategy})
                    ids.append(article["guardian_id"])
            if page>=pages:
                row["status"]="complete"
                break
            row["status"]="partial_page_cap"
        row["complete_api_window"]=row["status"]=="complete"
    except (PilotStopped,QueryRejected,ValueError,KeyError,TypeError) as exc:
        row.update(status="stopped" if isinstance(exc,PilotStopped) else "query_failed",error=str(exc),complete_api_window=False)
    row.update(records_in_et_window=len(ids),duplicates_within_strategy=len(ids)-len(set(ids)),unique_articles=len(set(ids)),
        total_changed_across_pages=len(set(totals))>1)
    if row["total_changed_across_pages"]:
        row["complete_api_window"]=False
    return row


def collect_catalogs(client,settings):
    catalogs={}
    for name,endpoint,params in [("sections","/sections",{}),("editions","/editions",{}),
                                 ("stock_market_tags","/tags",{"q":"stock markets","page-size":200,"page":1})]:
        try:
            envelope,_=client.fetch(endpoint,params,"catalog",settings.raw/"catalogs"/f"{name}.json")
            catalogs[name]=envelope["response"]
        except QueryRejected as exc:
            catalogs[name]={"status":"query_failed","error":str(exc)}
    write_json(catalogs,settings.results/"catalog_inspection.json")
    return catalogs


def count_only(client,settings,strategy):
    # 10년 전체에서 단 한 페이지, 단 한 결과만 조회한다. 여기서는 pagination하지 않는다.
    params={**STRATEGIES[strategy],**window_params(TARGET_START,TARGET_END),"page-size":1,"page":1,"use-date":"published"}
    envelope,_=client.fetch("/search",params,"count_only",settings.raw/"count_metadata"/f"{strategy}.json")
    return {"status":"ok","total":int(envelope["response"]["total"]),"retrieved_at":envelope["retrieved_at"],
            "scope":"UTC-buffered search count, not exact target ET count"}


def resolve_previous_stop(settings,status):
    path=settings.results/"stop_reason.json"
    if status not in {"completed","completed_with_limits_or_query_errors"} or not path.exists():
        return
    previous=json.loads(path.read_text(encoding="utf-8"))
    if previous.get("status")=="stopped":
        write_json({"status":"resolved","current_run_status":status,"previous_stop":previous,
            "resolution":"robots auxiliary failure is logged independently; official Content API requests completed"},path)


def write_report(settings,status,comparison,dedup,credential_source):
    table=pd.DataFrame(comparison)
    rows=["| 전략 | 2017 | 2020 | 2026 |","|---|---|---|---|"]
    for strategy in STRATEGIES:
        parts=[]
        for year in PILOT_WINDOWS:
            found=table.loc[table.query_strategy.eq(strategy)&table.year.eq(year)]
            if len(found):
                row=found.iloc[0]
                total=row.get("total_search_results")
                total="미조회" if pd.isna(total) else str(int(total))
                parts.append(f"{row['status']}; API total={total}; ET unique={int(row.get('unique_articles',0))}")
            else:
                parts.append("미조회")
        rows.append("| "+" | ".join([strategy,*parts])+" |")
    text=f"""# Guardian API 2차 파일럿 비교

목표 기간: **{TARGET_START}~{TARGET_END}**. 상태: **{status}**.
키 설정 위치 확인 결과: `{credential_source}`. 키 값은 기록하지 않는다.

## 실제 실행 상태

UTC 버퍼 포함 API 수신 레코드 {dedup.get('api_results_received_including_utc_buffer',0)}건.
ET 파일럿 구간 내 전략 소속 중복 포함 관측치 {dedup['raw_retrieved_records']}건,
중복 {dedup['duplicate_count']}건, 고유 기사 {dedup['unique_article_count']}건.
키가 없으면 이 0은 실제 검색 결과 0건을 뜻하지 않는다. API total·규모 추정은 미조회로 남긴다.
이전 공개 HTML 9건은 공식 API 비교 자료에 넣지 않았다.

{chr(10).join(rows)}

## 검색 전략

"""
    for name,params in STRATEGIES.items():
        text+=f"- **{name}**: `{json.dumps(params,ensure_ascii=False)}`. {STRATEGY_NOTES[name]}\n"
    text+="""

## 키 설정과 실행

같은 폴더의 **`.env`**에서 다음 등호 뒤에 개발자 키를 입력한다.

```dotenv
GUARDIAN_API_KEY=발급받은_키
```

파일은 git에서 제외된다. `.env.example`은 배포용 빈 예시이다.
현재 프로세스의 `GUARDIAN_API_KEY` 환경변수가 있으면 `.env`보다 우선한다.
VS Code 커널/터미널을 이미 실행 중이라면 파일 방식이 간단하다. 키를 채팅이나 노트북에 쓰지 않는다.

```powershell
python guardian_api_pilot.py
```

실제 환경 실행 파일이 필요하면 `C:\\Users\\rhffl\\anaconda3\\envs\\hwenv\\python.exe`를 사용한다.
등록: [Guardian Developer API](https://open-platform.theguardian.com/access/).

## 표본·시각·중복 검증

- 2017-01-09~15, 2020-03-16~22, 2026-10-19~25의 일주일 구간을 비교한다.
- 모든 원응답은 조건 해시+연도+전략+페이지별 저장이다. 성공한 캐시는 재사용한다.
- 기본 page-size=200, 주간별 최대 3페이지다. 응답의 pageSize/pages를 확인하며 실제 지원 여부는 키 설정 후 검증한다.
  400이면 지원되지 않은 파라미터/페이지 크기를 보고하고 조용히 조건을 바꾸지 않는다.
- `partial_page_cap`이면 검색 결과 전체를 받지 못한 것이다. 다운로드 순서는 oldest이므로 이때는 주간 전체의 무작위 표본이 아니다.
- 날짜 필터는 UTC 날짜 끝에 하루 버퍼를 두고, 원래 발행 문자열·UTC·America/New_York·ET 날짜를 저장한다.
  API total은 버퍼 구간의 값이다. 실제 ET 구간 문서 수는 별도 저장하며 둘을 같은 수로 표시하지 않는다.
- 안정적인 Guardian ID(없으면 URL)로 중복을 제거하고 전략별 소속을 모두 보존한다.
- `manual_review_sample.csv`: 다운로드된 고유 기사 합집합에서 연도별 최대 50건을 고정 시드로 무작위 추출한다.
  `relevant_us_equity_market`는 빈칸이다. 사람이 1/0을 입력하면 다시 실행 시 전략별 precision과 Wilson 구간을 계산한다.
  표본의 전략별 수는 `manual_precision.csv`에서 확인한다. 일부 전략에 검토 표본이 적으면 정밀도 결론을 유보한다.
- 기존 사람이 입력한 라벨은 ID로 재사용하고 `manual_review_labels.csv`에도 보존한다.
- sections·editions 목록과 keyword tags·productionOffice 분포를 저장한다. 미국 제작 오피스/US 에디션은 시장 관련성 라벨이 아니다.
- q 검색 범위와 반환한 headline/trailText의 범위가 같다고 가정하지 않는다. 반환한 관측 텍스트 자체의 관련성을 수동 검토한다.
- 캐시 시점이 다르면 검색 인덱스 변경으로 페이지가 달라질 수 있다. total 변화는 완전 수집으로 표시하지 않는다.

## 전체 기간 규모 추정

각 전략당 **page-size=1, page=1의 메타데이터 요청만** 전체 목표 기간에 보낸다.
full-period pages를 순회하는 기능은 없다. 이 total과 파일럿의 실제 페이지 크기·원응답 바이트를 사용한다.
UTC 경계 버퍼가 있어 수집량은 약간 넓은 범위의 추정값이며 전략별 total을 더해 합집합 수로 쓰지 않는다.

- API 호출 수 ≈ ceil(total / observed pageSize).
- 속도상 최소 실행 시간 ≈ (호출 수 − 1) × 1.1초. 통신 지연/일일 한도 대기는 제외한다.
- 일일 한도상 최소 일수 ≈ ceil(호출 수 / 500). 같은 키의 다른 사용량은 알 수 없다.
- 원응답 저장 용량 ≈ total × 파일럿에서 관측된 JSON 바이트/문서. 수정·태그 길이에 따라 달라진다.
- `volume_estimates.csv`에 근거와 수치를 저장한다. 키가 없거나 메타데이터 실패 시 숫자를 만들지 않는다.
- 세 주간만 10년으로 단순 외삽하지 않는다. COVID 주간은 보통 시기의 무작위 표본이 아니다.

공식 개발자 한도: 초당 1회·일 500회. 로컬 클라이언트는 1.1초 간격, 실행당 최대 90요청,
보수적인 최근 24시간 500요청 제한을 사용한다. robots 요청도 로컬 예산에 포함한다.
401/403/418/429는 자동 재시도하지 않고 전체 실행을 중단한다.
이는 **Content API 응답**의 처리이다. `/robots.txt`의 401은 별도 보조 검사 실패로 기록하고
`/search` 인증 실패로 판정하지 않는다. 실제로 받은 robots 금지 규칙은 준수하며 공식 JSON API 외 HTML 경로는 허용하지 않는다.

이전 중단 경로는 `fetch → PublicClient.get → robots 요청 → PublicClient._request`가 401을
`AccessError`로 바꾼 후, 래퍼 `logged_request`가 엔드포인트를 구분하지 않고
`halted=True`와 `PilotStopped`를 발생시킨 것이었다.
수정 후 robots 검사는 분리했고, 실제 API 401/403/429 중단·요청 간격·예산은 유지했다.

## 현재 추천과 채택 조건

**잠정 기준 후보는 `anchors_global`**이다. 명확한 미국시장 명칭을 쓰면서 섹션·제작 오피스 제한으로
Technology 등 관련 기사나 영국에서 작성한 미국시장 뉴스를 미리 제외하지 않는 접근이다.
이는 실측 정밀도 우승 전략이라는 뜻이 아니다. 키가 없거나 수동 검토가 끝나지 않았다면 최종 선택을 하지 않는다.
`geography_and_equity`의 추가 포괄성과 Business/태그/오피스 제한의 정밀도·누락을 세 연도별로 비교한 후 결정한다.
FGI 상관이 높다는 이유로 검색 전략을 고르지 않는다.

Guardian만 다룬다. Reddit/FNSPID/FGI/S&P 500 결합, 일별 감성 집계, 전체 수집은 실행하지 않는다.
Reddit은 전 기간 주 자료로 보지 않고, FNSPID는 향후 독립적인 2017~2023 강건성 자료로만 검토한다.

근거: [공식 검색/필드 문서](https://open-platform.theguardian.com/documentation/),
[공식 클라이언트의 검색·태그·페이지 처리](https://github.com/guardian/content-api-scala-client),
[개발자 키·한도](https://open-platform.theguardian.com/access/).
"""
    (settings.root/"GUARDIAN_API_PILOT_REVIEW.md").write_text(text,encoding="utf-8")


def run(settings=None,client=None):
    settings=settings or PilotSettings()
    settings.results.mkdir(parents=True,exist_ok=True)
    settings.raw.mkdir(parents=True,exist_ok=True)
    key,key_source=load_key(settings.root)
    previous_status=settings.results/"run_status.json"
    if not key and client is None and previous_status.exists():
        previous=json.loads(previous_status.read_text(encoding="utf-8"))
        if previous.get("unique_article_count",0)>0:
            print("키가 없어 새 요청을 보내지 않습니다. 기존 실제 결과·검토 표본을 보존했습니다.")
            return {"status":"credentials_missing_cached_results_preserved"}
    records=[]
    comparisons=[]
    page_observations=[]
    counts={}
    status="credentials_missing"
    write_json({"target_start":TARGET_START,"target_end":TARGET_END,"windows":PILOT_WINDOWS,"strategies":STRATEGIES,
        "page_size":settings.page_size,"max_pages_per_window":settings.max_pages_per_window,
        "max_requests_per_run":settings.max_requests_per_run,"full_collection":False,
        "full_period_requests":"count metadata only: page-size=1, page=1; no pagination"},settings.results/"pilot_plan.json")
    if key or client is not None:
        client=client or GuardianClient(settings,key)
        status="completed"
        try:
            catalogs=collect_catalogs(client,settings)
            verified_tags={x["id"] for x in catalogs.get("stock_market_tags",{}).get("results",[])}
            for strategy,query in STRATEGIES.items():
                if query.get("tag") and query["tag"] not in verified_tags:
                    for year in PILOT_WINDOWS:
                        comparisons.append({"query_strategy":strategy,"year":year,"status":"tag_not_verified","unique_articles":0})
                    counts[strategy]={"status":"tag_not_verified"}
                    continue
                for year in PILOT_WINDOWS:
                    if getattr(client,"halted",False):
                        raise PilotStopped("접근 제한 후 추가 요청 중단")
                    row=download_window(client,settings,strategy,year,records,page_observations)
                    comparisons.append(row)
                    pd.DataFrame(comparisons).to_csv(settings.results/"strategy_comparison.csv",index=False,encoding="utf-8-sig")
                    print(f"{strategy} {year}: {row['status']}, unique={row.get('unique_articles',0)}",flush=True)
                    if row["status"]=="stopped":
                        raise PilotStopped(row["error"])
                try:
                    counts[strategy]=count_only(client,settings,strategy)
                except QueryRejected as exc:
                    counts[strategy]={"status":"metadata_query_failed","error":str(exc)}
        except PilotStopped as exc:
            status="stopped"
            write_json({"status":status,"error":str(exc)},settings.results/"stop_reason.json")
    existing={(x["query_strategy"],x["year"]) for x in comparisons}
    if status=="completed" and any(x["status"]!="complete" for x in comparisons):
        status="completed_with_limits_or_query_errors"
    for strategy in STRATEGIES:
        for year in PILOT_WINDOWS:
            if (strategy,year) not in existing:
                comparisons.append({"query_strategy":strategy,"year":year,"status":"credentials_missing" if not key and client is None else "not_attempted_after_stop","unique_articles":0})
        counts.setdefault(strategy,{"status":"credentials_missing" if status=="credentials_missing" else "not_queried"})
    raw,unique,dedup=deduplicate(records)
    dedup["raw_records_scope"]="ET pilot windows, including repeated strategy memberships"
    dedup["api_results_received_including_utc_buffer"]=sum(x.get("downloaded_records_with_utc_buffer",0) for x in comparisons)
    sample=manual_sample(unique,settings)
    save_diagnostics(raw,unique,sample,comparisons,counts,page_observations,settings)
    write_json(counts,settings.results/"count_metadata_summary.json")
    write_json(dedup,settings.results/"deduplication_summary.json")
    write_json({"status":status,"key_source":key_source,"api_requests_this_run":getattr(client,"used",0) if client else 0,
        "manual_sample_records":len(sample),"full_collection":False,**dedup},settings.results/"run_status.json")
    resolve_previous_stop(settings,status)
    write_report(settings,status,comparisons,dedup,key_source)
    print(f"Guardian API pilot: {status}; unique={len(unique)}, manual sample={len(sample)}")
    if status=="credentials_missing":
        print(f"API 키 설정 위치: {settings.root/'.env'} 의 GUARDIAN_API_KEY= 뒤")
    return {"status":status,"articles":unique,"comparisons":pd.DataFrame(comparisons),"sample":sample}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size",type=int,default=200)
    parser.add_argument("--max-pages",type=int,default=3)
    args=parser.parse_args()
    run(PilotSettings(page_size=args.page_size,max_pages_per_window=args.max_pages))


if __name__=="__main__":
    main()
