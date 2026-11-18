"""승인된 두 Guardian 전략의 전체 원자료 수집. 텍스트 전처리·감성 분석 없음."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
if (ROOT/".packages").is_dir():
    sys.path.insert(0,str(ROOT/".packages"))
if "midterm_common" not in sys.modules:
    runpy.run_path(str(ROOT/"midterm paper.py"))
import pandas as pd
from guardian_api_config import STRATEGIES,TARGET_START,TARGET_END,load_key,window_params
from guardian_api_client import GuardianClient,PilotStopped,write_json

SELECTED={"broad_business_control":"primary","anchors_stock_market_tag":"strict_robustness"}


@dataclass
class FullSettings:
    root: Path=ROOT
    page_size: int=200
    interval_seconds: float=1.1
    daily_request_limit: int=500
    max_requests_per_run: int=200

    def __post_init__(self):
        self.root=Path(self.root).resolve()
        if not 1<=self.page_size<=200 or self.interval_seconds<1 or not 1<=self.daily_request_limit<=500 or not 1<=self.max_requests_per_run<=500:
            raise ValueError("공식 속도·일일 요청 제한과 page_size <=200을 유지해야 합니다.")

    @property
    def raw(self):
        return self.root/"data"/"raw"/"news"/"guardian_full"

    @property
    def results(self):
        return self.root/"results"/"guardian_full"


def request_params(strategy,page,settings):
    return {**STRATEGIES[strategy],**window_params(TARGET_START,TARGET_END),"page-size":settings.page_size,"page":page,
            "order-by":"oldest","use-date":"published","show-fields":"headline,trailText,productionOffice","show-tags":"all"}


def open_database(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript('''
    CREATE TABLE IF NOT EXISTS articles(
      guardian_id TEXT PRIMARY KEY, record_json TEXT NOT NULL, date_et TEXT NOT NULL, publication_utc TEXT NOT NULL,
      missing_title INTEGER NOT NULL, missing_description INTEGER NOT NULL,
      in_broad INTEGER NOT NULL, in_strict INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS seen(strategy TEXT,guardian_id TEXT,in_range INTEGER,PRIMARY KEY(strategy,guardian_id));
    CREATE TABLE IF NOT EXISTS pages(strategy TEXT,page INTEGER,request_sha TEXT,total INTEGER,pages INTEGER,
      n_items INTEGER,within_et INTEGER,missing_date INTEGER,received_at TEXT,PRIMARY KEY(strategy,page));
    CREATE TABLE IF NOT EXISTS excluded(strategy TEXT,page INTEGER,item_index INTEGER,guardian_id TEXT,reason TEXT,record_json TEXT,
      PRIMARY KEY(strategy,page,item_index));
    CREATE TABLE IF NOT EXISTS metadata_variants(guardian_id TEXT,strategy TEXT,page INTEGER,record_json TEXT);
    ''')
    return connection


def article_record(item,retrieved_at):
    identifier=item.get("id")
    original=item.get("webPublicationDate")
    if not identifier:
        return None,"missing_id"
    if not original:
        return None,"missing_date"
    try:
        stamp=pd.Timestamp(original)
        if pd.isna(stamp) or stamp.tzinfo is None:
            return None,"missing_or_unzoned_date"
    except (ValueError,TypeError):
        return None,"invalid_date"
    utc=stamp.tz_convert("UTC")
    eastern=stamp.tz_convert("America/New_York")
    local_date=eastern.date().isoformat()
    if not TARGET_START<=local_date<=TARGET_END:
        return None,"outside_et_target"
    fields=item.get("fields") or {}
    # 문자·HTML·대문자·이모지를 그대로 둔다. 시간대 변환만 수행한다.
    record={"guardian_id":identifier,"published_at_original":original,"published_at_utc":utc.isoformat(),
        "published_at_et":eastern.isoformat(),"date_et":local_date,"web_title":item.get("webTitle"),
        "headline":fields.get("headline"),"description":fields.get("trailText"),
        "section_id":item.get("sectionId"),"section_name":item.get("sectionName"),
        "tags":json.dumps(item.get("tags",[]),ensure_ascii=False),"web_url":item.get("webUrl"),
        "content_type":item.get("type"),"production_office":fields.get("productionOffice"),
        "api_record_json":json.dumps(item,ensure_ascii=False),"first_retrieved_at":retrieved_at,"last_retrieved_at":retrieved_at}
    return record,None


def ingest_page(connection,strategy,page,envelope):
    prior=connection.execute("SELECT request_sha FROM pages WHERE strategy=? AND page=?",(strategy,page)).fetchone()
    if prior:
        if prior[0]!=envelope["request_sha256"]:
            raise ValueError("기존 완료 페이지와 요청 해시가 다릅니다.")
        return False
    response=envelope["response"]
    broad=int(strategy=="broad_business_control")
    strict=int(strategy=="anchors_stock_market_tag")
    inside=missing_date=0
    with connection:
        for index,item in enumerate(response["results"]):
            record,reason=article_record(item,envelope["retrieved_at"])
            identifier=item.get("id")
            if identifier:
                connection.execute("INSERT OR IGNORE INTO seen VALUES(?,?,?)",(strategy,identifier,int(record is not None)))
            if reason:
                missing_date+=int(reason in {"missing_date","missing_or_unzoned_date","invalid_date"})
                connection.execute("INSERT INTO excluded VALUES(?,?,?,?,?,?)",(strategy,page,index,identifier,reason,json.dumps(item,ensure_ascii=False)))
                continue
            inside+=1
            old=connection.execute("SELECT record_json FROM articles WHERE guardian_id=?",(identifier,)).fetchone()
            if old:
                previous=json.loads(old[0])
                if previous["api_record_json"]!=record["api_record_json"]:
                    connection.execute("INSERT INTO metadata_variants VALUES(?,?,?,?)",(identifier,strategy,page,record["api_record_json"]))
                previous["last_retrieved_at"]=envelope["retrieved_at"]
                connection.execute("UPDATE articles SET record_json=?,in_broad=MAX(in_broad,?),in_strict=MAX(in_strict,?) WHERE guardian_id=?",
                    (json.dumps(previous,ensure_ascii=False),broad,strict,identifier))
            else:
                empty=lambda x:x is None or not str(x).strip()
                connection.execute("INSERT INTO articles VALUES(?,?,?,?,?,?,?,?)",(identifier,json.dumps(record,ensure_ascii=False),
                    record["date_et"],record["published_at_utc"],int(empty(record["web_title"])),int(empty(record["description"])),broad,strict))
        connection.execute("INSERT INTO pages VALUES(?,?,?,?,?,?,?,?,?)",(strategy,page,envelope["request_sha256"],
            int(response["total"]),int(response["pages"]),len(response["results"]),inside,missing_date,envelope["retrieved_at"]))
    return True


def collect(settings=None,client=None,export_csv=False):
    settings=settings or FullSettings()
    settings.results.mkdir(parents=True,exist_ok=True)
    settings.raw.mkdir(parents=True,exist_ok=True)
    plan={"target_start":TARGET_START,"target_end":TARGET_END,"timezone":"America/New_York",
          "strategies":{name:{"role":role,"params":STRATEGIES[name]} for name,role in SELECTED.items()},
          "page_size":settings.page_size,"api_date_filter":window_params(TARGET_START,TARGET_END),
          "text_processing":False,"sentiment_scoring":False,"external_merges":False}
    plan_path=settings.results/"collection_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8"))!=plan:
        raise ValueError("진행 중인 수집 설정이 다릅니다. 기존 자료를 덮어쓰지 않습니다.")
    write_json(plan,plan_path)
    key,key_source=load_key(settings.root)
    if not key and client is None:
        raise ValueError("프로젝트 .env 또는 GUARDIAN_API_KEY 환경변수에 키를 설정하세요.")
    client=client or GuardianClient(settings,key)
    connection=open_database(settings.raw/"collection.sqlite3")
    state={"status":"collecting","key_source":key_source,"strategies":{},"downstream_analysis_started":False}
    try:
        for strategy in SELECTED:
            expected_total=expected_pages=None
            page=1
            while True:
                params=request_params(strategy,page,settings)
                signature=hashlib.sha256(json.dumps({k:v for k,v in params.items() if k!="page"},sort_keys=True).encode()).hexdigest()[:16]
                path=settings.raw/"pages"/strategy/signature/f"page_{page:05d}.json"
                envelope,cached=client.fetch("/search",params,"historical",path)
                response=envelope["response"]
                total,pages=int(response["total"]),int(response["pages"])
                if expected_total is None:
                    expected_total,expected_pages=total,pages
                if total!=expected_total or pages!=expected_pages or int(response.get("currentPage",page))!=page:
                    raise ValueError("페이지 간 total/pages/currentPage 변화. 일관된 전체 수집으로 표시하지 않고 중단합니다.")
                ingest_page(connection,strategy,page,envelope)
                state["strategies"][strategy]={"last_completed_page":page,"pages_reported":pages,"api_total":total,"complete":page>=pages}
                write_json(state,settings.results/"collection_status.json")
                print(f"{strategy}: page {page}/{pages}, {len(response['results'])} records, {'cache' if cached else 'API'}",flush=True)
                if page>=pages:
                    break
                page+=1
            observed=connection.execute("SELECT COALESCE(SUM(n_items),0) FROM pages WHERE strategy=?",(strategy,)).fetchone()[0]
            unique=connection.execute("SELECT COUNT(*) FROM seen WHERE strategy=?",(strategy,)).fetchone()[0]
            if observed!=expected_total or unique!=expected_total:
                raise ValueError(f"{strategy}: API total={expected_total}, received={observed}, unique IDs={unique}; 완전성 검증 실패")
        state.update(status="raw_collection_complete",api_requests_this_run=getattr(client,"used",0))
        write_json(state,settings.results/"collection_status.json")
        from guardian_full_diagnostics import finalize
        summary=finalize(connection,settings,export_csv=export_csv)
        state.update(status="complete",quality_summary="quality_summary.json",unique_articles=summary["final_unique_article_count"])
        write_json(state,settings.results/"collection_status.json")
        return summary
    except Exception as exc:
        # API 예외에는 키를 포함하지 않는 클라이언트 메시지만 사용한다.
        error=str(exc).replace(key,"REDACTED") if key else str(exc)
        state.update(status="stopped",error_type=type(exc).__name__,error=error,api_requests_this_run=getattr(client,"used",0))
        write_json(state,settings.results/"collection_status.json")
        raise RuntimeError(f"전체 수집 중단: {type(exc).__name__}: {error}") from None
    finally:
        connection.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-csv",action="store_true",help="완료 후 전체 원자료 CSV도 청크 단위로 내보내기")
    parser.add_argument("--diagnostics-only",action="store_true",help="완료한 DB로만 원자료 내보내기와 진단 재생성; API 요청 없음")
    args=parser.parse_args()
    settings=FullSettings()
    if args.diagnostics_only:
        state=json.loads((settings.results/"collection_status.json").read_text(encoding="utf-8"))
        if state["status"] not in {"complete","raw_collection_complete"}:
            raise ValueError("전체 수집 완료 후에만 최종 진단을 만들 수 있습니다.")
        from guardian_full_diagnostics import finalize
        connection=open_database(settings.raw/"collection.sqlite3")
        try:
            finalize(connection,settings,export_csv=args.export_csv)
        finally:
            connection.close()
    else:
        collect(settings,export_csv=args.export_csv)


if __name__=="__main__":
    main()
