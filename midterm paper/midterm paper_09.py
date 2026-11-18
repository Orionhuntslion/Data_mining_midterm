"""09. 출처 선택용 소규모 파일럿. 전체 기간 수집은 하지 않는다."""
from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup

from midterm_common import AccessError, PublicClient, digest, save_csv, save_json, utc_now

TITLE = "Validating the Fear & Greed Index with Web-Based Financial Sentiment: Correlation and Lead-Lag Analysis"
WINDOWS = ["2017-01-03", "2020-03-16", "2026-10-23"]
REDDIT_POLICY = "https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data"
REDDIT_RFR = "https://support.reddithelp.com/hc/en-us/articles/49381918834964-Reddit-for-Researchers-Program"


def safe_url(value):
    parts = urlsplit(str(value))
    query = [(k, "REDACTED" if k.lower() in {"api-key", "api_key", "token", "access_token"} else v)
             for k,v in parse_qsl(parts.query,keep_blank_values=True)]
    return urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode(query),parts.fragment))


class PilotClient(PublicClient):
    """기존 robots/속도 제한 클라이언트에 요청 단위 진단만 추가한다."""
    def __init__(self,root,interval=6):
        super().__init__(interval)
        self.log_path=root/"results"/"source_pilot"/"http_log.jsonl"
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.blocked_origins=set()

    def _request(self,url,**kwargs):
        request_url=requests.Request("GET",url,params=kwargs.get("params")).prepare().url
        origin=urlsplit(url).netloc
        if origin in self.blocked_origins:
            raise AccessError(f"앞선 접근 제한 이후 해당 호스트 요청 중단: {origin}")
        row={"checked_at":utc_now(),"url":safe_url(request_url)}
        try:
            response=super()._request(url,**kwargs)
            row.update(http_status=response.status_code,bytes=len(response.content))
            return response
        except (requests.RequestException,AccessError) as exc:
            message=str(exc)
            secret=os.environ.get("GUARDIAN_API_KEY","")
            if secret:
                message=message.replace(secret,"REDACTED")
            match=re.search(r"HTTP (401|403|418|429)",message)
            row.update(http_status=int(match[1]) if match else None,error_type=type(exc).__name__,error=message)
            if match:
                self.blocked_origins.add(origin)
            raise
        finally:
            with self.log_path.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")


def metadata_from_article(html,url):
    """공개 title/description/date 메타데이터만 읽는다. 유료 본문은 추출하지 않는다."""
    soup=BeautifulSoup(html,"html.parser")
    def meta(*keys):
        for key in keys:
            node=soup.find("meta",attrs={"property":key}) or soup.find("meta",attrs={"name":key})
            if node and node.get("content"):
                return node["content"]
        return ""
    title=meta("og:title","twitter:title")
    description=meta("description","og:description")
    published=meta("article:published_time")
    if not published:
        node=soup.find("time",attrs={"datetime":True})
        published=node.get("datetime","") if node else ""
    return {"doc_id":digest(url),"url":url,"title_raw":title,"description_raw":description,
            "raw_text":title+("\n"+description if description else ""),"published_at":published,
            "source":"The Guardian","source_type":"news","timestamp_kind":"publication" if published else "missing"}


def guardian_html(client,day,limit):
    stamp=pd.Timestamp(day)
    month=["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"][stamp.month-1]
    url=f"https://www.theguardian.com/business/{stamp.year}/{month}/{stamp.day:02d}/all"
    response=client.get(url)
    soup=BeautifulSoup(response.text,"html.parser")
    # archive 페이지의 heading에 연결된 기사만 선택. 메뉴/광고 링크 제외.
    links=[]
    for heading in soup.select("h3"):
        anchor=heading.find("a",href=True) or heading.find_parent("a",href=True)
        if not anchor:
            parent=heading.parent
            anchor=parent.find("a",href=True) if parent else None
        if anchor:
            target=urljoin(url,anchor["href"]).split("#")[0]
            if re.search(r"theguardian\.com/.+/\d{4}/[a-z]{3}/\d{2}/.+",target) and target not in links:
                links.append(target)
    rows=[]
    for target in links[:limit]:
        article=client.get(target)
        rows.append(metadata_from_article(article.text,target))
    return rows,{"archive_url":url,"archive_html":response.text,"n_archive_links":len(links)}


def guardian_api(client,day,limit):
    key=os.environ.get("GUARDIAN_API_KEY","").strip()
    if not key:
        return [],{"status":"credentials_missing","note":"GUARDIAN_API_KEY 없음. 인증 실패를 유도하는 요청은 보내지 않음."}
    response=client.get("https://content.guardianapis.com/search",params={
        "api-key":key,"section":"business","from-date":day,"to-date":day,"page-size":limit,
        "show-fields":"headline,trailText","order-by":"oldest"})
    payload=response.json()["response"]
    rows=[]
    for item in payload.get("results",[]):
        fields=item.get("fields",{})
        title=fields.get("headline",item["webTitle"])
        description=fields.get("trailText","")
        rows.append({"doc_id":digest(item["webUrl"]),"url":item["webUrl"],"title_raw":title,
            "description_raw":description,"raw_text":title+("\n"+description if description else ""),
            "published_at":item["webPublicationDate"],"source":"The Guardian","source_type":"news","timestamp_kind":"publication"})
    return rows,{"total_matching":payload.get("total"),"payload":payload}


def gdelt(client,day,limit):
    stamp=pd.Timestamp(day)
    response=client.get("https://api.gdeltproject.org/api/v2/doc/doc",params={
        "query":'("stock market" OR "Wall Street" OR "S&P 500") sourcelang:english sourcecountry:US',
        "mode":"artlist","format":"json","sort":"datedesc","maxrecords":limit,
        "startdatetime":stamp.strftime("%Y%m%d000000"),"enddatetime":stamp.strftime("%Y%m%d235959")})
    try:
        payload=response.json()
    except ValueError as exc:
        raise ValueError(f"GDELT 비 JSON 응답: {response.text[:350]}") from exc
    if "articles" not in payload:
        raise ValueError(f"GDELT 응답 스키마/과거 접근 범위 오류: {str(payload)[:350]}")
    rows=[{"doc_id":digest(item["url"]),"url":item["url"],"title_raw":item["title"],"description_raw":"",
           "raw_text":item["title"],"published_at":"","observed_at":item["seendate"],
           "timestamp_kind":"gdelt_first_seen_not_publication","source":item["domain"],"source_type":"news"}
          for item in payload["articles"]]
    return rows,{"payload":payload,"cap_hit":len(rows)>=limit}


def inspect_and_save(root,source,day,rows,extra):
    folder=root/"data"/"raw"/"news"/source/day[:4]/day[5:7]
    folder.mkdir(parents=True,exist_ok=True)
    save_json({"requested_day":day,"retrieved_at":utc_now(),"records":rows,**extra},folder/f"pilot_{day}.json")
    frame=pd.DataFrame(rows)
    if frame.empty:
        return {"n_raw":0,"n_deduplicated":0,"n_exact_timestamps":0,"n_in_requested_et_date":0,"status":"no_records"}
    frame=frame.drop_duplicates("doc_id").copy()
    n_exact=n_match=0
    et_dates=[]
    for _,row in frame.iterrows():
        text=row.get("published_at","")
        stamp=pd.Timestamp(text) if text else pd.NaT
        exact=not pd.isna(stamp) and stamp.tzinfo is not None
        n_exact+=int(exact)
        et=stamp.tz_convert("America/New_York").date().isoformat() if exact else None
        et_dates.append(et)
        n_match+=int(et==day)
    frame["date_et"]=et_dates
    frame["pilot_source"]=source
    frame["requested_day"]=day
    processed=root/"data"/"processed"/"pilot"/"news"/source/day[:4]/day[5:7]
    processed.mkdir(parents=True,exist_ok=True)
    # 파일럿은 작으므로 JSONL. 월별 단위로 저장하고 모든 연도 자료를 메모리에 합치지 않는다.
    frame.to_json(processed/f"pilot_{day}.jsonl",orient="records",lines=True,force_ascii=False)
    return {"n_raw":len(rows),"n_deduplicated":len(frame),"n_exact_timestamps":n_exact,
            "n_in_requested_et_date":n_match,"status":"sample_obtained" if n_exact else "sample_without_publication_time"}


def audit_fgi(cfg,client):
    """이미 받은 원응답을 재사용. 2021년 이전 계보 차이와 빈 날짜를 별도 표시."""
    from midterm_stage_02 import ARCHIVE_URL
    cache=cfg.raw/"fgi_archive.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True,exist_ok=True)
        cache.write_bytes(client.get(ARCHIVE_URL).content)
    frame=pd.read_csv(cache).rename(columns={"Date":"date","Fear Greed":"fgi","Rating":"category"})
    frame["date"]=pd.to_datetime(frame.date,errors="raise")
    if frame.date.duplicated().any() or not frame.fgi.between(0,100).all():
        raise ValueError("FGI 중복 날짜/점수 범위 오류")
    frame=frame.loc[frame.date.between(pd.Timestamp(cfg.start),pd.Timestamp(cfg.end))].sort_values("date")
    frame["source_url"]=ARCHIVE_URL
    frame["source_era"]="cnn_endpoint_via_third_party_archive"
    frame.loc[frame.date.lt("2021-02-01"),"source_era"]="pre_2021_frozen_multi_source_archive"
    destination=cfg.root/"data"/"external"/f"fear_greed_{cfg.start}_{cfg.end}.csv"
    save_csv(frame,destination)
    missing=pd.bdate_range(cfg.start,cfg.end).difference(frame.date)
    save_csv(pd.DataFrame({"date":missing,"meaning":"weekday_without_value; may include exchange holidays; not automatically a missing trading day"}),
             cfg.root/"results"/"source_pilot"/"fgi_weekday_gaps.csv")
    annual=frame.groupby(frame.date.dt.year).agg(n_observations=("fgi","size"),first_date=("date","min"),last_date=("date","max"))
    save_csv(annual.rename_axis("year").reset_index(),cfg.root/"results"/"source_pilot"/"fgi_annual_coverage.csv")
    return {"status":"cached_archive_audited","n_observations":len(frame),"min_date":str(frame.date.min().date()),
        "max_date":str(frame.date.max().date()),"weekday_dates_without_value":len(missing),
        "era_counts":frame.source_era.value_counts().to_dict(),"no_interpolation":True,
        "sha256_raw":hashlib.sha256(cache.read_bytes()).hexdigest(),"path":str(destination),
        "note":"원응답 캐시 점검이며 최신 서버 재검증은 아님. 2021년 전후 자료 계보 차이는 공식 지수 방법론 변경 확정과 다름."}


def run(cfg,limit=3):
    if not 1<=limit<=5:
        raise ValueError("파일럿은 날짜별 최대 5문서로 제한합니다.")
    folder=cfg.root/"results"/"source_pilot"
    folder.mkdir(parents=True,exist_ok=True)
    windows=[d for d in WINDOWS if cfg.start<=d<=cfg.end]
    client=PilotClient(cfg.root,cfg.request_interval)
    summary=[]
    for source,collector in [("guardian_html",guardian_html),("guardian_api",guardian_api),("gdelt_doc",gdelt)]:
        # 최근 대조 구간을 먼저 확인. 접근 거부 시 같은 호스트를 반복 호출하지 않는다.
        for day in reversed(windows):
            row={"source":source,"source_type":"news","requested_start":day,"requested_end":day,
                 "checked_at":utc_now(),"n_raw":0,"n_deduplicated":0}
            try:
                cache=cfg.root/"data"/"raw"/"news"/source/day[:4]/day[5:7]/f"pilot_{day}.json"
                if cache.exists() and not cfg.refresh:
                    envelope=json.loads(cache.read_text(encoding="utf-8"))
                    rows=envelope.pop("records")
                    extra=envelope
                    row["cache_used"]=True
                else:
                    rows,extra=collector(client,day,limit)
                    row["cache_used"]=False
                if extra.get("status")=="credentials_missing":
                    row.update(extra)
                else:
                    row.update(inspect_and_save(cfg.root,source,day,rows,extra))
            except (requests.RequestException,AccessError,ValueError,KeyError,TypeError) as exc:
                # 오류를 자료 0건과 구분해 저장하고 다른 독립 출처만 계속 점검한다.
                message=str(exc)
                secret=os.environ.get("GUARDIAN_API_KEY","")
                if secret:
                    message=message.replace(secret,"REDACTED")
                row.update(status="not_attempted_after_host_denial" if "앞선 접근 제한" in message else "failed",
                           error_type=type(exc).__name__,error=message)
            summary.append(row)
            save_csv(pd.DataFrame(summary),folder/"pilot_summary.csv")
            print(f"{source} {day}: {row['status']} (n={row.get('n_deduplicated',0)})",flush=True)

    # Reddit은 공개 접근 성공과 연구 이용 승인 여부를 혼동하지 않는다.
    for source,note in [
        ("reddit_official_rfr","공식 연구 승인이 확인되지 않아 게시물 요청 안 함. 공식 제공 범위는 최근 5년, 6개월 지연; 2017/2020/최신 2026-10 표본을 모두 충족하지 못함."),
        ("reddit_arctic_shift","아카이브 API 문서는 존재하지만 공식 정책은 미승인 제3자 도구의 연구 이용을 제한. 연구 이용 허용 근거가 확인되지 않아 게시물 요청 안 함.")]:
        for day in windows:
            summary.append({"source":source,"source_type":"reddit","requested_start":day,"requested_end":day,
                "status":"not_attempted_research_access_unconfirmed","n_raw":0,"n_deduplicated":0,
                "note":note,"policy_url":REDDIT_RFR if source=="reddit_official_rfr" else REDDIT_POLICY})

    # FNSPID는 대형 CSV를 내려받지 않고 공개 파일 목록/설명만 확인한다.
    try:
        url="https://huggingface.co/api/datasets/Zihan1004/FNSPID/tree/main/Stock_news"
        payload=client.get(url).json()
        save_json(payload,folder/"fnspid_file_metadata.json")
        status="metadata_only; historical_rows_not_tested"
    except (requests.RequestException,AccessError,ValueError) as exc:
        status=f"metadata_failed: {type(exc).__name__}: {exc}"
    summary.append({"source":"fnspid_archive","source_type":"news","status":status,"n_raw":0,"n_deduplicated":0,
                    "note":"작성자 공시 1999–2023. 2017/2020 후보지만 날짜별 실제 표본 미검증; 2026은 공시 범위 밖. 대용량 파일 다운로드 안 함."})
    try:
        fgi=audit_fgi(cfg,client)
    except (requests.RequestException,AccessError,ValueError,KeyError) as exc:
        fgi={"status":"failed","error_type":type(exc).__name__,"error":str(exc)}
    save_json(fgi,folder/"fgi_coverage.json")
    save_csv(pd.DataFrame(summary),folder/"pilot_summary.csv")
    save_json({"project_title":TITLE,"target_start":cfg.start,"target_end":cfg.end,"timezone":"America/New_York",
        "primary_index":"WSI_count","robustness_index":"WSI_continuous","sources_kept_separate":["news","reddit"],
        "combined_index":False,"pilot_dates":windows,"max_documents_per_source_date":limit,
        "full_collection_launched":False,"scale_up":"pending pilot review","storage":"raw JSON envelopes + monthly/yearly partitioned JSONL; full scale will use Parquet/chunks",
        "optional_market_returns":"deferred until text sources and core pipeline are validated"},folder/"pilot_plan.json")
    print(f"\n파일럿 결과: {folder}")
    return pd.DataFrame(summary)
