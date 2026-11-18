"""Guardian 파일럿의 중복·검토 표본·규모 추정. 감성/일별 집계/외부 자료 결합 없음."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from guardian_api_config import REVIEW_COLUMNS,STRATEGIES

ARTICLE_COLUMNS=["year","guardian_id","published_at","published_at_utc","published_at_et","date_et",
    "title","description","section","section_id","tags","keyword_tags","production_office","content_type","url","raw_text"]
COMPARISON_COLUMNS=["query_strategy","year","status","api_from_date","api_to_date","total_search_results",
    "downloaded_records_with_utc_buffer","records_in_et_window","duplicates_within_strategy","unique_articles",
    "pages_downloaded","pages_reported","page_size_observed","complete_api_window","total_changed_across_pages","error"]
ESTIMATE_COLUMNS=["query_strategy","status","api_total_with_utc_buffer","estimated_api_calls",
    "page_size_observed","runtime_rate_floor_seconds","minimum_quota_days","approx_raw_bytes",
    "mean_observed_raw_bytes_per_article","basis"]


def normalise(item,year):
    fields=item.get("fields",{})
    original=item["webPublicationDate"]
    stamp=pd.Timestamp(original)
    if stamp.tzinfo is None:
        raise ValueError("Guardian 발행 시각에 시간대가 없습니다.")
    utc=stamp.tz_convert("UTC")
    eastern=utc.tz_convert("America/New_York")
    title=fields.get("headline") or item["webTitle"]
    description=fields.get("trailText","")
    tags=item.get("tags",[])
    return {"year":year,"guardian_id":item.get("id") or item["webUrl"],"published_at":original,
        "published_at_utc":utc.isoformat(),"published_at_et":eastern.isoformat(),"date_et":eastern.date().isoformat(),
        "title":title,"description":description,"section":item.get("sectionName",""),"section_id":item.get("sectionId",""),
        "tags":json.dumps(tags,ensure_ascii=False),
        "keyword_tags":json.dumps([t for t in tags if t.get("type")=="keyword"],ensure_ascii=False),
        "production_office":fields.get("productionOffice",""),"content_type":item.get("type",""),
        "url":item["webUrl"],"raw_text":title+("\n"+description if description else "")}


def deduplicate(records):
    frame=pd.DataFrame(records,columns=ARTICLE_COLUMNS+["query_strategy"])
    if frame.empty:
        return frame,pd.DataFrame(columns=ARTICLE_COLUMNS+["query_strategy"]),{"raw_retrieved_records":0,"duplicate_count":0,"unique_article_count":0}
    memberships=frame[["year","guardian_id","query_strategy"]].drop_duplicates()
    unique=frame.drop_duplicates(["year","guardian_id"]).drop(columns="query_strategy")
    names=memberships.groupby(["year","guardian_id"]).query_strategy.agg(lambda x:" | ".join(sorted(set(x))))
    unique=unique.merge(names,on=["year","guardian_id"],validate="one_to_one")
    return frame,unique,{"raw_retrieved_records":len(frame),"duplicate_count":len(frame)-len(unique),"unique_article_count":len(unique)}


def manual_sample(unique,settings):
    """연도별 고유 기사 합집합에서 균등 추출하고 기존 수동 라벨을 보존한다."""
    path=settings.results/"manual_review_sample.csv"
    archive=settings.results/"manual_review_labels.csv"
    labels={}
    for previous in [archive,path]:
        if previous.exists():
            old=pd.read_csv(previous,keep_default_na=False)
            for row in old.to_dict("records"):
                if str(row.get("relevant_us_equity_market","")).strip() or str(row.get("review_notes","")).strip():
                    labels[str(row["guardian_id"])]=row
    if unique.empty:
        sample=pd.DataFrame(columns=REVIEW_COLUMNS)
    else:
        samples=[]
        for year,group in unique.groupby("year"):
            group=group.sort_values("guardian_id")
            samples.append(group.sample(n=min(settings.manual_per_year,len(group)),random_state=settings.seed+int(year)))
        sample=pd.concat(samples,ignore_index=True)
        sample["relevant_us_equity_market"]=""
        sample["review_notes"]=""
        sample=sample.reindex(columns=REVIEW_COLUMNS)
        for index,row in sample.iterrows():
            saved=labels.get(str(row.guardian_id),{})
            for col in ["relevant_us_equity_market","review_notes"]:
                sample.at[index,col]=saved.get(col,"")
    sample.to_csv(path,index=False,encoding="utf-8-sig")
    pd.DataFrame(list(labels.values()),columns=REVIEW_COLUMNS).to_csv(archive,index=False,encoding="utf-8-sig")
    return sample


def precision_table(sample):
    rows=[]
    for strategy in STRATEGIES:
        for year in [2017,2020,2026]:
            subset=sample.loc[sample.year.eq(year)&sample.query_strategy.fillna("").str.split(" | ",regex=False).map(lambda x:strategy in x)] if len(sample) else sample
            labels=subset.relevant_us_equity_market.astype(str).str.strip().str.lower()
            labels=labels.map({"1":1,"0":0,"1.0":1,"0.0":0,"yes":1,"no":0,"true":1,"false":0}).dropna()
            n=len(labels)
            p=labels.mean() if n else np.nan
            z=1.959963984540054
            center=(p+z*z/(2*n))/(1+z*z/n) if n else np.nan
            half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n) if n else np.nan
            rows.append({"query_strategy":strategy,"year":year,"n_sampled":len(subset),"n_reviewed":n,
                "precision":p,"wilson_95_low":center-half,"wilson_95_high":center+half,
                "scope":"uniform sample of downloaded unique union; truncated windows are not a full result population"})
    return pd.DataFrame(rows)


def volume_estimates(comparison,count_metadata,page_envelopes,settings):
    rows=[]
    for strategy in STRATEGIES:
        metadata=count_metadata.get(strategy,{})
        if metadata.get("status")!="ok":
            rows.append({"query_strategy":strategy,"status":metadata.get("status","not_queried"),"basis":"API key/metadata unavailable; no numeric extrapolation"})
            continue
        total=metadata["total"]
        observations=[x for x in page_envelopes if x["strategy"]==strategy]
        sizes=[int(x["response"].get("pageSize",settings.page_size)) for x in observations]
        actual_size=min(sizes) if sizes else settings.page_size
        n=sum(len(x["response"]["results"]) for x in observations)
        raw_per=sum(x["saved_bytes"] for x in observations)/n if n else np.nan
        calls=math.ceil(total/actual_size) if total else 0
        rows.append({"query_strategy":strategy,"status":"metadata_based_upper_window_estimate","api_total_with_utc_buffer":total,
            "estimated_api_calls":calls,"page_size_observed":actual_size if sizes else np.nan,
            "runtime_rate_floor_seconds":max(0,calls-1)*settings.interval_seconds,
            "minimum_quota_days":math.ceil(calls/settings.daily_request_limit),
            "approx_raw_bytes":total*raw_per,"mean_observed_raw_bytes_per_article":raw_per,
            "basis":"full-period one-result metadata query; UTC buffer may include out-of-target ET records; no full pagination; runtime excludes network/daily reset waits"})
    return pd.DataFrame(rows).reindex(columns=ESTIMATE_COLUMNS)


def save_diagnostics(raw,unique,sample,comparison,counts,pages,settings):
    from guardian_api_client import write_json
    settings.results.mkdir(parents=True,exist_ok=True)
    settings.processed.mkdir(parents=True,exist_ok=True)
    comparison=pd.DataFrame(comparison).reindex(columns=COMPARISON_COLUMNS)
    comparison.to_csv(settings.results/"strategy_comparison.csv",index=False,encoding="utf-8-sig")
    unique.to_csv(settings.processed/"unique_pilot_articles.csv",index=False,encoding="utf-8-sig")
    raw.to_csv(settings.processed/"strategy_article_memberships.csv",index=False,encoding="utf-8-sig")
    precision_table(sample).to_csv(settings.results/"manual_precision.csv",index=False,encoding="utf-8-sig")
    volume_estimates(comparison,counts,pages,settings).to_csv(settings.results/"volume_estimates.csv",index=False,encoding="utf-8-sig")
    # 메타데이터 분포는 검토 단서이며 사람이 붙인 관련성 라벨이 아니다.
    rows=[]
    tag_rows=[]
    for record in raw.drop_duplicates(["year","guardian_id","query_strategy"]).to_dict("records"):
        rows.append({"query_strategy":record["query_strategy"],"year":record["year"],"section":record["section"],"production_office":record["production_office"]})
        for tag in json.loads(record["tags"]):
            tag_rows.append({"query_strategy":record["query_strategy"],"year":record["year"],"tag_id":tag.get("id"),"tag_title":tag.get("webTitle"),"tag_type":tag.get("type")})
    for data,columns,name in [(rows,["query_strategy","year","section","production_office"],"section_office_counts.csv"),
                              (tag_rows,["query_strategy","year","tag_id","tag_title","tag_type"],"tag_counts.csv")]:
        frame=pd.DataFrame(data,columns=columns)
        if len(frame):
            frame=frame.groupby(columns,dropna=False).size().rename("n_articles").reset_index()
        else:
            frame["n_articles"]=pd.Series(dtype=int)
        frame.to_csv(settings.results/name,index=False,encoding="utf-8-sig")
    overlap=[]
    for year in [2017,2020,2026]:
        for a in STRATEGIES:
            left=set(raw.loc[raw.year.eq(year)&raw.query_strategy.eq(a),"guardian_id"])
            for b in STRATEGIES:
                right=set(raw.loc[raw.year.eq(year)&raw.query_strategy.eq(b),"guardian_id"])
                union=left|right
                available=all(bool(ids) or comparison.loc[comparison.year.eq(year)&comparison.query_strategy.eq(name),"status"].eq("complete").any()
                              for name,ids in [(a,left),(b,right)])
                overlap.append({"year":year,"strategy_a":a,"strategy_b":b,"n_overlap":len(left&right) if available else np.nan,
                    "jaccard":len(left&right)/len(union) if union else np.nan,
                    "scope":"downloaded pilot only; blank is unavailable, zero may mean no downloaded data"})
    pd.DataFrame(overlap).to_csv(settings.results/"query_overlap.csv",index=False,encoding="utf-8-sig")
