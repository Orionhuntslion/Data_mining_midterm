"""완료된 Guardian 원자료의 Parquet 내보내기와 품질 진단만 수행한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from guardian_api_config import TARGET_START,TARGET_END
from guardian_api_client import write_json

SCHEMA=pa.schema([
    ("guardian_id",pa.string()),("published_at_original",pa.string()),
    ("published_at_utc",pa.timestamp("us",tz="UTC")),("published_at_et",pa.timestamp("us",tz="America/New_York")),
    ("date_et",pa.date32()),("web_title",pa.string()),("headline",pa.string()),("description",pa.string()),
    ("section_id",pa.string()),("section_name",pa.string()),("tags",pa.string()),("web_url",pa.string()),
    ("content_type",pa.string()),("production_office",pa.string()),("api_record_json",pa.string()),
    ("first_retrieved_at",pa.timestamp("us",tz="UTC")),("last_retrieved_at",pa.timestamp("us",tz="UTC")),
    ("in_broad_business_control",pa.bool_()),("in_anchors_stock_market_tag",pa.bool_()),("query_strategy_membership",pa.string()),
])


def rows_from_database(connection,where="",params=()):
    query="SELECT record_json,in_broad,in_strict FROM articles "+where+" ORDER BY date_et,publication_utc,guardian_id"
    cursor=connection.execute(query,params)
    while True:
        batch=cursor.fetchmany(200)
        if not batch:
            break
        rows=[]
        for text,broad,strict in batch:
            record=json.loads(text)
            record["in_broad_business_control"]=bool(broad)
            record["in_anchors_stock_market_tag"]=bool(strict)
            record["query_strategy_membership"]=" | ".join(name for name,flag in [
                ("broad_business_control",broad),("anchors_stock_market_tag",strict)] if flag)
            rows.append(record)
        yield rows


def arrow_table(rows):
    frame=pd.DataFrame(rows)
    for col in ["published_at_utc","first_retrieved_at","last_retrieved_at"]:
        frame[col]=pd.to_datetime(frame[col],utc=True)
    frame["published_at_et"]=frame.published_at_utc.dt.tz_convert("America/New_York")
    frame["date_et"]=pd.to_datetime(frame.date_et).dt.date
    return pa.Table.from_pandas(frame,schema=SCHEMA,preserve_index=False)


def export_parquet(connection,settings,export_csv=False):
    folder=settings.raw/"articles"
    folder.mkdir(parents=True,exist_ok=True)
    written=0
    manifest=[]
    for year in range(2017,2027):
        partition=folder/f"year={year}"
        partition.mkdir(exist_ok=True)
        output=partition/"part-00000.parquet"
        temporary=partition/"part-00000.parquet.tmp"
        n=0
        with pq.ParquetWriter(temporary,SCHEMA,compression="zstd") as writer:
            for rows in rows_from_database(connection,"WHERE substr(date_et,1,4)=?",(str(year),)):
                writer.write_table(arrow_table(rows))
                n+=len(rows)
        temporary.replace(output)
        manifest.append({"year":year,"records":n,"file":str(output.relative_to(settings.root)),"bytes":output.stat().st_size})
        written+=n
    if export_csv:
        output=settings.raw/"guardian_articles.csv"
        temporary=output.with_suffix(".csv.tmp")
        first=True
        with temporary.open("w",encoding="utf-8-sig",newline="") as stream:
            for rows in rows_from_database(connection):
                pd.DataFrame(rows).to_csv(stream,index=False,header=first)
                first=False
        temporary.replace(output)
    pd.DataFrame(manifest).to_csv(settings.results/"parquet_manifest.csv",index=False,encoding="utf-8-sig")
    return written,manifest


def finalize(connection,settings,export_csv=False):
    settings.results.mkdir(parents=True,exist_ok=True)
    count=connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    if count==0:
        raise ValueError("전체 수집 결과가 0건입니다. 완성 자료로 내보내지 않습니다.")
    written,manifest=export_parquet(connection,settings,export_csv)
    if written!=count:
        raise ValueError("Parquet와 원자료 DB의 건수가 다릅니다.")
    strategy=[]
    for name in ["broad_business_control","anchors_stock_market_tag"]:
        raw,inside=connection.execute("SELECT SUM(n_items),SUM(within_et) FROM pages WHERE strategy=?",(name,)).fetchone()
        api_unique=connection.execute("SELECT COUNT(*) FROM seen WHERE strategy=?",(name,)).fetchone()[0]
        eligible=connection.execute("SELECT COUNT(*) FROM seen WHERE strategy=? AND in_range=1",(name,)).fetchone()[0]
        strategy.append({"query_strategy":name,"role":"primary" if name=="broad_business_control" else "strict_robustness",
            "api_records_retrieved_including_utc_buffer":raw,"unique_api_ids_including_utc_buffer":api_unique,
            "records_in_et_target_before_dedup":inside,"unique_articles_in_et_target":eligible,
            "duplicates_within_api_pagination":raw-api_unique})
    strategy=pd.DataFrame(strategy)
    strategy.to_csv(settings.results/"strategy_counts.csv",index=False,encoding="utf-8-sig")
    daily=pd.read_sql_query('''SELECT date_et,COUNT(*) AS unique_articles,SUM(in_broad) AS broad_business_control,
        SUM(in_strict) AS anchors_stock_market_tag,SUM(in_broad*in_strict) AS overlap FROM articles GROUP BY date_et''',connection)
    daily["date_et"]=pd.to_datetime(daily.date_et)
    daily=daily.set_index("date_et").reindex(pd.date_range(TARGET_START,TARGET_END,name="date_et"),fill_value=0).reset_index()
    columns=["unique_articles","broad_business_control","anchors_stock_market_tag","overlap"]
    assert daily.unique_articles.sum()==count
    assert (daily.unique_articles==daily.broad_business_control+daily.anchors_stock_market_tag-daily.overlap).all()
    daily.to_csv(settings.results/"daily_article_counts.csv",index=False,encoding="utf-8-sig")
    yearly=daily.assign(year=daily.date_et.dt.year).groupby("year")[columns].sum().reset_index()
    monthly=daily.assign(month=daily.date_et.dt.to_period("M").astype(str)).groupby("month")[columns].sum().reset_index()
    yearly.to_csv(settings.results/"yearly_article_counts.csv",index=False,encoding="utf-8-sig")
    monthly.to_csv(settings.results/"monthly_article_counts.csv",index=False,encoding="utf-8-sig")
    zero=daily.loc[daily.unique_articles.eq(0),["date_et"]].copy()
    zero["weekday"]=zero.date_et.dt.day_name()
    zero.to_csv(settings.results/"zero_article_days.csv",index=False,encoding="utf-8-sig")
    for name in ["broad_business_control","anchors_stock_market_tag"]:
        daily.loc[daily[name].eq(0),["date_et"]].to_csv(settings.results/f"zero_days_{name}.csv",index=False,encoding="utf-8-sig")
    overlap=connection.execute("SELECT COUNT(*) FROM articles WHERE in_broad=1 AND in_strict=1").fetchone()[0]
    missing_title,missing_description=connection.execute("SELECT SUM(missing_title),SUM(missing_description) FROM articles").fetchone()
    first,last=connection.execute("SELECT MIN(publication_utc),MAX(publication_utc) FROM articles").fetchone()
    first_et,last_et=connection.execute("SELECT MIN(date_et),MAX(date_et) FROM articles").fetchone()
    excluded=pd.read_sql_query("SELECT strategy,reason,COUNT(*) AS records FROM excluded GROUP BY strategy,reason",connection)
    excluded.to_csv(settings.results/"excluded_record_counts.csv",index=False,encoding="utf-8-sig")
    missing_dates=connection.execute("SELECT COUNT(*) FROM excluded WHERE reason IN ('missing_date','missing_or_unzoned_date','invalid_date')").fetchone()[0]
    outside=connection.execute("SELECT COUNT(*) FROM excluded WHERE reason='outside_et_target'").fetchone()[0]
    variants=connection.execute("SELECT COUNT(*) FROM metadata_variants").fetchone()[0]
    # 무작위 검사에는 ID만 메모리에 읽고 선택된 문서만 조회한다.
    ids=[row[0] for row in connection.execute("SELECT guardian_id FROM articles ORDER BY guardian_id")]
    picked=np.random.default_rng(42).choice(ids,size=min(100,len(ids)),replace=False).tolist()
    sample=[]
    for identifier in picked:
        sample.extend(next(rows_from_database(connection,"WHERE guardian_id=?",(identifier,))))
    pd.DataFrame(sample).to_csv(settings.results/"random_inspection_sample.csv",index=False,encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4.5))
    dates=pd.to_datetime(monthly.month)
    for col,label,color in [("broad_business_control","Primary: broad business","#286c98"),
                            ("anchors_stock_market_tag","Strict robustness: stock-market tag","#d88736"),
                            ("unique_articles","Unique union","#60736a")]:
        ax.plot(dates,monthly[col],label=label,color=color)
    ax.set(title="Guardian raw article volume by month",xlabel="Month (America/New_York)",ylabel="Articles")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    figure=settings.root/"figures"/"guardian_full_article_volume.png"
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure,dpi=160)
    plt.close(fig)
    summary={"collection_complete":True,"target_start":TARGET_START,"target_end":TARGET_END,
        "total_api_records_retrieved_including_utc_buffer":int(strategy.api_records_retrieved_including_utc_buffer.sum()),
        "records_in_et_target_before_dedup":int(strategy.records_in_et_target_before_dedup.sum()),
        "strategy_counts":strategy.to_dict("records"),"overlap_between_strategies":overlap,
        "final_unique_article_count":count,"duplicate_count_removed_in_et_target":int(strategy.records_in_et_target_before_dedup.sum())-count,
        "earliest_publication_utc":first,"latest_publication_utc":last,"earliest_date_et":first_et,"latest_date_et":last_et,
        "calendar_days":len(daily),"days_with_zero_articles":len(zero),
        "zero_days_per_strategy":{name:int(daily[name].eq(0).sum()) for name in ["broad_business_control","anchors_stock_market_tag"]},
        "missing_title_count":missing_title,"missing_description_count":missing_description,"missing_date_count":0,
        "missing_or_invalid_date_records_quarantined":missing_dates,"out_of_target_et_records_excluded":outside,
        "metadata_variants_preserved":variants,"random_inspection_sample_size":len(sample),
        "parquet_records":written,"parquet_bytes":sum(x["bytes"] for x in manifest),
        "main_dataset":str((settings.raw/"articles").relative_to(settings.root)),
        "missing_definition":"null, empty or whitespace-only; no HTML stripping",
        "original_responses":"all JSON pages retained; first seen document metadata kept; strategy flags unioned",
        "not_performed":["text preprocessing","sentiment scoring","WSI construction","FGI merging","market-return analysis","classifier training","pilot relevance-label import"]}
    write_json(summary,settings.results/"quality_summary.json")
    lines=["# Guardian 전체 원자료 수집·품질 진단","",f"목표 기간: **{TARGET_START}~{TARGET_END}**, 뉴욕 날짜 기준.",
        f"최종 고유 기사 **{count:,}건**, 두 전략 공통 **{overlap:,}건**.",
        "", "| 전략 | API 수신(UTC 버퍼 포함) | ET 기간 고유 기사 | 역할 |", "|---|---:|---:|---|"]
    for row in strategy.itertuples():
        lines.append(f"| {row.query_strategy} | {row.api_records_retrieved_including_utc_buffer:,} | {row.unique_articles_in_et_target:,} | {row.role} |")
    lines.extend(["",f"ET 구간 중복 제거 전 {summary['records_in_et_target_before_dedup']:,}건에서 중복 {summary['duplicate_count_removed_in_et_target']:,}건을 제거했다.",
        f"UTC 경계 버퍼 중 범위 밖 레코드 {outside}건은 제외했다. 날짜 결측/오류 격리 {missing_dates}건.",
        f"최초 발행 UTC: {first}. 최종 발행 UTC: {last}.",
        f"기사 0건인 달력 날짜 {len(zero):,}일 / 전체 {len(daily):,}일. 주말·휴일을 포함하며 거래일 결측과 동일하지 않다.",
        f"제목 결측 {missing_title}건, 설명 결측 {missing_description}건, 최종 자료 날짜 결측 0건.",
        f"Parquet {written:,}행을 연도별로 저장했고, 무작위 검사 표본 {len(sample)}건을 저장했다.","",
        "## 연도별 고유 기사 수", "", "| 연도 | 합집합 | 주 자료 | 엄격 강건성 | 공통 |", "|---|---:|---:|---:|---:|"])
    for row in yearly.itertuples():
        lines.append(f"| {row.year} | {row.unique_articles:,} | {row.broad_business_control:,} | {row.anchors_stock_market_tag:,} | {row.overlap:,} |")
    lines.extend(["", "## 저장·재개", "", "원응답 JSON과 페이지별 SQLite 트랜잭션을 보존한다. 같은 명령을 재실행하면 완료 페이지를 다시 요청하지 않는다.",
        "한 문서당 레코드는 하나이고 두 전략 소속 플래그와 query_strategy_membership을 보존한다. tags와 전체 API 문서는 JSON 문자열로도 보존한다.",
        "메타데이터가 전략별 응답에서 달라지면 첫 응답을 대표값으로 유지하고 모든 변형은 원응답/SQLite에 남긴다.",
        "Parquet의 UTC/뉴욕 시각은 시간대 타입이며 원래 발행 문자열도 보존한다.",
        "CSV 전체 내보내기: `python guardian_full_collection.py --diagnostics-only --export-csv`.","",
        "## 작업 경계", "", "기사 내용은 전처리하거나 감성 점수를 매기지 않았다. 파일럿 relevance 라벨도 가져오지 않았다.",
        "WSI·FGI·수익률 분석·분류기 학습을 시작하지 않았다. 주 자료의 검색식은 미국시장 관련성의 보증이 아니며 품질 판단은 원자료 검토 후 수행한다.",
        "2026년은 10월 25일까지인 부분 연도이며 연도별 기사 수의 단순 비교에 유의한다."])
    (settings.root/"GUARDIAN_FULL_COLLECTION_REVIEW.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({key:summary[key] for key in ["final_unique_article_count","overlap_between_strategies","days_with_zero_articles","missing_title_count","missing_description_count","missing_date_count"]},ensure_ascii=False),flush=True)
    return summary
