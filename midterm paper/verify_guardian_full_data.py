"""실제 완료 원자료를 배치로 읽어 건수·원문·플래그·범위·진단을 검증. API 요청 없음."""
import hashlib
import importlib.metadata
import json
import runpy
import sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parent
module=runpy.run_path(str(ROOT/"guardian_full_collection.py"))
from guardian_api_config import TARGET_START,TARGET_END,load_key
from guardian_api_client import write_json
import pyarrow.dataset as ds
import pandas as pd


def main():
    settings=module["FullSettings"]()
    state=json.loads((settings.results/"collection_status.json").read_text(encoding="utf-8"))
    summary=json.loads((settings.results/"quality_summary.json").read_text(encoding="utf-8"))
    assert state["status"]=="complete" and state["downstream_analysis_started"] is False
    dataset=ds.dataset(settings.raw/"articles",format="parquet",partitioning="hive")
    prohibited={"compound","sentiment_label","WSI_count","WSI_continuous","fgi","relevant_us_equity_market","frequency_text","sentiment_text"}
    assert not prohibited.intersection(dataset.schema.names)
    ids=set()
    broad=strict=overlap=count=0
    for batch in dataset.scanner(batch_size=200).to_batches():
        for row in batch.to_pylist():
            count+=1
            assert row["guardian_id"] not in ids
            ids.add(row["guardian_id"])
            original=json.loads(row["api_record_json"])
            fields=original.get("fields") or {}
            assert row["guardian_id"]==original["id"] and row["web_title"]==original.get("webTitle")
            assert row["description"]==fields.get("trailText") and row["headline"]==fields.get("headline")
            assert json.loads(row["tags"])==original.get("tags",[])
            assert row["published_at_original"]==original["webPublicationDate"]
            assert row["published_at_utc"]==pd.Timestamp(original["webPublicationDate"]).tz_convert("UTC").to_pydatetime()
            assert row["published_at_et"].date()==row["date_et"]
            assert TARGET_START<=row["date_et"].isoformat()<=TARGET_END
            assert row["year"]==row["date_et"].year
            b,s=row["in_broad_business_control"],row["in_anchors_stock_market_tag"]
            assert b or s
            broad+=int(b)
            strict+=int(s)
            overlap+=int(b and s)
    assert count==summary["final_unique_article_count"]==broad+strict-overlap
    assert overlap==summary["overlap_between_strategies"]
    assert broad==summary["strategy_counts"][0]["unique_articles_in_et_target"]
    assert strict==summary["strategy_counts"][1]["unique_articles_in_et_target"]
    for filename,rows in [("yearly_article_counts.csv",10),("monthly_article_counts.csv",118),("daily_article_counts.csv",3585)]:
        table=pd.read_csv(settings.results/filename)
        assert len(table)==rows and table.unique_articles.sum()==count
    assert len(pd.read_csv(settings.results/"zero_article_days.csv"))==summary["days_with_zero_articles"]
    sample=pd.read_csv(settings.results/"random_inspection_sample.csv")
    assert len(sample)==100 and sample.guardian_id.is_unique and set(sample.guardian_id)<=ids
    connection=sqlite3.connect(settings.raw/"collection.sqlite3")
    assert connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0]==39
    assert connection.execute("SELECT SUM(n_items) FROM pages").fetchone()[0]==summary["total_api_records_retrieved_including_utc_buffer"]
    connection.close()
    key,_=load_key(ROOT)
    paths=list((settings.raw/"pages").rglob("*.json"))
    assert len(paths)==39
    if key:
        for path in paths+list(settings.results.glob("*.json"))+list(settings.results.glob("*.jsonl")):
            assert key not in path.read_text(encoding="utf-8")
    manifest=[{"file":str(path.relative_to(settings.root)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(paths)]
    write_json(manifest,settings.results/"raw_page_hashes.json")
    validation={"status":"passed","records_verified":count,"unique_ids_verified":len(ids),"raw_pages_verified":len(paths),
        "raw_text_and_metadata_preserved":True,"UTC_ET_and_target_range_verified":True,"strategy_flags_verified":True,
        "year_month_day_totals_verified":True,"inspection_sample_verified":True,"no_sentiment_or_relevance_columns":True,
        "secret_check":"passed" if key else "not_tested_without_key","network_requests":0,
        "versions":{name:importlib.metadata.version(name) for name in ["pandas","pyarrow","requests"]}}
    write_json(validation,settings.results/"validation_summary.json")
    current=ROOT/"results"/"execution_status.json"
    status=json.loads(current.read_text(encoding="utf-8"))
    status["code_status"]="Guardian full raw collection and quality diagnostics complete; downstream analysis awaiting review"
    status["guardian_full_collection"]={"target_start":TARGET_START,"target_end":TARGET_END,"status":"complete",
        "primary_strategy":"broad_business_control","strict_robustness_strategy":"anchors_stock_market_tag",
        "unique_articles":count,"overlap":overlap,"primary_unique":broad,"strict_unique":strict,
        "days_with_zero_articles":summary["days_with_zero_articles"],"missing_title":summary["missing_title_count"],
        "missing_description":summary["missing_description_count"],"missing_date":summary["missing_date_count"],
        "validation":"verify_guardian_full_data.py passed","downstream_analysis_started":False}
    write_json(status,current)
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
