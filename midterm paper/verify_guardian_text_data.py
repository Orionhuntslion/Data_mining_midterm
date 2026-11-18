"""처리 Parquet의 모든 원본 열·텍스트·주별 토큰 합계를 독립 검증. 네트워크/모델 호출 없음."""
import hashlib
import json
import runpy
from collections import Counter,defaultdict
from datetime import timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_full_collection.py"))
import pandas as pd
import pyarrow.dataset as ds
from guardian_api_client import write_json


def digest(row,columns):
    canonical=json.dumps({name:row[name] for name in columns},sort_keys=True,ensure_ascii=False,default=lambda value:value.isoformat())
    return hashlib.sha256(canonical.encode()).hexdigest()


def main():
    results=ROOT/"results/guardian_text_preprocessing"
    raw=ds.dataset(ROOT/"data/raw/news/guardian_full/articles",format="parquet",partitioning="hive")
    processed=ds.dataset(ROOT/"data/processed/guardian_text/articles",format="parquet",partitioning="hive")
    columns=raw.schema.names
    assert set(columns)<=set(processed.schema.names)
    assert not {"compound","sentiment_label","WSI_count","WSI_continuous","fgi","relevant_us_equity_market"}.intersection(processed.schema.names)
    original={}
    for batch in raw.scanner(batch_size=200).to_batches():
        for row in batch.to_pylist():
            original[row["guardian_id"]]=digest(row,columns)
    seen=set();articles=Counter();tokens=Counter();week_tokens=Counter();week_docs=Counter();week_vocabulary=defaultdict(set)
    for batch in processed.scanner(batch_size=200).to_batches():
        for row in batch.to_pylist():
            identifier=row["guardian_id"]
            assert identifier not in seen and digest(row,columns)==original[identifier]
            seen.add(identifier)
            assert row["text_clean"]==" ".join(row["tokens_clean"])
            assert len(row["tokens_clean"])==row["clean_token_count"]
            assert len(set(row["tokens_clean"]))==row["clean_unique_token_count"]
            start=row["date_et"]-timedelta(days=row["date_et"].weekday())
            assert row["week_start"]==start and row["week_end"]==start+timedelta(days=6)
            iso=start.isocalendar()
            assert row["week_id"]==f"{iso.year}-W{iso.week:02d}"
            for strategy,flag in [("broad_business_control","in_broad_business_control"),("anchors_stock_market_tag","in_anchors_stock_market_tag")]:
                if row[flag]:
                    articles[strategy]+=1;tokens[strategy]+=row["clean_token_count"]
                    key=(strategy,start.isoformat())
                    week_docs[key]+=1;week_tokens[key]+=row["clean_token_count"];week_vocabulary[key].update(row["tokens_clean"])
    assert seen==set(original) and len(seen)==5265
    weekly=pd.read_csv(results/"weekly_article_token_counts.csv")
    assert len(weekly)==1026
    for row in weekly.itertuples():
        key=(row.query_strategy,row.week_start)
        assert row.article_count==week_docs[key] and row.token_count==week_tokens[key]
        assert row.unique_token_count==len(week_vocabulary[key])
    frequency=pd.read_csv(results/"full_period_word_frequencies.csv")
    for strategy,n in tokens.items():
        assert frequency.loc[frequency.query_strategy.eq(strategy),"count"].sum()==n
        assert weekly.loc[weekly.query_strategy.eq(strategy),"token_count"].sum()==n
    assert articles=={"broad_business_control":5054,"anchors_stock_market_tag":2512}
    policy=json.loads((results/"cleaning_policy.json").read_text(encoding="utf-8"))
    diagnostics=json.loads((results/"preprocessing_diagnostics.json").read_text(encoding="utf-8"))
    assert policy["custom_stopwords_applied"]==[] and policy["lemmatization_applied"] is False
    for filename,sha in diagnostics["raw_hashes"].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==sha
    assert len(diagnostics["wordcloud_files"])==22 and all((ROOT/f).exists() for f in diagnostics["wordcloud_files"])
    result={"status":"passed","raw_columns_preserved_exactly":columns,"article_ids_verified":len(seen),
        "one_physical_record_per_article":True,"weekly_article_token_and_unique_vocabulary_verified":True,
        "frequency_totals_verified":True,"raw_file_hashes_unchanged":True,"no_custom_stopwords_or_lemmatization":True,
        "no_sentiment_WSI_FGI_columns":True,"network_requests":0}
    write_json(result,results/"validation_summary.json")
    current=ROOT/"results/execution_status.json"
    status=json.loads(current.read_text(encoding="utf-8"))
    status["code_status"]="Guardian preprocessing complete; awaiting cleaned-text review; no sentiment/WSI/FGI analysis"
    status["guardian_text_preprocessing"]={"status":"complete_pending_review","articles":len(seen),
        "primary_frequency":"weekly","robustness_frequency":"weekly","exact_visible_headline_description_duplicates":diagnostics["headline_description_exact_visible"],
        "near_duplicate_candidates":diagnostics["headline_description_near_duplicate"],"custom_stopwords_applied":[],
        "lemmatization_applied":False,"yearly_wordclouds":20,"yearly_panels":2,"weekly_descriptive_rows":len(weekly),
        "validation":"all original metadata columns and raw hashes preserved; weekly counts crosschecked",
        "downstream_sentiment_analysis_started":False}
    write_json(status,current)
    print("PASS: 5,265 IDs; all original columns/hash equality; strategy flags; clean tokens; 1,026 weekly article/token/unique-vocabulary rows; frequency totals; 22 wordcloud files; no scores or custom stopwords")


if __name__=="__main__":
    main()
