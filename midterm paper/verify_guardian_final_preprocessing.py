"""최종 전처리: 이전 자료와의 정확한 차이·정책 해시·주간/빈도/그림을 오프라인 검증."""
import hashlib
import json
import runpy
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_full_collection.py"))
import pyarrow.dataset as ds
import pandas as pd
from guardian_api_client import write_json

APPROVED={"happened","latest","coverage","rolling","live","news"}
CHANGED={"text_clean","tokens_clean","clean_token_count","clean_unique_token_count","text_policy_version"}


def stable_hash(row):
    return hashlib.sha256(json.dumps({k:v for k,v in row.items() if k not in CHANGED},sort_keys=True,ensure_ascii=False,default=lambda x:x.isoformat()).encode()).hexdigest()


def main():
    folder=ROOT/"results/guardian_text_final"
    baseline=ds.dataset(ROOT/"data/processed/guardian_text/articles",format="parquet",partitioning="hive")
    final=ds.dataset(ROOT/"data/processed/guardian_text_final/articles",format="parquet",partitioning="hive")
    assert baseline.schema.names==final.schema.names
    assert not {"compound","sentiment_label","WSI_count","fgi","relevant_us_equity_market"}.intersection(final.schema.names)
    before={}
    for batch in baseline.scanner(batch_size=200).to_batches():
        for row in batch.to_pylist():
            before[row["guardian_id"]]=(stable_hash(row),row["tokens_clean"])
    seen=set();tokens=defaultdict(Counter);week_docs=Counter();week_tokens=Counter();week_unique=defaultdict(set);empty=0
    for batch in final.scanner(batch_size=200).to_batches():
        for row in batch.to_pylist():
            identifier=row["guardian_id"]
            assert identifier not in seen
            seen.add(identifier)
            old_hash,old_tokens=before[identifier]
            assert stable_hash(row)==old_hash
            expected=[t for t in old_tokens if t not in APPROVED]
            assert row["tokens_clean"]==expected and row["text_clean"]==" ".join(expected)
            assert row["clean_token_count"]==len(expected) and row["clean_unique_token_count"]==len(set(expected))
            empty+=int(not expected)
            for strategy in ["broad_business_control","anchors_stock_market_tag"]:
                if row["in_"+strategy]:
                    tokens[strategy].update(expected)
                    key=(strategy,row["week_start"].isoformat())
                    week_docs[key]+=1;week_tokens[key]+=len(expected);week_unique[key].update(expected)
    assert seen==set(before) and len(seen)==5265
    frequencies=pd.read_csv(folder/"full_period_word_frequencies.csv")
    for strategy,counter in tokens.items():
        rows=frequencies.loc[frequencies.query_strategy.eq(strategy)]
        assert dict(zip(rows.term,rows["count"]))==dict(counter)
        assert not APPROVED.intersection(rows.term)
    bigrams=pd.read_csv(folder/"common_bigrams.csv")
    assert not any(APPROVED.intersection(phrase.split()) for phrase in bigrams.bigram)
    weekly=pd.read_csv(folder/"weekly_article_token_counts.csv")
    for row in weekly.itertuples():
        key=(row.query_strategy,row.week_start)
        assert row.article_count==week_docs[key] and row.token_count==week_tokens[key] and row.unique_token_count==len(week_unique[key])
    finance=pd.read_csv(folder/"financial_vocabulary_preservation.csv")
    assert finance.before_count.eq(finance.after_count).all()
    policy=json.loads((ROOT/"FINAL_PREPROCESSING_POLICY.json").read_text(encoding="utf-8"))
    expected_sha=policy.pop("policy_sha256")
    assert hashlib.sha256(json.dumps(policy,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==expected_sha
    assert policy["status"]=="FINAL_PREPROCESSING_SPECIFICATION" and set(policy["custom_stopwords_applied"])==APPROVED
    assert policy["lemmatization_applied"] is False and policy["phrase_level_deletion_rules"]==[]
    validation=json.loads((folder/"validation_summary.json").read_text(encoding="utf-8"))
    assert len(validation["wordcloud_files"])==25 and all((ROOT/path).exists() for path in validation["wordcloud_files"])
    for filename,sha in {**validation["raw_hashes"],**validation["baseline_hashes"]}.items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==sha
    result={"status":"passed","records_verified":len(seen),"all_non_clean_fields_including_raw_and_sentiment_identical":True,
        "tokens_exactly_baseline_minus_six_approved_stopwords":True,"frequency_bigrams_and_weekly_counts_verified":True,
        "financial_terms_preserved":True,"final_policy_hash_verified":expected_sha,"wordclouds_verified":25,
        "empty_final_clean_text":empty,"network_requests":0,"sentiment_scoring":False}
    write_json(result,folder/"independent_validation.json")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
