"""승인된 빈도 불용어 6개만 적용하고 전처리 명세를 동결한다. 감성 모델 호출 없음."""
from __future__ import annotations

import hashlib
import json
import runpy
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_full_collection.py"))
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import regex
from guardian_api_client import write_json
from guardian_text_preprocess import write_table,term_rows,file_sha
import guardian_text_rules as rules

BEFORE=ROOT/"data/processed/guardian_text/articles"
AFTER=ROOT/"data/processed/guardian_text_final/articles"
RESULTS=ROOT/"results/guardian_text_final"
PREVIOUS=ROOT/"results/guardian_text_preprocessing"
FLAGS={"broad_business_control":"in_broad_business_control","anchors_stock_market_tag":"in_anchors_stock_market_tag"}
APPROVED=frozenset({"happened","latest","coverage","rolling","live","news"})
KEPT={"new","year","business","says"}
FINANCE={"equities":["stock","stocks","share","shares","equity","equities"],"market":["market","markets"],
    "Fed/Federal Reserve":["fed","federal","reserve"],"interest rates":["interest","rate","rates"],
    "inflation":["inflation"],"oil":["oil"],"tariffs/trade":["tariff","tariffs","trade"],"banks":["bank","banks"],
    "technology":["technology","tech"],"S&P 500":["s&p_500","s&p"],"Nasdaq":["nasdaq","nasdaq_100"],"Dow":["dow","jones"],"US":["us"]}
PHRASES={**rules.BOILERPLATE,"rolling coverage":r"\brolling coverage\b",
    "economic and financial":r"\beconomic and financial\b","financial and economic":r"\bfinancial and economic\b",
    "standard rolling-news description":r"rolling coverage of the latest economic and financial news"}


def sequence_present(tokens,sequence):
    return bool(sequence) and any(tokens[i:i+len(sequence)]==sequence for i in range(len(tokens)-len(sequence)+1))


def baseline_tokens(text):
    return [t for t in rules.tokenize(rules.frequency_input(rules.visible_text(text))) if t not in rules.ACTIVE_STOPWORDS]


def main():
    assert rules.CUSTOM_STOPWORDS==APPROVED and not KEPT.intersection(APPROVED|rules.ACTIVE_STOPWORDS)
    previous_policy=json.loads((PREVIOUS/"cleaning_policy.json").read_text(encoding="utf-8"))
    assert previous_policy["custom_stopwords_applied"]==[] and previous_policy["lemmatization_applied"] is False
    assert set(previous_policy["active_stopwords"])==rules.ACTIVE_STOPWORDS
    if "token_pattern" in previous_policy:
        assert previous_policy["token_pattern"]==rules.TOKEN_PATTERN.pattern
    RESULTS.mkdir(parents=True,exist_ok=True)
    AFTER.mkdir(parents=True,exist_ok=True)
    baseline_hashes={str(p.relative_to(ROOT)):file_sha(p) for p in sorted(BEFORE.rglob("*.parquet"))}
    raw_hashes=json.loads((PREVIOUS/"preprocessing_diagnostics.json").read_text(encoding="utf-8"))["raw_hashes"]
    source=ds.dataset(BEFORE,format="parquet",partitioning="hive")
    before_tf=defaultdict(Counter);after_tf=defaultdict(Counter);after_df=defaultdict(Counter)
    annual_tf=defaultdict(Counter);annual_df=defaultdict(Counter);bigrams=defaultdict(Counter);bigram_df=defaultdict(Counter)
    docs=Counter();weekly_docs=Counter();weekly_tokens=Counter();weekly_vocab=defaultdict(set)
    description_counts=defaultdict(Counter);phrase_counts=Counter();residue_counts=Counter();sentiment_digests={}
    inspection=[];ids=set();manifest=[]
    for year in range(2017,2027):
        folder=AFTER/f"year={year}"
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/"part-00000.parquet"
        temporary=folder/"part-00000.parquet.tmp"
        n=0
        with pq.ParquetWriter(temporary,source.schema,compression="zstd") as writer:
            for batch in source.scanner(filter=ds.field("year")==year,batch_size=200).to_batches():
                rows=[]
                for before in batch.to_pylist():
                    identifier=before["guardian_id"]
                    assert identifier not in ids and before["text_policy_version"]=="guardian_text_v1"
                    ids.add(identifier)
                    old=before["tokens_clean"]
                    new=[t for t in old if t not in APPROVED]
                    assert new==rules.clean_tokens(before["text_sentiment"])[1]
                    regenerated,pairs=rules.article_texts(before)
                    assert regenerated["text_raw"]==before["text_raw"] and regenerated["text_sentiment"]==before["text_sentiment"]
                    after={**before,"tokens_clean":new,"text_clean":" ".join(new),"clean_token_count":len(new),
                        "clean_unique_token_count":len(set(new)),"text_policy_version":rules.POLICY_VERSION}
                    changed={"tokens_clean","text_clean","clean_token_count","clean_unique_token_count","text_policy_version"}
                    assert all(after[k]==before[k] for k in before if k not in changed)
                    sentiment_digests[identifier]=hashlib.sha256(json.dumps([before["text_raw"],before["text_sentiment"]],ensure_ascii=False).encode()).hexdigest()
                    rows.append(after)
                    if n<3:
                        inspection.append({"guardian_id":identifier,"year":year,"text_raw":after["text_raw"],"text_sentiment":after["text_sentiment"],
                            "text_clean_before":before["text_clean"],"text_clean_after":after["text_clean"]})
                    for strategy,flag in FLAGS.items():
                        if not before[flag]:
                            continue
                        docs[strategy]+=1
                        before_tf[strategy].update(old);after_tf[strategy].update(new);after_df[strategy].update(set(new))
                        annual_tf[(strategy,year)].update(new);annual_df[(strategy,year)].update(set(new))
                        bigrams[strategy].update(pairs);bigram_df[strategy].update(set(pairs))
                        week=(strategy,before["week_start"].isoformat())
                        weekly_docs[week]+=1;weekly_tokens[week]+=len(new);weekly_vocab[week].update(new)
                        description_counts[strategy].update([rules.visible_text(before.get("description") or "")])
                        for label,pattern in PHRASES.items():
                            if regex.search(pattern,before["text_sentiment"],regex.I):
                                phrase_counts[(strategy,label)]+=1
                                remaining=[t for t in baseline_tokens(label if label!="standard rolling-news description" else "Rolling coverage of the latest economic and financial news") if t not in APPROVED]
                                residue_counts[(strategy,label)]+=int(sequence_present(new,remaining))
                    n+=1
                writer.write_table(pa.Table.from_pylist(rows,schema=source.schema))
        temporary.replace(path)
        manifest.append({"year":year,"articles":n,"file":str(path.relative_to(ROOT)),"sha256":file_sha(path)})
        print(f"Final clean text {year}: {n} articles",flush=True)
    assert len(ids)==5265
    assert baseline_hashes=={str(p.relative_to(ROOT)):file_sha(p) for p in sorted(BEFORE.rglob("*.parquet"))}
    assert all(file_sha(ROOT/path)==sha for path,sha in raw_hashes.items())
    comparisons=[];removed=[];top_before=[];top_after=[];entrants=[];finance=[]
    for strategy in FLAGS:
        before,after=before_tf[strategy],after_tf[strategy]
        assert all(after[t]==0 for t in APPROVED)
        assert all(after[t]==count for t,count in before.items() if t not in APPROVED)
        assert set(before)-set(after)==APPROVED
        old_top=[t for t,_ in sorted(before.items(),key=lambda x:(-x[1],x[0]))[:30]]
        new_top=[t for t,_ in sorted(after.items(),key=lambda x:(-x[1],x[0]))[:30]]
        for rank,token in enumerate(old_top,1):
            top_before.append({"query_strategy":strategy,"rank":rank,"term":token,"count":before[token]})
        for rank,token in enumerate(new_top,1):
            row={"query_strategy":strategy,"rank":rank,"term":token,"count":after[token],"entered_top30":token not in old_top}
            top_after.append(row)
            if row["entered_top30"]:
                entrants.append(row)
        comparisons.append({"query_strategy":strategy,"articles":docs[strategy],"clean_tokens_before":sum(before.values()),
            "clean_tokens_after":sum(after.values()),"tokens_removed":sum(before[t] for t in APPROVED),
            "removed_pct":100*sum(before[t] for t in APPROVED)/sum(before.values()),
            "unique_tokens_before":len(before),"unique_tokens_after":len(after),"new_top30_terms":" | ".join(t for t in new_top if t not in old_top)})
        for t in sorted(APPROVED):
            removed.append({"query_strategy":strategy,"term":t,"before_count":before[t],"after_count":after[t]})
        for group,terms in {**FINANCE,"explicitly_kept_non_custom":sorted(KEPT)}.items():
            for token in terms:
                finance.append({"query_strategy":strategy,"concept":group,"term":token,"before_count":before[token],"after_count":after[token],
                    "status":"preserved" if before[token]>0 else "not_present; not_inserted"})
    write_table(comparisons,RESULTS/"custom_stopword_before_after.csv",list(comparisons[0]))
    write_table(removed,RESULTS/"custom_stopword_removed_counts.csv",list(removed[0]))
    write_table(top_before,RESULTS/"top30_before.csv",["query_strategy","rank","term","count"])
    write_table(top_after,RESULTS/"top30_after.csv",["query_strategy","rank","term","count","entered_top30"])
    write_table(entrants,RESULTS/"top30_new_entries.csv",["query_strategy","rank","term","count","entered_top30"])
    write_table(finance,RESULTS/"financial_vocabulary_preservation.csv",list(finance[0]))
    columns=["query_strategy","rank","term","count","document_count"]
    for filename,limit in [("full_period_word_frequencies.csv",None),("top_100_cleaned_tokens.csv",100)]:
        write_table((row for s in FLAGS for row in term_rows(after_tf[s],after_df[s],s,limit=limit)),RESULTS/filename,columns)
    for filename,limit in [("yearly_word_frequencies.csv",None),("yearly_top_word_frequencies.csv",100)]:
        write_table((row for s in FLAGS for year in range(2017,2027) for row in term_rows(annual_tf[(s,year)],annual_df[(s,year)],s,year,limit)),
            RESULTS/filename,["query_strategy","year","rank","term","count","document_count"])
    yearly_top=pd.read_csv(RESULTS/"yearly_top_word_frequencies.csv")
    yearly_top.loc[yearly_top.year.isin([2017,2020,2026])].to_csv(RESULTS/"representative_year_word_frequencies.csv",index=False,encoding="utf-8-sig")
    write_table(( {"query_strategy":s,"rank":rank,"bigram":" ".join(pair),"count":count,"document_count":bigram_df[s][pair]}
        for s in FLAGS for rank,(pair,count) in enumerate(sorted(bigrams[s].items(),key=lambda x:(-x[1],x[0]))[:100],1)),
        RESULTS/"common_bigrams.csv",["query_strategy","rank","bigram","count","document_count"])
    for s in FLAGS:
        assert all(not APPROVED.intersection(pair) for pair in bigrams[s])
    weekly=pd.read_csv(PREVIOUS/"weekly_article_token_counts.csv")
    for index,row in weekly.iterrows():
        key=(row.query_strategy,row.week_start)
        assert row.article_count==weekly_docs[key]
        weekly.at[index,"token_count"]=weekly_tokens[key]
        weekly.at[index,"unique_token_count"]=len(weekly_vocab[key])
    weekly.to_csv(RESULTS/"weekly_article_token_counts.csv",index=False,encoding="utf-8-sig")
    write_table(inspection,RESULTS/"custom_stopword_inspection.csv",list(inspection[0]))
    # 남은 상용구는 관측 증거만 저장한다. 추가 단어/문구 삭제는 수행하지 않는다.
    phrase_rows=[];template_rows=[]
    for s in FLAGS:
        for label,pattern in PHRASES.items():
            text=label if label!="standard rolling-news description" else "Rolling coverage of the latest economic and financial news"
            old=baseline_tokens(text);new=[t for t in old if t not in APPROVED]
            phrase_rows.append({"query_strategy":s,"phrase":label,"source_sentiment_article_count":phrase_counts[(s,label)],
                "frequency_tokens_before":" ".join(old),"frequency_tokens_after":" ".join(new),
                "source_articles_retaining_residue":residue_counts[(s,label)],"additional_phrase_deletion_applied":False})
        for text,count in description_counts[s].most_common(20):
            if count>=3 and text:
                old=baseline_tokens(text);new=[t for t in old if t not in APPROVED]
                template_rows.append({"query_strategy":s,"description_visible":text,"identical_description_articles":count,
                    "text_clean_before":" ".join(old),"text_clean_after":" ".join(new),"additional_deletion_applied":False})
    write_table(phrase_rows,RESULTS/"phrase_artifact_review.csv",list(phrase_rows[0]))
    write_table(template_rows,RESULTS/"repeated_description_templates.csv",["query_strategy","description_visible","identical_description_articles","text_clean_before","text_clean_after","additional_deletion_applied"])
    write_json({"before":previous_policy,"approved_custom_stopwords":sorted(APPROVED),"explicitly_not_added":sorted(KEPT)},RESULTS/"before_policy_snapshot.json")
    from guardian_text_outputs import make_wordclouds
    clouds=make_wordclouds(ROOT,annual_tf,docs,figure_directory=ROOT/"figures/guardian_text_final",result_directory=RESULTS,
        include_full_period=True,representative_years=[2017,2020,2026])
    # 저장된 최종 파일도 다시 읽어 텍스트 불변/정확한 토큰 차이/한 레코드 원칙을 검증한다.
    verified=set()
    for batch in ds.dataset(AFTER,format="parquet",partitioning="hive").scanner(columns=["guardian_id","text_raw","text_sentiment","tokens_clean","text_clean","text_policy_version"],batch_size=200).to_batches():
        for row in batch.to_pylist():
            identifier=row["guardian_id"]
            assert identifier not in verified
            verified.add(identifier)
            assert hashlib.sha256(json.dumps([row["text_raw"],row["text_sentiment"]],ensure_ascii=False).encode()).hexdigest()==sentiment_digests[identifier]
            assert not APPROVED.intersection(row["tokens_clean"]) and row["text_clean"]==" ".join(row["tokens_clean"])
            assert row["text_policy_version"]==rules.POLICY_VERSION
    assert verified==ids
    policy={**previous_policy,"version":rules.POLICY_VERSION,"status":"FINAL_PREPROCESSING_SPECIFICATION",
        "custom_stopwords_applied":sorted(APPROVED),"explicitly_not_custom_stopwords":sorted(KEPT),
        "custom_stopword_scope":"text_clean and its token/frequency/bigram/cloud/weekly-count derivatives only",
        "lemmatization_applied":False,"stemming_applied":False,"token_pattern":rules.TOKEN_PATTERN.pattern,"protected_finance_terms":sorted(rules.PROTECTED),
        "financial_vocabulary_audit":FINANCE,
        "sentiment_text_unchanged_from_v1":True,"raw_text_unchanged_from_v1":True,
        "phrase_level_deletion_rules":[],"bigrams":"original adjacency within document, field and sentence; both base and six custom stopwords excluded; no bridging",
        "output_dataset":str(AFTER.relative_to(ROOT)),"baseline_dataset":str(BEFORE.relative_to(ROOT)),
        "representative_years":[2017,2020,2026],"all_years_preserved":list(range(2017,2027)),
        "validation":"passed on all 5265 records; all non-custom token counts unchanged; raw/sentiment identical; original metadata and flags preserved",
        "policy_changes_require":"explicit policy revision; do not silently change this specification"}
    canonical=json.dumps(policy,sort_keys=True,ensure_ascii=False).encode()
    policy["policy_sha256"]=hashlib.sha256(canonical).hexdigest()
    write_json(policy,ROOT/"FINAL_PREPROCESSING_POLICY.json")
    write_json(policy,RESULTS/"final_preprocessing_policy.json")
    validation={"status":"passed","physical_articles":len(ids),"strategy_article_counts":dict(docs),
        "only_six_approved_tokens_removed":True,"all_other_token_counts_unchanged":True,"raw_and_sentiment_text_unchanged":True,
        "raw_parquet_and_v1_baseline_unchanged":True,"baseline_hashes":baseline_hashes,"raw_hashes":raw_hashes,
        "financial_vocabulary_preserved":True,"new_year_business_says_preserved":True,"bigrams_exclude_approved_stopwords":True,
        "no_phrase_level_rules_added":True,"lemmatization":False,"policy_sha256":policy["policy_sha256"],"wordcloud_files":clouds,
        "output_manifest":manifest,"network_requests":0,"sentiment_scoring":False,"WSI_constructed":False,"FGI_merged":False}
    write_json(validation,RESULTS/"validation_summary.json")
    from guardian_final_review import write_review
    write_review(ROOT,policy,comparisons,entrants,phrase_rows,template_rows,validation)
    current=ROOT/"results/execution_status.json"
    status=json.loads(current.read_text(encoding="utf-8"))
    status["code_status"]="Final preprocessing specification frozen; awaiting cleaned-vocabulary review; no sentiment or WSI"
    status["guardian_final_preprocessing"]={"status":"validated_and_frozen","version":rules.POLICY_VERSION,
        "policy_sha256":policy["policy_sha256"],"custom_stopwords":sorted(APPROVED),"lemmatization":False,
        "articles":len(ids),"wordcloud_files":len(clouds),"raw_and_sentiment_unchanged":True,"downstream_sentiment_analysis_started":False}
    write_json(status,current)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print("Final preprocessing frozen; no sentiment scoring or downstream analysis.")


if __name__=="__main__":
    main()
