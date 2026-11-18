"""Guardian 원문 보존 전처리·기술통계. API·감성 모델·WSI·외부 자료 결합 없음."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import runpy
from collections import Counter,defaultdict
from datetime import timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_full_collection.py"))
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import regex
from guardian_api_client import write_json
import guardian_text_rules as rules

FLAGS={"broad_business_control":"in_broad_business_control","anchors_stock_market_tag":"in_anchors_stock_market_tag"}
OUTPUT=ROOT/"data"/"processed"/"guardian_text"
RESULTS=ROOT/"results"/"guardian_text_preprocessing"
ARTIFACT_TOKENS={"amp","nbsp","quot","apos","href","http","https","www","com","html","br","p","div","span","lt","gt"}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(rows,path,columns):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def term_rows(counter,docfreq,strategy,year=None,limit=None):
    ordered=sorted(counter.items(),key=lambda x:(-x[1],x[0]))
    for rank,(term,count) in enumerate(ordered[:limit] if limit else ordered,1):
        row={"query_strategy":strategy,"rank":rank,"term":term,"count":count,"document_count":docfreq[term]}
        if year is not None:
            row["year"]=year
        yield row


def assess_lemmatization(vocabulary):
    from nltk.stem import WordNetLemmatizer
    model=WordNetLemmatizer()
    mapping=[]
    naive_vocab=set()
    guarded_vocab=set()
    changed_mass=0
    for token,count in sorted(vocabulary.items(),key=lambda x:(-x[1],x[0])):
        word=bool(regex.fullmatch(r"\p{L}+",token))
        naive=model.lemmatize(token,pos="n") if word else token
        guarded=token if token in rules.PROTECTED else naive
        naive_vocab.add(naive)
        guarded_vocab.add(guarded)
        changed_mass+=count*int(guarded!=token)
        mapping.append({"token":token,"count":count,"naive_noun_lemma":naive,"protected_noun_lemma":guarded,
            "protected_token":token in rules.PROTECTED,"would_change":guarded!=token})
    write_table(mapping,RESULTS/"lemmatization_mapping.csv",["token","count","naive_noun_lemma","protected_noun_lemma","protected_token","would_change"])
    probes=[{"token":t,"noun_lemma":model.lemmatize(t,pos="n"),"verb_lemma":model.lemmatize(t,pos="v"),"protected":t in rules.PROTECTED}
            for t in ["us","fed","stocks","shares","equities","losses","earnings","bears","bulls","shorts","hodl","fucking"]]
    write_table(probes,RESULTS/"lemmatization_risk_probes.csv",["token","noun_lemma","verb_lemma","protected"])
    return {"scope":"all unique articles; diagnostics only","baseline_vocabulary":len(vocabulary),
        "naive_noun_lemma_vocabulary":len(naive_vocab),"protected_noun_lemma_vocabulary":len(guarded_vocab),
        "protected_noun_vocab_reduction_pct":100*(1-len(guarded_vocab)/len(vocabulary)) if vocabulary else 0,
        "tokens_that_would_change":changed_mass,"token_change_share":changed_mass/sum(vocabulary.values()) if vocabulary else 0,
        "applied_to_text_clean":False,"decision":"not applied: vocabulary compression is not evidence of semantic improvement; no POS/context validation; inspect mapping first"}


def refresh_review_candidates():
    """검토 후보만 갱신한다. 처리 Parquet/토큰/워드클라우드는 바꾸지 않는다."""
    dataset=ds.dataset(OUTPUT/"articles",format="parquet",partitioning="hive")
    tf=defaultdict(Counter);df=defaultdict(Counter);docs=Counter();examples=defaultdict(list)
    for batch in dataset.scanner(columns=["guardian_id","year","headline_for_text","tokens_clean",*FLAGS.values()],batch_size=200).to_batches():
        for record in batch.to_pylist():
            words=record["tokens_clean"]
            for strategy,flag in FLAGS.items():
                if record[flag]:
                    docs[strategy]+=1;tf[strategy].update(words);df[strategy].update(set(words))
                    for word in set(words)&rules.CANDIDATES.keys():
                        if len(examples[(strategy,word)])<3:
                            examples[(strategy,word)].append({"guardian_id":record["guardian_id"],"headline":rules.visible_text(record["headline_for_text"]),"year":record["year"]})
    rows=[{"query_strategy":s,"candidate":t,"count":tf[s][t],"document_count":df[s][t],"document_share":df[s][t]/docs[s],
           "reason":reason,"applied":False,"examples":json.dumps(examples[(s,t)],ensure_ascii=False)}
          for s in FLAGS for t,reason in rules.CANDIDATES.items() if tf[s][t]]
    write_table(rows,RESULTS/"candidate_custom_stopwords.csv",["query_strategy","candidate","count","document_count","document_share","reason","applied","examples"])
    from guardian_text_outputs import write_report
    write_report(ROOT,json.loads((RESULTS/"preprocessing_diagnostics.json").read_text(encoding="utf-8")))


def main():
    if rules.POLICY_VERSION=="guardian_text_v2_final":
        # v1 자료를 보존하고 최종 규칙은 별도 경로에 저장한다.
        from guardian_preprocessing_finalize import main as finalize
        return finalize()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    source=ROOT/"data"/"raw"/"news"/"guardian_full"/"articles"
    inputs=sorted(source.rglob("*.parquet"))
    hashes={str(p.relative_to(ROOT)):file_sha(p) for p in inputs}
    dataset=ds.dataset(source,format="parquet",partitioning="hive")
    assert not {"compound","sentiment_label","WSI_count","fgi","relevant_us_equity_market"}.intersection(dataset.schema.names)
    derived=[("headline_for_text",pa.string()),("headline_source",pa.string()),("text_raw",pa.string()),
        ("text_sentiment",pa.string()),("text_clean",pa.string()),("tokens_clean",pa.list_(pa.string())),
        ("raw_token_count",pa.int64()),("token_count_before_stopwords",pa.int64()),("clean_token_count",pa.int64()),("clean_unique_token_count",pa.int64()),
        ("headline_description_exact_raw",pa.bool_()),("headline_description_exact_visible",pa.bool_()),
        ("headline_description_near_duplicate",pa.bool_()),("headline_description_similarity",pa.float64()),
        ("text_policy_version",pa.string()),("week_id",pa.string()),("week_start",pa.date32()),("week_end",pa.date32())]
    schema=pa.schema([*dataset.schema,*[pa.field(name,dtype) for name,dtype in derived]])
    raw_tf=defaultdict(Counter); raw_df=defaultdict(Counter)
    clean_tf=defaultdict(Counter); clean_df=defaultdict(Counter)
    annual_tf=defaultdict(Counter); annual_df=defaultdict(Counter)
    bigrams=defaultdict(Counter); bigram_df=defaultdict(Counter)
    artifact_counts=defaultdict(Counter); boilerplate=Counter(); examples=defaultdict(list)
    weekly_docs=Counter(); weekly_tokens=Counter(); weekly_vocab=defaultdict(set)
    vocabulary=Counter(); totals=defaultdict(Counter); documents=Counter(); overall=Counter()
    duplicate_examples=[]; artifact_examples=[]; inspection=[]; seen_ids=set(); output_manifest=[]
    for year in range(2017,2027):
        folder=OUTPUT/"articles"/f"year={year}"
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/"part-00000.parquet"
        temporary=folder/"part-00000.parquet.tmp"
        n=0
        with pq.ParquetWriter(temporary,schema,compression="zstd") as writer:
            scanner=dataset.scanner(filter=ds.field("year")==year,batch_size=200)
            for batch in scanner.to_batches():
                processed=[]
                for original in batch.to_pylist():
                    identifier=original["guardian_id"]
                    assert identifier not in seen_ids
                    seen_ids.add(identifier)
                    text,pairs=rules.article_texts(original)
                    record={**original,**text}
                    date=original["date_et"]
                    start=date-timedelta(days=date.weekday())
                    iso=start.isocalendar()
                    record.update(week_start=start,week_end=start+timedelta(days=6),week_id=f"{iso.year}-W{iso.week:02d}")
                    processed.append(record)
                    raw_tokens=rules.tokenize(text["text_raw"])
                    visible_tokens=rules.tokenize(text["text_sentiment"])
                    tokens=text["tokens_clean"]
                    vocabulary.update(tokens)
                    overall["articles"]+=1
                    for field in ["headline_description_exact_raw","headline_description_exact_visible","headline_description_near_duplicate"]:
                        overall[field]+=int(text[field])
                    overall["headline_differs_from_web_title"]+=int(text["headline_for_text"]!=original["web_title"])
                    overall["empty_text_sentiment"]+=int(not text["text_sentiment"])
                    overall["empty_text_clean"]+=int(not tokens)
                    has_html=bool(rules.HTML_TAG.search(text["text_raw"]))
                    has_entity=bool(regex.search(r"&(?:#\d+|#x[0-9a-f]+|[a-z]+);",text["text_raw"],regex.I))
                    has_url=bool(rules.URL.search(text["text_raw"]))
                    overall["articles_with_html"]+=has_html
                    overall["articles_with_html_entities"]+=has_entity
                    overall["articles_with_urls"]+=has_url
                    if text["headline_description_exact_visible"] or text["headline_description_near_duplicate"]:
                        duplicate_examples.append({"guardian_id":identifier,"year":year,"headline":text["headline_for_text"],"description":original["description"],
                            "exact_raw":text["headline_description_exact_raw"],"exact_visible":text["headline_description_exact_visible"],
                            "near_duplicate":text["headline_description_near_duplicate"],"similarity":text["headline_description_similarity"]})
                    if has_html or has_entity or has_url:
                        artifact_examples.append({"guardian_id":identifier,"year":year,"has_html":has_html,"has_entity":has_entity,"has_url":has_url,
                            "text_raw":text["text_raw"],"text_sentiment":text["text_sentiment"]})
                    if n<2:
                        inspection.append({k:record[k] for k in ["guardian_id","year","published_at_original","date_et","week_id",*FLAGS.values(),"headline_for_text","description","text_raw","text_sentiment","text_clean"]})
                    for strategy,flag in FLAGS.items():
                        if not original[flag]:
                            continue
                        documents[strategy]+=1
                        raw_tf[strategy].update(raw_tokens); raw_df[strategy].update(set(raw_tokens))
                        clean_tf[strategy].update(tokens); clean_df[strategy].update(set(tokens))
                        annual_tf[(strategy,year)].update(tokens); annual_df[(strategy,year)].update(set(tokens))
                        bigrams[strategy].update(pairs); bigram_df[strategy].update(set(pairs))
                        key=(strategy,start)
                        weekly_docs[key]+=1; weekly_tokens[key]+=len(tokens); weekly_vocab[key].update(tokens)
                        for field,value in [("raw_tokens",len(raw_tokens)),("sentiment_visible_tokens",len(visible_tokens)),
                            ("before_stopwords_tokens",text["token_count_before_stopwords"]),("clean_tokens",len(tokens))]:
                            totals[strategy][field]+=value
                        for stage,words in [("raw",raw_tokens),("visible",visible_tokens),("clean",tokens)]:
                            artifact_counts[(strategy,stage)].update(w.lower() for w in words if w.lower() in ARTIFACT_TOKENS)
                        for phrase,pattern in rules.BOILERPLATE.items():
                            if regex.search(pattern,text["text_sentiment"],regex.I):
                                boilerplate[(strategy,phrase)]+=1
                        for word in set(tokens)&rules.CANDIDATES.keys():
                            if len(examples[(strategy,word)])<3:
                                examples[(strategy,word)].append({"guardian_id":identifier,"headline":rules.visible_text(text["headline_for_text"]),"year":year})
                    n+=1
                writer.write_table(pa.Table.from_pylist(processed,schema=schema))
        temporary.replace(path)
        output_manifest.append({"year":year,"articles":n,"file":str(path.relative_to(ROOT)),"sha256":file_sha(path)})
        print(f"Preprocessed {year}: {n} unique articles",flush=True)
    assert len(seen_ids)==5265
    assert {name:documents[name] for name in FLAGS}=={"broad_business_control":5054,"anchors_stock_market_tag":2512}
    assert hashes=={str(p.relative_to(ROOT)):file_sha(p) for p in inputs}
    word_columns=["query_strategy","rank","term","count","document_count"]
    for name,tf,df,limit in [("top_100_raw_tokens.csv",raw_tf,raw_df,100),("top_100_cleaned_tokens.csv",clean_tf,clean_df,100),
                             ("full_period_word_frequencies.csv",clean_tf,clean_df,None)]:
        write_table((row for strategy in FLAGS for row in term_rows(tf[strategy],df[strategy],strategy,limit=limit)),RESULTS/name,word_columns)
    for name,limit in [("yearly_word_frequencies.csv",None),("yearly_top_word_frequencies.csv",100)]:
        write_table((row for strategy in FLAGS for year in range(2017,2027) for row in term_rows(annual_tf[(strategy,year)],annual_df[(strategy,year)],strategy,year,limit)),
            RESULTS/name,["query_strategy","year","rank","term","count","document_count"])
    write_table(( {"query_strategy":strategy,"rank":rank,"bigram":" ".join(pair),"count":count,"document_count":bigram_df[strategy][pair]}
        for strategy in FLAGS for rank,(pair,count) in enumerate(sorted(bigrams[strategy].items(),key=lambda x:(-x[1],x[0]))[:100],1)),
        RESULTS/"common_bigrams.csv",["query_strategy","rank","bigram","count","document_count"])
    calendar=pd.read_csv(ROOT/"results/guardian_full/weekly_article_counts.csv",parse_dates=["week_start","week_end","observed_start","observed_end"])
    week_rows=[]
    for strategy in FLAGS:
        for row in calendar.to_dict("records"):
            start=row["week_start"].date()
            key=(strategy,start)
            assert weekly_docs[key]==row[strategy]
            week_rows.append({"query_strategy":strategy,"week_id":f"{row['iso_year']}-W{int(row['iso_week']):02d}",
                "week_start":start,"week_end":row["week_end"].date(),"days_observed":row["days_observed"],"is_complete_week":row["is_complete_week"],
                "article_count":weekly_docs[key],"token_count":weekly_tokens[key],"unique_token_count":len(weekly_vocab[key]),"token_representation":"tokens_clean"})
    write_table(week_rows,RESULTS/"weekly_article_token_counts.csv",list(week_rows[0]))
    token_summary=[{"query_strategy":s,"articles":documents[s],**totals[s],"tokens_removed_raw_to_clean_pct":100*(1-totals[s]["clean_tokens"]/totals[s]["raw_tokens"]),
                   "clean_vocabulary":len(clean_tf[s])} for s in FLAGS]
    write_table(token_summary,RESULTS/"token_count_summary.csv",list(token_summary[0]))
    write_table(duplicate_examples,RESULTS/"headline_description_duplicates.csv",["guardian_id","year","headline","description","exact_raw","exact_visible","near_duplicate","similarity"])
    write_table(artifact_examples,RESULTS/"html_web_artifact_examples.csv",["guardian_id","year","has_html","has_entity","has_url","text_raw","text_sentiment"])
    write_table(inspection,RESULTS/"preprocessing_inspection_sample.csv",list(inspection[0]))
    candidate_rows=[]
    for strategy in FLAGS:
        for token,reason in rules.CANDIDATES.items():
            if clean_tf[strategy][token]:
                candidate_rows.append({"query_strategy":strategy,"candidate":token,"count":clean_tf[strategy][token],"document_count":clean_df[strategy][token],
                    "document_share":clean_df[strategy][token]/documents[strategy],"reason":reason,"applied":False,
                    "examples":json.dumps(examples[(strategy,token)],ensure_ascii=False)})
    write_table(candidate_rows,RESULTS/"candidate_custom_stopwords.csv",["query_strategy","candidate","count","document_count","document_share","reason","applied","examples"])
    write_table(({"query_strategy":strategy,"token":token,"raw_count":artifact_counts[(strategy,'raw')][token],
                  "visible_count":artifact_counts[(strategy,'visible')][token],"clean_count":artifact_counts[(strategy,'clean')][token],
                  "note":"diagnostic only: some strings can be legitimate names/abbreviations; no custom removal"}
        for strategy in FLAGS for token in sorted(ARTIFACT_TOKENS)),RESULTS/"artifact_token_counts.csv",["query_strategy","token","raw_count","visible_count","clean_count","note"])
    write_table(({"query_strategy":s,"phrase":p,"document_count":boilerplate[(s,p)],"removed":False} for s in FLAGS for p in rules.BOILERPLATE),
        RESULTS/"boilerplate_phrase_counts.csv",["query_strategy","phrase","document_count","removed"])
    lemma=assess_lemmatization(vocabulary)
    write_json(lemma,RESULTS/"lemmatization_evaluation.json")
    policy={"version":rules.POLICY_VERSION,"headline_priority":["headline","web_title"],"concatenation":"headline + newline + description; omit description only if visible text exactly equals headline",
        "near_duplicate_detection":"token SequenceMatcher >=.90; >=5 tokens each; length ratio >=.80; flag only, no deletion",
        "sentiment":"HTML/entities/nonlinguistic whitespace only; case/punctuation/negation/slang/profanity/emojis retained; no stopwords",
        "frequency":"NFKC/lowercase, URLs removed, Unicode word/number/emoji tokenization; internal apostrophes/hyphens/tickers preserved",
        "market_normalization":{"U.S.":"us","U.K.":"uk","S&P 500":"s&p_500","FTSE 100":"ftse_100","FTSE 250":"ftse_250","Nasdaq 100":"nasdaq_100"},
        "baseline_stopword_source":"NLTK English stopwords","baseline_stopwords":sorted(rules.BASE_STOPWORDS),
        "protected_terms":sorted(rules.PROTECTED),"active_stopwords":sorted(rules.ACTIVE_STOPWORDS),"custom_stopwords_applied":[],
        "lemmatization_applied":False,"numerals_and_emojis":"retained in token tables; emojis can be excluded from wordcloud rendering only if font unsupported",
        "bigrams":"within original adjacent surviving tokens, same field and sentence; never across documents", "week":"Monday–Sunday, New York; partial boundary week retained and flagged",
        "resource_manifest":json.loads((ROOT/".nltk_data/resource_manifest.json").read_text(encoding="utf-8")),
        "versions":{name:importlib.metadata.version(name) for name in ["pandas","pyarrow","beautifulsoup4","nltk","regex","wordcloud"]}}
    write_json(policy,RESULTS/"cleaning_policy.json")
    from guardian_text_outputs import make_wordclouds,write_report
    clouds=make_wordclouds(ROOT,annual_tf,documents)
    diagnostics={**overall,"strategy_token_counts":token_summary,"custom_stopwords_applied":[],"lemmatization":lemma,
        "wordcloud_files":clouds,"physical_article_records":len(seen_ids),"output_manifest":output_manifest,
        "raw_parquet_unchanged":True,"raw_hashes":hashes,"weekly_rows":len(week_rows),"weekly_article_counts_match_raw":True,
        "network_requests":0,"sentiment_scoring":False,"WSI_constructed":False,"FGI_merged":False,
        "next_step":"review preprocessing outputs; no downstream analysis started"}
    write_json(diagnostics,RESULTS/"preprocessing_diagnostics.json")
    write_report(ROOT,diagnostics)
    print(json.dumps({"articles":len(seen_ids),"duplicates":{k:overall[k] for k in ["headline_description_exact_raw","headline_description_exact_visible","headline_description_near_duplicate"]},
        "strategy_token_counts":token_summary,"wordclouds":len(clouds),"global_lemmatization_applied":False},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
