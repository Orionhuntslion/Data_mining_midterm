"""06. 빈도/문서빈도, 국면별 TF-IDF, 단어 시계열, 수동 검토 후보."""
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from midterm_common import REGIMES, read_csv, save_csv, save_json


def frequency_table(texts):
    counts, doccounts = Counter(), Counter()
    for text in texts.fillna(""):
        tokens = text.split()
        counts.update(tokens)
        doccounts.update(set(tokens))
    total = sum(counts.values())
    n = len(texts)
    rows = [{"term": term, "count": count, "document_count": doccounts[term],
             "document_share": doccounts[term] / n if n else float("nan"),
             "per_1000_tokens": count / total * 1000 if total else float("nan")}
            for term, count in counts.most_common()]
    return pd.DataFrame(rows, columns=["term", "count", "document_count", "document_share", "per_1000_tokens"])


def run(cfg):
    docs = read_csv(cfg.processed / "regime_text_data.csv", ["date"])
    overall = frequency_table(docs.frequency_text)
    if overall.empty:
        raise ValueError("빈도분석에 사용할 토큰이 없습니다. 전처리/원문을 확인하세요.")
    save_csv(overall, cfg.results / "word_frequencies.csv")
    regime_tables = []
    for regime in REGIMES:
        table = frequency_table(docs.loc[docs.regime.eq(regime), "frequency_text"])
        table.insert(0, "regime", regime)
        regime_tables.append(table)
    save_csv(pd.concat(regime_tables, ignore_index=True), cfg.results / "regime_word_frequencies.csv")
    # Extreme 포함 Fear vs Greed. 중립 및 FGI 없는 날짜는 여기서 제외한다.
    comparisons = []
    for group, regimes in [("Fear", REGIMES[:2]), ("Greed", REGIMES[3:])]:
        table = frequency_table(docs.loc[docs.regime.isin(regimes), "frequency_text"])
        table.insert(0, "group", group)
        comparisons.append(table)
    save_csv(pd.concat(comparisons, ignore_index=True), cfg.results / "fear_greed_word_comparison.csv")

    tfidf_rows = []
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
        ngram_range=(1, 2), min_df=2 if len(docs) >= 20 else 1, max_features=15000)
    matrix = vectorizer.fit_transform(docs.frequency_text.fillna(""))
    names = vectorizer.get_feature_names_out()
    for regime in REGIMES:
        mask = docs.regime.eq(regime).to_numpy()
        if not mask.any():
            continue
        means = matrix[mask].mean(axis=0).A1
        order = means.argsort()[::-1]
        for idx in order[:cfg.top_n]:
            if means[idx] > 0:
                tfidf_rows.append({"regime": regime, "term": names[idx], "mean_tfidf": means[idx], "n_docs": int(mask.sum())})
    save_csv(pd.DataFrame(tfidf_rows, columns=["regime", "term", "mean_tfidf", "n_docs"]), cfg.results / "regime_tfidf.csv")
    save_json({"unit": "one document", "aggregation": "mean L2-normalized document TF-IDF within regime",
        "fit_on": "all deduplicated documents (including dates without FGI)", "ngram_range": [1,2],
        "min_df": vectorizer.min_df, "max_features": vectorizer.max_features,
        "vocabulary_size": len(names)}, cfg.results / "tfidf_method.json")

    # FGI와의 상관을 보지 않고 전 기간 문서빈도 순서로 고정 선택한다.
    selected = overall.sort_values(["document_count", "term"], ascending=[False, True]).head(cfg.timing_terms_n).copy()
    selected["selection_rule"] = "global_document_frequency_desc_then_term_asc; no FGI optimization"
    save_csv(selected, cfg.results / "selected_timing_terms.csv")
    token_lists = docs.frequency_text.fillna("").map(str.split)
    daily_rows = []
    for term in selected.term:
        term_docs = pd.DataFrame({"date": docs.date, "frequency": token_lists.map(lambda x: x.count(term)),
                                  "document_count": token_lists.map(lambda x: int(term in x))})
        day = term_docs.groupby("date").agg(frequency=("frequency", "sum"), document_count=("document_count", "sum"), n_docs=("frequency", "size"))
        day = day.reindex(cfg.calendar)
        day["n_docs"] = day.n_docs.fillna(0).astype(int)
        # 無뉴스일은 단어빈도 0으로 만들어 상관에 넣지 않는다.
        day["document_share"] = day.document_count / day.n_docs.replace(0, float("nan"))
        day["term"] = term
        daily_rows.append(day.reset_index())
    word_daily = pd.concat(daily_rows, ignore_index=True)
    save_csv(word_daily, cfg.results / "word_daily_patterns.csv")
    word_regimes = []
    for regime in REGIMES:
        ids = docs.index[docs.regime.eq(regime)]
        for term in selected.term:
            entries = token_lists.loc[ids]
            word_regimes.append({"regime": regime, "term": term, "n_docs": len(ids),
                "frequency": int(entries.map(lambda x: x.count(term)).sum()),
                "document_count": int(entries.map(lambda x: term in x).sum()),
                "document_share": entries.map(lambda x: term in x).mean()})
    save_csv(pd.DataFrame(word_regimes), cfg.results / "word_regime_patterns.csv")

    # 국면과 반대 방향의 점수는 '모델 오류'의 정답 라벨이 아니다.
    disagree = ((docs.regime.isin(REGIMES[:2]) & docs.compound.ge(0.05)) |
                (docs.regime.isin(REGIMES[3:]) & docs.compound.le(-0.05)))
    candidates = docs.loc[disagree].copy()
    candidates["review_reason"] = "opposite_to_daily_FGI; not_verified_model_error"
    candidates["abs_compound"] = candidates.compound.abs()
    candidates = candidates.sort_values(["regime", "abs_compound", "doc_id"], ascending=[True, False, True]).groupby("regime", sort=False).head(20)
    save_csv(candidates, cfg.results / "regime_disagreement_candidates.csv")
    sample = docs.assign(review_regime=docs.regime.fillna("Unmatched"))
    sample = sample.groupby("review_regime", group_keys=False)[list(sample.columns)].apply(
        lambda x: x.sample(n=min(20,len(x)), random_state=cfg.seed)).reset_index(drop=True)
    sample["human_sentiment"] = ""
    sample["slang_or_irony"] = ""
    sample["review_notes"] = ""
    save_csv(sample, cfg.results / "human_review_template.csv")
    print(f"빈도 토큰 {len(overall):,}개, TF-IDF 특징 {len(names):,}개, 사전 선택 단어 {len(selected)}개")
    return overall
