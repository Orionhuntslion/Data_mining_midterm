"""Frozen text_sentiment -> unchanged VADER 3.3.2 -> separate weekly WSI datasets.

No collection, preprocessing, FGI reads, returns or regressions are performed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".packages"))
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from vaderSentiment import vaderSentiment as vader_module
from guardian_weekly_sentiment import FLAGS, classify, weekly_sentiment, descriptive_tables

SOURCE = ROOT / "data/processed/guardian_text_final/articles"
DESTINATION = ROOT / "data/processed/guardian_vader/articles"
RESULTS = ROOT / "results/guardian_sentiment"
FIGURES = ROOT / "figures/guardian_sentiment"
SCORE_COLUMNS = ["vader_neg", "vader_neu", "vader_pos", "vader_compound", "sentiment_class"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(frame, name):
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def protected_inputs():
    return [*sorted(SOURCE.rglob("*.parquet")), ROOT / "FINAL_PREPROCESSING_POLICY.json",
            ROOT / "FINAL_PREPROCESSING_POLICY.md", ROOT / "results/guardian_text_final/final_preprocessing_policy.json"]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    policy = json.loads((ROOT / "FINAL_PREPROCESSING_POLICY.json").read_text(encoding="utf-8"))
    expected = policy.pop("policy_sha256")
    assert hashlib.sha256(json.dumps(policy, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == expected
    assert policy["version"] == "guardian_text_v2_final" and not policy["lemmatization_applied"]
    assert importlib.metadata.version("vaderSentiment") == "3.3.2"
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected_inputs()}
    analyzer = vader_module.SentimentIntensityAnalyzer()
    module_folder = Path(vader_module.__file__).parent
    model = {
        "model": "VADER", "package": "vaderSentiment", "version": "3.3.2", "input": "text_sentiment",
        "input_policy": "guardian_text_v2_final", "input_policy_sha256": expected,
        "positive_threshold_inclusive": 0.05, "negative_threshold_inclusive": -0.05,
        "scoring_unit": "one complete preserved article text_sentiment per polarity_scores call",
        "lexicon_or_rule_modifications": False, "manual_label_overrides": False,
        "lexicon_hashes": {name: sha(module_folder / name) for name in ("vader_lexicon.txt", "emoji_utf8_lexicon.txt")},
        "versions": {p: importlib.metadata.version(p) for p in ("pandas", "pyarrow", "numpy", "matplotlib", "vaderSentiment")},
        "python_version": sys.version,
        "article_std": "sample standard deviation ddof=1",
        "article_skewness": "pandas bias-corrected sample skewness",
        "article_kurtosis": "pandas bias-corrected Fisher excess; Pearson = excess + 3",
        "weekly_std": "sample standard deviation ddof=1; undefined for n < 2",
        "week": "existing America/New_York Monday-Sunday; first partial week retained and flagged",
        "primary_WSI": "50 * (1 + (positive_count - negative_count) / article_count)",
        "robustness_WSI": "50 * (1 + mean_compound)",
        "empty_week": "counts zero; shares, sentiment, WSI, SE and intervals missing; no interpolation",
        "interval": "multinomial plug-in SE; WSI +/- 1.96*SE; approximate, not clipped to [0,100]",
        "interval_limitations": "articles can be dependent; few documents or unanimous labels can yield misleadingly narrow or zero-width intervals; not model or relevance uncertainty",
        "reference": "https://github.com/cjhutto/vaderSentiment/blob/master/README.rst",
        "no_FGI_merge_or_validation": True,
    }
    dataset = ds.dataset(SOURCE, format="parquet", partitioning="hive")
    assert not set(SCORE_COLUMNS).intersection(dataset.schema.names)
    assert dataset.count_rows() == 5265
    summaries, manifests, seen = [], [], set()
    score_schema = list(dataset.schema) + [pa.field(c, pa.float64()) for c in SCORE_COLUMNS[:4]] + [pa.field("sentiment_class", pa.string())]
    schema = pa.schema(score_schema, metadata={**(dataset.schema.metadata or {}),
        b"sentiment_model": b"vaderSentiment==3.3.2", b"input_policy_sha256": expected.encode()})
    diagnostics_columns = ["guardian_id", "web_title", "headline_for_text", "description", "text_sentiment",
        "published_at_utc", "published_at_et", "date_et", "week_start", "week_end", "week_id", "year", *FLAGS.values(), *SCORE_COLUMNS]
    for year in range(2017, 2027):
        folder = DESTINATION / f"year={year}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "part-00000.parquet"
        temp = folder / "part-00000.parquet.tmp"
        n = 0
        with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
            for batch in dataset.scanner(filter=ds.field("year") == year, batch_size=200).to_batches():
                records = []
                for row in batch.to_pylist():
                    assert row["guardian_id"] not in seen
                    seen.add(row["guardian_id"])
                    assert row["text_policy_version"] == "guardian_text_v2_final"
                    assert isinstance(row["text_sentiment"], str) and row["text_sentiment"].strip()
                    scores = analyzer.polarity_scores(row["text_sentiment"])
                    scored = {**row, **{"vader_" + k: float(v) for k, v in scores.items()},
                              "sentiment_class": classify(scores["compound"])}
                    records.append(scored)
                    summaries.append({k: scored[k] for k in diagnostics_columns})
                writer.write_table(pa.Table.from_pylist(records, schema=schema))
                n += len(records)
        temp.replace(path)
        manifests.append({"year": year, "articles": n, "path": str(path.relative_to(ROOT)), "sha256": sha(path)})
        print(f"VADER {year}: {n} articles", flush=True)
    articles = pd.DataFrame(summaries)
    assert len(articles) == len(seen) == 5265
    assert articles.vader_compound.between(-1, 1).all()
    assert articles[["vader_neg", "vader_neu", "vader_pos"]].sum(axis=1).sub(1).abs().le(0.00101).all()
    calendar = pd.read_csv(ROOT / "results/guardian_text_final/weekly_article_token_counts.csv")
    weekly = pd.concat([weekly_sentiment(articles, calendar, s) for s in FLAGS], ignore_index=True)
    weekly["week_start"] = weekly.week_start.dt.date
    for label, strategy in zip(("broad", "strict"), FLAGS):
        selected = weekly.loc[weekly.query_strategy.eq(strategy)]
        write_csv(selected, f"weekly_{label}_sentiment.csv")
        selected.to_parquet(RESULTS / f"weekly_{label}_sentiment.parquet", index=False)
        wsi_columns = ["query_strategy", "week_id", "week_start", "week_end", "days_observed", "is_complete_week", "article_count",
            "positive_count", "neutral_count", "negative_count", "sentiment_balance", "WSI_count", "WSI_continuous", "se_balance",
            "se_WSI_count", "WSI_count_ci95_low", "WSI_count_ci95_high", "small_sample_lt3", "small_sample_lt5",
            "unanimous_sentiment_week", "approx_interval_degenerate", "interval_method"]
        write_csv(selected[wsi_columns], f"weekly_{label}_wsi.csv")
        selected[wsi_columns].to_parquet(RESULTS / f"weekly_{label}_wsi.parquet", index=False)
    write_csv(weekly, "weekly_sentiment_all_strategies.csv")
    article_stats, class_stats, weekly_stats, comparison = descriptive_tables(articles, weekly)
    for frame, filename in [(article_stats, "article_descriptive_statistics.csv"), (class_stats, "sentiment_class_distribution.csv"),
                             (weekly_stats, "weekly_descriptive_statistics.csv"), (comparison, "broad_strict_wsi_comparison.csv")]:
        write_csv(frame, filename)
    from guardian_sentiment_outputs import inspection_sample, make_figures, write_report
    inspection_inventory = inspection_sample(articles, RESULTS)
    figures = make_figures(articles, weekly, class_stats, FIGURES)
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in protected_inputs()}
    write_json(model, RESULTS / "sentiment_model.json")
    write_json({"protected_input_sha256": hashes, "output_manifest": manifests}, RESULTS / "input_output_manifest.json")
    complete = weekly.loc[weekly.is_complete_week]
    validation = {"status": "pipeline_checks_passed", "physical_articles": len(articles),
        "articles_by_strategy": {s: int(articles[f].sum()) for s, f in FLAGS.items()},
        "overlap_articles": int(articles[list(FLAGS.values())].all(axis=1).sum()),
        "frozen_policy_and_source_parquet_unchanged": True, "input_policy_sha256": expected,
        "weeks_per_strategy_including_partial": int(len(weekly) / 2), "complete_weeks_per_strategy": int(len(complete) / 2),
        "empty_complete_weeks": {s: int(complete.loc[complete.query_strategy.eq(s)].article_count.eq(0).sum()) for s in FLAGS},
        "score_ranges_and_polarity_sum_checked": True, "empty_weeks_remain_missing": True,
        "figures": figures, "inspection_inventory": inspection_inventory,
        "guardian_requests": 0, "FGI_read_or_merged": False, "returns_or_regression": False}
    write_json(validation, RESULTS / "validation_summary.json")
    write_report(ROOT, model, article_stats, weekly_stats, comparison, validation)
    status_path = ROOT / "results/execution_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["code_status"] = "VADER and separate weekly count/continuous WSI complete; awaiting sentiment review; no FGI or return analysis"
    status["guardian_vader_weekly_WSI"] = {"status": "diagnostics_complete", "articles": len(articles),
        "model": "vaderSentiment==3.3.2", "preprocessing_policy_unchanged": True, "FGI_merged": False,
        "lead_lag_or_market_analysis_started": False}
    write_json(status, status_path)
    print(article_stats.to_string(index=False))
    print(weekly_stats.to_string(index=False))
    print(comparison.to_string(index=False))
    print("Stopped after VADER / weekly WSI diagnostics; no FGI or market analysis.", flush=True)


if __name__ == "__main__":
    main()
