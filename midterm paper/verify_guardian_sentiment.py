"""Independent full-data verification plus boundary/empty-week formula checks."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".packages"))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from guardian_weekly_sentiment import FLAGS, classify, weekly_sentiment, SENTIMENT_COLUMNS

FOLDER = ROOT / "results/guardian_sentiment"


def close(a, b):
    assert np.isclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True), (a, b)


def synthetic_checks():
    assert [classify(x) for x in [-1, -0.05, -0.0499, 0, 0.0499, 0.05, 1]] == [
        "negative", "negative", "neutral", "neutral", "neutral", "positive", "positive"]
    dates = pd.date_range("2026-01-05", periods=3, freq="7D")
    strategy = "broad_business_control"
    calendar = pd.DataFrame({"query_strategy": strategy, "week_id": ["2026-W02", "2026-W03", "2026-W04"],
        "week_start": dates, "week_end": dates + pd.Timedelta(days=6), "days_observed": 7,
        "is_complete_week": True, "article_count": [4, 0, 1]})
    articles = pd.DataFrame({"week_start": [dates[0]] * 4 + [dates[2]], "vader_compound": [0.7, 0.2, 0.0, -0.6, 0.5],
        "sentiment_class": ["positive", "positive", "neutral", "negative", "positive"], FLAGS[strategy]: True})
    result = weekly_sentiment(articles, calendar, strategy)
    close(result.loc[0, "WSI_count"], 62.5)
    close(result.loc[0, "WSI_continuous"], 53.75)
    close(result.loc[0, "se_WSI_count"], 50 * math.sqrt((0.5 + 0.25 - 0.25 ** 2) / 4))
    assert result.loc[1, SENTIMENT_COLUMNS].isna().all()
    assert result.loc[2, "approx_interval_degenerate"] and pd.isna(result.loc[2, "std_compound"])
    assert result.loc[2, "WSI_count_ci95_low"] == result.loc[2, "WSI_count_ci95_high"] == 100


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
        default=lambda x: x.isoformat()).encode()).hexdigest()


def main():
    synthetic_checks()
    provenance = json.loads((FOLDER / "input_output_manifest.json").read_text(encoding="utf-8"))
    for path, digest in provenance["protected_input_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for entry in provenance["output_manifest"]:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    source = ds.dataset(ROOT / "data/processed/guardian_text_final/articles", format="parquet", partitioning="hive")
    scored = ds.dataset(ROOT / "data/processed/guardian_vader/articles", format="parquet", partitioning="hive")
    assert scored.schema.names == source.schema.names + ["vader_neg", "vader_neu", "vader_pos", "vader_compound", "sentiment_class"]
    for name in source.schema.names:
        assert scored.schema.field(name).type == source.schema.field(name).type
    originals = {}
    for batch in source.scanner(batch_size=200).to_batches():
        for row in batch.to_pylist():
            assert row["guardian_id"] not in originals
            originals[row["guardian_id"]] = row_hash(row)
    analyzer = SentimentIntensityAnalyzer()
    seen, small = set(), []
    for batch in scored.scanner(batch_size=200).to_batches():
        for row in batch.to_pylist():
            identifier = row["guardian_id"]
            assert identifier not in seen
            seen.add(identifier)
            assert row_hash({k: row[k] for k in source.schema.names}) == originals[identifier]
            expected_scores = analyzer.polarity_scores(row["text_sentiment"])
            assert all(row["vader_" + key] == value for key, value in expected_scores.items())
            expected_class = "positive" if row["vader_compound"] >= 0.05 else "negative" if row["vader_compound"] <= -0.05 else "neutral"
            assert row["sentiment_class"] == expected_class
            assert -1 <= row["vader_compound"] <= 1
            assert all(0 <= row["vader_" + c] <= 1 for c in ["neg", "neu", "pos"])
            local = row["published_at_utc"].astimezone(row["published_at_et"].tzinfo)
            assert local == row["published_at_et"] and local.date() == row["date_et"]
            assert row["week_start"].weekday() == 0 and row["week_end"].weekday() == 6
            assert (row["date_et"] - row["week_start"]).days == row["date_et"].weekday()
            iso = row["date_et"].isocalendar()
            assert row["week_id"] == f"{iso.year}-W{iso.week:02d}"
            small.append({k: row[k] for k in ["guardian_id", "vader_compound", "sentiment_class", "week_start", *FLAGS.values()]})
    assert seen == set(originals) and len(seen) == 5265
    articles = pd.DataFrame(small)
    stats = pd.read_csv(FOLDER / "article_descriptive_statistics.csv").set_index("query_strategy")
    all_weeks = []
    empty = {}
    for label, (strategy, flag) in zip(["broad", "strict"], FLAGS.items()):
        subset = articles.loc[articles[flag]]
        values = subset.vader_compound.tolist()
        assert len(subset) == stats.loc[strategy, "article_count"]
        mean, sd, n = statistics.mean(values), statistics.stdev(values), len(values)
        close(stats.loc[strategy, "mean_compound"], mean)
        close(stats.loc[strategy, "median_compound"], statistics.median(values))
        close(stats.loc[strategy, "std_compound"], sd)
        skew = n / ((n - 1) * (n - 2)) * sum(((x - mean) / sd) ** 3 for x in values)
        kurt = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * sum(((x - mean) / sd) ** 4 for x in values) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        close(stats.loc[strategy, "skewness"], skew)
        close(stats.loc[strategy, "excess_kurtosis"], kurt)
        counts = subset.sentiment_class.value_counts()
        for category in ["positive", "neutral", "negative"]:
            assert stats.loc[strategy, category + "_count"] == counts[category]
            close(stats.loc[strategy, category + "_share"], counts[category] / n)
        weeks = pd.read_csv(FOLDER / f"weekly_{label}_sentiment.csv")
        parquet = pd.read_parquet(FOLDER / f"weekly_{label}_sentiment.parquet")
        assert len(weeks) == len(parquet) == 513 and weeks.is_complete_week.sum() == 512
        assert weeks.week_start.is_unique
        assert pd.to_datetime(weeks.week_start).diff().dropna().eq(pd.Timedelta(days=7)).all()
        for column in SENTIMENT_COLUMNS:
            assert np.allclose(weeks[column], parquet[column], equal_nan=True)
        groups = {key.isoformat(): rows for key, rows in subset.groupby("week_start")}
        for row in weeks.to_dict("records"):
            group = groups.get(row["week_start"])
            n = len(group) if group is not None else 0
            assert row["article_count"] == n
            if n == 0:
                assert all(row[c + "_count"] == 0 for c in ["positive", "neutral", "negative"])
                assert all(pd.isna(row[c]) for c in SENTIMENT_COLUMNS)
                continue
            scores = group.vader_compound.tolist()
            p = int(group.sentiment_class.eq("positive").sum())
            u = int(group.sentiment_class.eq("neutral").sum())
            neg = n - p - u
            for category, num in [("positive", p), ("neutral", u), ("negative", neg)]:
                assert row[category + "_count"] == num
                close(row[category + "_share"], num / n)
            b = (p - neg) / n
            wsi = 50 * (1 + b)
            se = 50 * math.sqrt(max(0, (p / n + neg / n - b * b) / n))
            close(row["mean_compound"], statistics.mean(scores))
            close(row["median_compound"], statistics.median(scores))
            close(row["std_compound"], statistics.stdev(scores) if n > 1 else float("nan"))
            close(row["WSI_count"], wsi)
            close(row["WSI_continuous"], 50 * (1 + statistics.mean(scores)))
            close(row["se_WSI_count"], se)
            close(row["WSI_count_ci95_low"], wsi - 1.96 * se)
            close(row["WSI_count_ci95_high"], wsi + 1.96 * se)
        assert weeks.article_count.sum() == len(subset)
        empty[strategy] = int(weeks.loc[weeks.is_complete_week].article_count.eq(0).sum())
        wsi = pd.read_csv(FOLDER / f"weekly_{label}_wsi.csv")
        for column in ["article_count", "WSI_count", "WSI_continuous", "se_WSI_count", "WSI_count_ci95_low", "WSI_count_ci95_high"]:
            assert np.allclose(weeks[column], wsi[column], equal_nan=True)
        all_weeks.append(weeks)
    combined = pd.concat(all_weeks, ignore_index=True)
    weekly_stats = pd.read_csv(FOLDER / "weekly_descriptive_statistics.csv")
    comparison = pd.read_csv(FOLDER / "broad_strict_wsi_comparison.csv")
    for row in weekly_stats.itertuples():
        subset = combined.loc[combined.query_strategy.eq(row.query_strategy)]
        if row.scope == "complete_weeks":
            subset = subset.loc[subset.is_complete_week]
        values = subset[row.metric].dropna().tolist()
        assert row.valid_weeks == len(values) and row.calendar_weeks == len(subset)
        close(row.mean, statistics.mean(values)); close(row.median, statistics.median(values)); close(row.std, statistics.stdev(values))
    for row in comparison.itertuples():
        subset = combined.loc[combined.is_complete_week] if row.scope == "complete_weeks" else combined
        paired = subset.pivot(index="week_start", columns="query_strategy", values=row.metric).dropna()
        assert len(paired) == row.paired_valid_weeks
        left, right = FLAGS
        close(np.corrcoef(paired[left], paired[right])[0, 1], row.broad_strict_pearson_correlation)
    sample = pd.read_csv(FOLDER / "vader_inspection_sample.csv")
    assert sample.guardian_id.is_unique and set(sample.guardian_id).issubset(seen)
    assert sample.manual_error_assessment.isna().all()
    for row in sample.itertuples():
        expected = analyzer.polarity_scores(row.text_sentiment)
        assert row.compound == expected["compound"] and row.sentiment_class == classify(row.compound)
    # PNGs must exist and be real, nonempty images, not just paths in a manifest.
    validation = json.loads((FOLDER / "validation_summary.json").read_text(encoding="utf-8"))
    from PIL import Image
    for name in validation["figures"]:
        with Image.open(ROOT / "figures/guardian_sentiment" / name) as image:
            assert image.width > 1000 and image.height > 500
            image.verify()
    for path, digest in provenance["protected_input_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    result = {"status": "passed", "article_records_verified": len(seen),
        "all_original_columns_and_types_unchanged": True, "VADER_recomputed_for_every_article": True,
        "threshold_boundary_and_empty_singleton_week_tests": "passed", "NY_week_assignment_verified": True,
        "weekly_counts_shares_compound_WSI_and_intervals_independently_verified": True,
        "article_moments_and_weekly_descriptives_verified": True, "strategy_correlations_verified": True,
        "empty_complete_weeks": empty, "inspection_sample_articles_verified": len(sample),
        "frozen_policy_and_input_hashes_unchanged": True, "FGI_or_market_data_used": False}
    (FOLDER / "independent_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
