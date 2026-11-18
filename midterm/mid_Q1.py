"""Q1: Dictionary-based sentiment analysis of the first 3,000 IMDB reviews."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (BASE / "IMDB Dataset.csv").exists():
    BASE = BASE / "midterm"
sys.path.insert(0, str(BASE / ".packages"))

import nltk
import pandas as pd
from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob

nltk.data.path.insert(0, str(BASE / "nltk_data"))


def evaluate(name, actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    assert matrix.sum() == len(actual)
    assert matrix.trace() == (actual == predicted).sum()
    print(f"\n{name} accuracy: {accuracy:.10f} ({accuracy:.4%})")
    print("Confusion matrix: rows = actual, columns = predicted; order = [0, 1]")
    print(matrix)
    print(classification_report(actual, predicted, labels=[0, 1],
                                target_names=["negative (0)", "positive (1)"],
                                digits=6, zero_division=0))
    return accuracy


def main():
    reviews = pd.read_csv(BASE / "IMDB Dataset.csv", nrows=3000)
    assert len(reviews) == 3000 and reviews.notna().all().all()
    assert set(reviews["sentiment"]) == {"positive", "negative"}
    actual = reviews["sentiment"].map({"positive": 1, "negative": 0})
    print("Q1: First 3,000 reviews; positive = 1, negative = 0")
    print(f"(1)(a) Positive reviews: {actual.sum()}")
    print(f"(1)(b) Negative reviews: {(actual == 0).sum()}; percentage: {(actual == 0).mean():.4%}")

    # Q2's preprocessing is not applied to Q1. Preserve the original reviews.
    afinn = Afinn()
    vader = SentimentIntensityAnalyzer()
    scores = pd.DataFrame({"review_number": range(1, 3001), "actual": actual})
    scores["afinn"] = reviews["review"].map(afinn.score)
    vader_scores = pd.DataFrame(reviews["review"].map(vader.polarity_scores).tolist())
    scores["vader_compound"] = vader_scores["compound"]
    scores["vader_pos"] = vader_scores["pos"]
    scores["vader_neg"] = vader_scores["neg"]
    scores["textblob_polarity"] = reviews["review"].map(lambda text: TextBlob(text).sentiment.polarity)

    scores["afinn_prediction"] = (scores["afinn"] > 0).astype(int)
    # The exam requires pos > neg, NOT a compound-score threshold.
    scores["vader_prediction"] = (scores["vader_pos"] > scores["vader_neg"]).astype(int)
    scores["textblob_prediction"] = (scores["textblob_polarity"] > 0).astype(int)

    summary = []
    for number, name, score_column, prediction_column in [
        (2, "AFINN", "afinn", "afinn_prediction"),
        (3, "VADER", "vader_compound", "vader_prediction"),
        (4, "TextBlob", "textblob_polarity", "textblob_prediction"),
    ]:
        average = scores[score_column].mean()
        print(f"\n({number})(a) {name} mean score: {average:.10f}")
        accuracy = evaluate(f"({number})(b) {name}", actual, scores[prediction_column])
        summary.append({"method": name, "mean_score": average, "accuracy": accuracy})
    output = BASE / "results"
    output.mkdir(exist_ok=True)
    scores.to_csv(output / "q1_scores.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "q1_summary.csv", index=False)
    return pd.DataFrame(summary)


if __name__ == "__main__":
    q1_summary = main()
