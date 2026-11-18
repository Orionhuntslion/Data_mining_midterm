"""Q2: Frequency/TF-IDF logistic regression, with an ordered 70/30 split."""
from pathlib import Path
import re
import sys
import warnings

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (BASE / "IMDB Dataset.csv").exists():
    BASE = BASE / "midterm"
sys.path.insert(0, str(BASE / ".packages"))

import nltk
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.data.path.insert(0, str(BASE / "nltk_data"))


def preprocess(text, stop_words):
    # Input is review text, even when it happens to resemble a file path.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        text = BeautifulSoup(text, "html5lib").get_text()  # Step 1
    text = re.sub(r"[^a-zA-Z]", " ", text)             # Step 2
    tokens = text.lower().split()                       # Step 3
    tokens = [word for word in tokens if word not in stop_words]  # Step 4
    return " ".join(tokens)                             # Step 5


def main():
    reviews = pd.read_csv(BASE / "IMDB Dataset.csv", nrows=3000)
    assert len(reviews) == 3000 and reviews.notna().all().all()
    assert set(reviews["sentiment"]) == {"positive", "negative"}
    stop_words = set(stopwords.words("english"))
    clean = reviews["review"].map(lambda text: preprocess(text, stop_words))
    labels = reviews["sentiment"].map({"positive": 1, "negative": 0})
    x_train, x_test = clean.iloc[:2100], clean.iloc[2100:]
    y_train, y_test = labels.iloc[:2100], labels.iloc[2100:]
    assert len(x_train) == 2100 and len(x_test) == 900
    print("Q2: train = reviews 1-2100; test = reviews 2101-3000 (no shuffle)")
    print(f"NLTK English stopwords: {len(stop_words)}")
    print("Preprocessing complete: HTML, non-letters, and stopwords removed.")
    summary = []
    predictions = pd.DataFrame({"review_number": range(2101, 3001), "actual": y_test.to_numpy()})
    for name, vectorizer in [("CountVectorizer", CountVectorizer()),
                             ("TfidfVectorizer", TfidfVectorizer())]:
        # Fit the vocabulary and IDF only on the training reviews.
        vec = vectorizer.fit(x_train)
        train_matrix = vec.transform(x_train)
        test_matrix = vec.transform(x_test)
        model = LogisticRegression()  # Lecture slide 34: default settings.
        model.fit(train_matrix, y_train)
        predicted = model.predict(test_matrix)
        accuracy = accuracy_score(y_test, predicted)
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        assert matrix.sum() == 900 and matrix.trace() == (predicted == y_test).sum()
        print(f"\n{name} + LogisticRegression")
        print(f"Train shape: {train_matrix.shape}; test shape: {test_matrix.shape}")
        print(f"Iterations: {model.n_iter_.tolist()}; max_iter: {model.max_iter}")
        print(f"Test accuracy: {accuracy:.10f} ({accuracy:.4%}); correct: {matrix.trace()}/900")
        print("Confusion matrix: rows = actual, columns = predicted; order = [0, 1]")
        print(matrix)
        print(classification_report(y_test, predicted, labels=[0, 1],
                                    target_names=["negative (0)", "positive (1)"],
                                    digits=6, zero_division=0))
        summary.append({"method": name, "test_accuracy": accuracy,
                        "correct": int(matrix.trace()), "test_size": 900,
                        "features": train_matrix.shape[1], "iterations": int(model.n_iter_[0])})
        predictions[name] = predicted
    output = BASE / "results"
    output.mkdir(exist_ok=True)
    predictions.to_csv(output / "q2_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "q2_summary.csv", index=False)
    return pd.DataFrame(summary)


if __name__ == "__main__":
    q2_summary = main()
