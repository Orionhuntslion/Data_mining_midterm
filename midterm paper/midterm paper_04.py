"""04. 수정하지 않은 VADER로 문서별 연속 점수/범주를 저장한다."""
import importlib.metadata

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from midterm_common import read_csv, save_csv, save_json


def run(cfg):
    frame = read_csv(cfg.processed / "processed_text_data.csv", ["date"])
    analyzer = SentimentIntensityAnalyzer()
    scores = pd.DataFrame(frame.sentiment_text.map(analyzer.polarity_scores).tolist(), index=frame.index)
    scores = scores.rename(columns={"neg": "negative", "neu": "neutral", "pos": "positive"})
    assert scores.compound.between(-1, 1).all()
    assert np.allclose(scores[["negative", "neutral", "positive"]].sum(axis=1), 1, atol=0.002)
    frame = pd.concat([frame, scores], axis=1)
    # 수정 설계의 원래 VADER 필드명도 유지한다. 기존 그림용 긴 열 이름은 호환 보존.
    frame["neg"] = frame["negative"]
    frame["neu"] = frame["neutral"]
    frame["pos"] = frame["positive"]
    frame["sentiment_label"] = np.select([frame.compound.ge(0.05), frame.compound.le(-0.05)], ["positive", "negative"], default="neutral")
    save_csv(frame, cfg.processed / "scored_text_data.csv")
    summary = frame.compound.describe().to_dict()
    summary.update({"median": frame.compound.median(), "skewness": frame.compound.skew(),
                    "excess_kurtosis": frame.compound.kurt(),
                    "positive_share": frame.sentiment_label.eq("positive").mean(),
                    "negative_share": frame.sentiment_label.eq("negative").mean(),
                    "neutral_share": frame.sentiment_label.eq("neutral").mean()})
    save_csv(pd.DataFrame([summary]), cfg.results / "sentiment_descriptive_statistics.csv")
    save_json({"model": "unmodified vaderSentiment", "version": importlib.metadata.version("vaderSentiment"),
               "positive_threshold": 0.05, "negative_threshold": -0.05,
               "input": "sentiment_text; case, punctuation and negation preserved"}, cfg.results / "sentiment_model.json")
    print(pd.Series(summary).to_string())
    return frame
