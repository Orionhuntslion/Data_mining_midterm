"""03. 원문 보존, 최소 감성 전처리, 별도 빈도 토큰, 중복/시간 진단."""
from __future__ import annotations

import html
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from midterm_common import digest, read_csv, require_columns, save_csv, save_json

# 금융 은어/욕설 목록으로 삭제하지 않는다. 부정어도 살린다.
STOPWORDS = set(ENGLISH_STOP_WORDS) - {"no", "not", "never", "nor", "nothing", "without"}
TOKEN_RE = re.compile(r"\$?[a-zA-Z]+(?:'[a-zA-Z]+)?")


def sentiment_text(raw):
    text = str(raw)
    if re.search(r"</?[A-Za-z][^>]*>", text):
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def tokens_for_frequency(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t.lstrip("$")) > 1]


def canonical_url(url):
    parts = urlsplit(str(url).strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"유효한 기사 URL이 아닙니다: {url!r}")
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(sorted(query)), ""))


def parse_timestamp(value):
    if pd.isna(value) or not str(value).strip():
        return pd.NaT
    value = str(value).strip()
    if re.fullmatch(r"\d{8}T\d{6}Z", value):
        return pd.to_datetime(value, format="%Y%m%dT%H%M%SZ", utc=True)
    # 로컬 입력은 시간대/UTC 오프셋 필수. 날짜만 있으면 임의 시각을 만들지 않는다.
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"시간대 없는 시각: {value!r}. ISO 8601 UTC/offset을 명시하세요.")
    return stamp.tz_convert("UTC")


def run(cfg):
    raw = read_csv(cfg.raw / "raw_text_data.csv")
    require_columns(raw, ["raw_text", "source", "url", "published_at"], "원문")
    for col in ["raw_text", "source", "url"]:
        if raw[col].isna().any() or raw[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"원문 필수 열 {col}에 빈 값이 있습니다.")
    raw_n = len(raw)
    frame = raw.copy()
    publication = pd.to_datetime(frame.published_at.map(parse_timestamp), utc=True)
    observed = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    if "observed_at" in frame:
        observed = pd.to_datetime(frame.observed_at.map(parse_timestamp), utc=True)
    frame["analysis_timestamp"] = publication.fillna(observed)
    frame["timestamp_kind"] = "publication"
    frame.loc[publication.isna(), "timestamp_kind"] = "gdelt_first_seen_not_publication"
    if frame.analysis_timestamp.isna().any():
        raise ValueError("발행/탐지 시각이 모두 없는 뉴스가 있습니다.")
    frame["date"] = frame.analysis_timestamp.dt.tz_convert(cfg.timezone).dt.tz_localize(None).dt.normalize()
    in_range = frame.date.between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))
    excluded = int((~in_range).sum())
    save_csv(frame.loc[~in_range], cfg.results / "out_of_range_texts.csv")
    frame = frame.loc[in_range].copy()
    if frame.empty:
        raise ValueError("분석 기간에 속하는 뉴스가 없습니다.")
    frame["sentiment_text"] = frame.raw_text.map(sentiment_text)
    if frame.sentiment_text.str.strip().eq("").any():
        raise ValueError("HTML 제거 후 비어 있는 텍스트가 있습니다.")
    frame["canonical_url"] = frame.url.map(canonical_url)
    frame["text_key"] = frame.sentiment_text.str.lower().str.replace(r"\s+", " ", regex=True)
    frame = frame.sort_values(["analysis_timestamp", "canonical_url"], kind="stable")
    # 동일 URL은 기간 전체에서 1개. 동일 헤드라인은 같은 날에 한 번만 센다.
    by_url = frame.duplicated("canonical_url", keep="first")
    save_csv(frame.loc[by_url], cfg.results / "duplicate_urls.csv")
    frame = frame.loc[~by_url].copy()
    by_text = frame.duplicated(["date", "text_key"], keep="first")
    save_csv(frame.loc[by_text], cfg.results / "duplicate_headlines.csv")
    frame = frame.loc[~by_text].copy()
    frame["doc_id"] = frame.canonical_url.map(digest)
    frame["frequency_text"] = frame.sentiment_text.map(lambda x: " ".join(tokens_for_frequency(x)))
    frame["cleaned_text"] = frame.frequency_text
    frame = frame.drop(columns="text_key").reset_index(drop=True)
    assert frame.doc_id.is_unique
    assert not frame.duplicated(["date", "sentiment_text"]).any()
    save_csv(frame, cfg.processed / "processed_text_data.csv")
    counts = frame.groupby("date").size().reindex(cfg.calendar, fill_value=0).rename("n_docs").reset_index()
    save_csv(counts, cfg.results / "daily_document_count.csv")
    save_csv(frame.groupby(["date", "source"]).size().rename("n_docs").reset_index(), cfg.results / "daily_source_counts.csv")
    diagnostic = {"raw_observations": raw_n, "out_of_range": excluded,
                  "duplicate_urls": int(by_url.sum()), "duplicate_headlines": int(by_text.sum()),
                  "after_deduplication": len(frame), "date_min": str(frame.date.min()), "date_max": str(frame.date.max()),
                  "zero_document_days": int(counts.n_docs.eq(0).sum()),
                  "timestamp_kinds": frame.timestamp_kind.value_counts().to_dict(),
                  "missing_values": frame.isna().sum().to_dict()}
    save_json(diagnostic, cfg.results / "preprocessing_diagnostics.json")
    print(pd.Series(diagnostic).to_string())
    return frame
