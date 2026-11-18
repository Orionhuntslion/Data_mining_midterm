"""02. GDELT 영문 금융 헤드라인 수집 / 로컬 CSV 가져오기 / CNN FGI."""
from __future__ import annotations

import io
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from midterm_common import (
    REGIMES, PublicClient, digest, read_csv, require_columns, save_csv,
    save_json, utc_now,
)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
ARCHIVE_URL = "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed.csv"
OFFICIAL_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"


def collect_news(cfg):
    cfg.setup()
    if cfg.text_input:
        path = Path(cfg.text_input).expanduser().resolve()
        frame = pd.read_csv(path, keep_default_na=False)
        require_columns(frame, ["published_at", "raw_text", "source", "url"], "뉴스 입력 CSV")
        frame["text"] = frame["raw_text"]
        frame["retrieved_at"] = utc_now()
        frame["collection_method"] = "local_import"
        save_csv(frame, cfg.raw / "raw_text_data.csv")
        save_json({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "rows": len(frame)}, cfg.results / "news_import_manifest.json")
        return frame

    if (pd.Timestamp(cfg.end)-pd.Timestamp(cfg.start)).days>7 and not cfg.allow_full_collection:
        raise ValueError("현재는 출처 파일럿 검토 단계입니다. 전체 뉴스 수집 전에 --stage pilot 결과를 검토하세요.")
    # GDELT seendate는 발행 시각이 아니라 최초 탐지 시각이다.
    # 같은 UTC 일자의 수집을 뉴욕 일자로 변환하므로 끝 다음 UTC 날짜까지 요청한다.
    end_utc = (pd.Timestamp(cfg.end).tz_localize(cfg.timezone) + pd.DateOffset(days=1)).tz_convert("UTC")
    if end_utc > pd.Timestamp.now(tz="UTC"):
        raise ValueError("현지 날짜가 완전히 끝난 기간만 수집하세요.")
    first_utc = pd.Timestamp(cfg.start).tz_localize(cfg.timezone).tz_convert("UTC").normalize()
    last_utc = (end_utc - pd.Timedelta(seconds=1)).normalize()
    # DOC API의 과거 접근 기간은 제한적이다. 오래된 연구 기간은 로컬 CSV를 사용한다.
    if first_utc < pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=90):
        raise ValueError("GDELT DOC의 최근 약 3개월 접근 범위를 벗어났습니다. text_input을 지정하세요.")
    client = PublicClient(cfg.request_interval)
    frames, manifest = [], []
    cache_folder = cfg.raw / "gdelt"
    cache_folder.mkdir(exist_ok=True)
    for day in pd.date_range(first_utc, last_utc):
        params = {
            "query": cfg.query, "mode": "artlist", "format": "json",
            "maxrecords": cfg.max_records, "sort": "datedesc",
            "startdatetime": day.strftime("%Y%m%d000000"),
            "enddatetime": day.strftime("%Y%m%d235959"),
        }
        key = digest(json.dumps(params, sort_keys=True))[:16]
        cache = cache_folder / f"{day:%Y-%m-%d}_{key}.json"
        if cache.exists() and not cfg.refresh:
            envelope = json.loads(cache.read_text(encoding="utf-8"))
        else:
            response = client.get(GDELT_URL, params=params)
            try:
                payload = response.json()
            except ValueError as exc:
                raise ValueError(f"GDELT 비 JSON 응답: {response.text[:300]}") from exc
            if not isinstance(payload, dict) or "articles" not in payload:
                raise ValueError(f"GDELT 스키마 변경/오류 응답: {str(payload)[:300]}")
            envelope = {"url": response.url, "retrieved_at": utc_now(), "params": params, "payload": payload}
            save_json(envelope, cache)
        articles = envelope["payload"]["articles"]
        if not isinstance(articles, list):
            raise ValueError(f"articles가 리스트가 아닙니다: {cache}")
        manifest.append({"date_utc": day.date().isoformat(), "n_returned": len(articles),
                         "cap_hit": len(articles) >= cfg.max_records, "cache": str(cache),
                         "request_url": envelope["url"], "retrieved_at": envelope["retrieved_at"]})
        # 중간 중단 시에도 완료된 날짜의 수집 진단은 남긴다.
        save_csv(pd.DataFrame(manifest), cfg.results / "collection_manifest.csv")
        for item in articles:
            require = {"title", "url", "seendate", "domain"}
            if not require.issubset(item):
                raise ValueError(f"GDELT 필수 필드 누락: {item}")
            frames.append({
                "published_at": None, "observed_at": item["seendate"],
                "timestamp_kind": "gdelt_first_seen_not_publication",
                "text": item["title"], "raw_text": item["title"],
                "source": item["domain"], "url": item["url"],
                "language": item.get("language", ""), "collection_method": "gdelt_doc_headline",
                "request_url": envelope["url"], "retrieved_at": envelope["retrieved_at"],
            })
        print(f"GDELT {day:%Y-%m-%d} UTC: {len(articles)}건, cap_hit={manifest[-1]['cap_hit']}", flush=True)
    frame = pd.DataFrame(frames)
    if frame.empty:
        raise ValueError("뉴스 수집 결과 0건. 검색 조건/접근 기간을 확인하세요.")
    save_csv(frame, cfg.raw / "raw_text_data.csv")
    return frame


def collect_fgi(cfg):
    cfg.setup()
    if cfg.fgi_input:
        file = Path(cfg.fgi_input).expanduser().resolve()
        raw_bytes = file.read_bytes()
        origin = cfg.fgi_source_url
        if not origin:
            raise ValueError("FGI 로컬 파일은 fgi_source_url에 실제 출처 URL을 명시하세요.")
        source_type = "user_import"
        retrieved_at = utc_now()
    else:
        client = PublicClient(cfg.request_interval)
        if cfg.fgi_source == "local":
            raise ValueError("fgi_source='local'이면 fgi_input을 지정하세요.")
        origin = ARCHIVE_URL if cfg.fgi_source == "archive" else OFFICIAL_URL + cfg.start
        source_type = "third_party_cnn_archive" if cfg.fgi_source == "archive" else "cnn_undocumented_endpoint"
        cache = cfg.raw / ("fgi_archive.csv" if cfg.fgi_source == "archive" else f"fgi_official_{cfg.start}.json")
        if cache.exists() and not cfg.refresh:
            raw_bytes = cache.read_bytes()
        else:
            raw_bytes = client.get(origin).content
            cache.write_bytes(raw_bytes)
        retrieved_at = datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc).isoformat()

    if source_type == "cnn_undocumented_endpoint":
        payload = json.loads(raw_bytes)
        try:
            data = pd.DataFrame(payload["fear_and_greed_historical"]["data"])
            frame = pd.DataFrame({"date": pd.to_datetime(data["x"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize(),
                                  "fgi": data["y"], "category_source": data["rating"]})
        except (KeyError, TypeError) as exc:
            raise ValueError("CNN의 비공식 JSON 응답 스키마가 예상과 다릅니다.") from exc
    else:
        frame = pd.read_csv(io.BytesIO(raw_bytes))
        frame = frame.rename(columns={"Date": "date", "Fear Greed": "fgi", "Rating": "category_source", "category": "category_source"})
        require_columns(frame, ["date", "fgi"], "FGI")
        if "category_source" not in frame:
            frame["category_source"] = pd.NA
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["fgi"] = pd.to_numeric(frame["fgi"], errors="raise")
    if frame["date"].isna().any() or frame["fgi"].isna().any() or not frame["fgi"].between(0, 100).all():
        raise ValueError("FGI 날짜/점수가 누락되었거나 0~100 범위를 벗어났습니다.")
    if frame.duplicated("date").any():
        raise ValueError("FGI에 중복 날짜가 있습니다. 출처의 버전을 확인하세요.")
    frame = frame.loc[frame["date"].between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))].copy()
    if frame.empty:
        raise ValueError("FGI 자료가 설정 기간과 겹치지 않습니다.")
    # 원 출처 범주를 우선 사용. 없을 때만 연구자가 정의한 연속 점수 구간을 사용한다.
    mapping = {v.lower(): v for v in REGIMES}
    label = frame["category_source"].astype("string").str.strip().str.lower().map(mapping)
    provided = frame["category_source"].notna() & frame["category_source"].astype(str).str.strip().ne("")
    if (provided & label.isna()).any():
        raise ValueError("알 수 없는 FGI 범주가 있습니다. 자동 변환하지 않습니다.")
    fallback = pd.cut(frame["fgi"], [-1, 25, 45, 55, 75, 100], labels=REGIMES, right=True).astype("string")
    frame["regime"] = label.astype("string").fillna(fallback)
    frame["regime_basis"] = "source_label"
    frame.loc[label.isna(), "regime_basis"] = "researcher_bins_le25_le45_le55_le75_le100"
    frame["source_url"] = origin
    frame["source_type"] = source_type
    frame = frame.sort_values("date")
    save_csv(frame, cfg.processed / "fear_greed_data.csv")
    save_json({"url": origin, "source_type": source_type, "retrieved_or_imported_at": retrieved_at,
               "sha256": hashlib.sha256(raw_bytes).hexdigest(), "rows": len(frame),
               "range": [str(frame.date.min()), str(frame.date.max())]}, cfg.results / "fgi_manifest.json")
    print(f"FGI: {len(frame)}일, {frame.date.min().date()} ~ {frame.date.max().date()} ({source_type})")
    return frame


def run(cfg):
    # 뉴스 제한으로 중단되더라도 확보 가능한 FGI는 먼저 저장한다.
    collect_fgi(cfg)
    return collect_news(cfg)
