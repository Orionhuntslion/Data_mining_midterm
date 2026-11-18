"""01. 공통 설정, 파일 저장, 접근 규칙을 지키는 HTTP 클라이언트."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
REGIMES = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
INDEX_COLUMNS = ["WSI_continuous", "WSI_count", "WSI_binary"]
PRIMARY_INDEX = "WSI_count"


@dataclass
class Config:
    root: Path = ROOT
    start: str = "2017-01-01"
    end: str = "2026-10-25"
    # 검색어는 감정 단어와 무관하게 미리 고정한다.
    query: str = ('("stock market" OR "Wall Street" OR "S&P 500") '
                  'sourcelang:english sourcecountry:US')
    timezone: str = "America/New_York"
    min_daily_docs: int = 20
    max_lag: int = 7
    top_n: int = 30
    timing_terms_n: int = 15
    seed: int = 42
    request_interval: float = 6.0
    # 각 UTC 날짜의 최대 250건. 전수조사/무작위 표본으로 해석하지 않는다.
    max_records: int = 250
    text_input: str | None = None
    fgi_input: str | None = None
    fgi_source: str = "archive"  # archive | official | local
    fgi_source_url: str = ""
    refresh: bool = False
    allow_full_collection: bool = False

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        a, b = pd.Timestamp(self.start), pd.Timestamp(self.end)
        if a > b or a != a.normalize() or b != b.normalize():
            raise ValueError("start/end는 오름차순 YYYY-MM-DD여야 합니다.")
        if self.min_daily_docs < 1 or self.max_lag < 0:
            raise ValueError("min_daily_docs >= 1, max_lag >= 0이어야 합니다.")
        if not 1 <= self.max_records <= 250 or self.request_interval < 5:
            raise ValueError("GDELT: max_records는 1~250, 요청 간격은 5초 이상입니다.")
        if self.top_n < 1 or self.timing_terms_n < 1:
            raise ValueError("단어 선택 개수는 1 이상이어야 합니다.")
        if self.fgi_source not in {"archive", "official", "local"}:
            raise ValueError("fgi_source: archive / official / local")

    @property
    def raw(self):
        return self.root / "data" / "raw"

    @property
    def processed(self):
        return self.root / "data" / "processed"

    @property
    def results(self):
        return self.root / "results"

    @property
    def figures(self):
        return self.root / "figures"

    @property
    def calendar(self):
        return pd.date_range(self.start, self.end, name="date")

    def setup(self):
        for folder in [self.raw, self.processed, self.results, self.figures]:
            folder.mkdir(parents=True, exist_ok=True)
        np.random.seed(self.seed)
        save_json(asdict(self), self.results / "config.json")


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path, dates=()):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. 앞 단계를 먼저 실행하세요.")
    # "NA", "null" 같은 실제 문자열을 원문에서 임의 결측값으로 바꾸지 않는다.
    return pd.read_csv(path, parse_dates=list(dates), keep_default_na=False, na_values=[""])


def require_columns(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: 필수 열 누락 {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label}: 관측치가 없습니다.")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AccessError(RuntimeError):
    pass


class PublicClient:
    """브라우저 위장/차단 우회/자동 반복 재시도 없이 공개 URL만 요청한다."""

    def __init__(self, interval=6):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "MidtermSentimentResearch/1.0 (academic text study)"
        self.interval = interval
        self.last_request = 0.0
        self.robots = {}

    def _request(self, url, **kwargs):
        time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        response = self.session.get(url, timeout=60, **kwargs)
        if response.status_code in {401, 403, 418, 429}:
            raise AccessError(
                f"HTTP {response.status_code}: {response.url}\n"
                f"접근/요청 제한으로 중단합니다. Retry-After={response.headers.get('Retry-After')}. "
                "우회하지 말고 허용된 시점에 재개하거나 적법하게 확보한 CSV를 사용하세요."
            )
        return response

    def get(self, url, **kwargs):
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots:
            robot_url = origin + "/robots.txt"
            response = self._request(robot_url)
            parser = RobotFileParser(robot_url)
            if response.status_code == 404:
                parser.parse([])
            else:
                response.raise_for_status()
                if "<html" in response.text[:500].lower():
                    raise AccessError(f"robots.txt가 HTML을 반환하여 접근 규칙 확인 불가: {robot_url}")
                parser.parse(response.text.splitlines())
            self.robots[origin] = parser
        parser = self.robots[origin]
        if not parser.can_fetch(self.session.headers["User-Agent"], url):
            raise AccessError(f"robots.txt에서 허용하지 않는 URL: {url}")
        delay = parser.crawl_delay(self.session.headers["User-Agent"]) or parser.crawl_delay("*")
        if delay:
            self.interval = max(self.interval, float(delay))
        response = self._request(url, **kwargs)
        response.raise_for_status()
        return response
