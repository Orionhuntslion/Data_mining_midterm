"""Guardian 공개 설정과 비밀값 로더. 전체 수집은 명시적으로 선택한 두 전략만 허용."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TARGET_START="2017-01-01"
TARGET_END="2026-10-25"
PILOT_WINDOWS={2017:("2017-01-09","2017-01-15"),2020:("2020-03-16","2020-03-22"),2026:("2026-10-19","2026-10-25")}
ANCHORS='"Wall Street" OR "S&P 500" OR Nasdaq OR "Dow Jones" OR "US stocks" OR "US markets"'
MARKET='stocks OR equities OR "Wall Street" OR "S&P 500" OR Nasdaq OR "Dow Jones"'
GEOGRAPHY='US OR "U.S." OR American OR "Wall Street" OR "Federal Reserve"'
STRATEGIES={
    "anchors_global":{"q":ANCHORS},
    "anchors_business":{"q":ANCHORS,"section":"business"},
    "geography_and_equity":{"q":f"({GEOGRAPHY}) AND ({MARKET})"},
    "broad_business_control":{"q":MARKET,"section":"business"},
    "anchors_stock_market_tag":{"q":ANCHORS,"tag":"business/stock-markets"},
    "anchors_us_office":{"q":ANCHORS,"production-office":"us"},
}
STRATEGY_NOTES={
    "anchors_global":"정확한 미국시장 명칭 중심. 섹션 제한 없이 기술/경제 기사 포함. Wall Street의 비시장 용례는 수동 검토.",
    "anchors_business":"동일 검색어 + Business. 정밀도 개선과 Technology 등 관련 기사 누락을 비교.",
    "geography_and_equity":"미국 지리어 AND 주식시장 용어. 일반 미국주식 표현을 포괄하되 본문에 우연히 함께 나오는 용례 점검.",
    "broad_business_control":"stocks/equities를 포함한 넓은 Business 대조군. 비미국 주식시장 혼입을 점검.",
    "anchors_stock_market_tag":"같은 명칭 검색 + stock-markets 태그. 실제 태그 ID 확인 후만 요청; 옛 기사 태깅 누락 가능.",
    "anchors_us_office":"같은 검색 + 미국 제작 오피스. US 에디션/미국시장 관련성과 동일하지 않음.",
}
REVIEW_COLUMNS=["year","guardian_id","published_at","published_at_utc","published_at_et","date_et",
                "title","description","section","tags","url","query_strategy","relevant_us_equity_market","review_notes"]


def load_key(root=ROOT):
    """GUARDIAN_API_KEY만 읽는다. .env를 실행하지 않고 환경변수가 우선한다."""
    key=os.environ.get("GUARDIAN_API_KEY","").strip()
    if key:
        return key,"environment"
    path=Path(root)/".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line=line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line=line[7:].strip()
            name,separator,value=line.partition("=")
            if separator and name.strip()=="GUARDIAN_API_KEY":
                value=value.strip()
                if len(value)>=2 and value[0]==value[-1] and value[0] in {"'",'"'}:
                    value=value[1:-1]
                if value and value.lower() not in {"your_key_here","your_api_key","replace_me"}:
                    return value,"project_dotenv"
    return None,"missing"


@dataclass
class PilotSettings:
    root: Path=ROOT
    page_size: int=200
    max_pages_per_window: int=3
    max_requests_per_run: int=90
    interval_seconds: float=1.1
    daily_request_limit: int=500
    seed: int=42
    manual_per_year: int=50

    def __post_init__(self):
        self.root=Path(self.root).resolve()
        if not 1<=self.page_size<=200 or not 1<=self.max_pages_per_window<=5:
            raise ValueError("page_size 1~200, 파일럿 max_pages 1~5만 허용합니다.")
        if self.interval_seconds<1 or not 1<=self.max_requests_per_run<=150:
            raise ValueError("초당 1회 이하, 실행당 최대 150요청을 지켜야 합니다.")
        if not 1<=self.daily_request_limit<=500 or not 1<=self.manual_per_year<=50:
            raise ValueError("일 최대 500요청, 연도별 검토 표본 최대 50개입니다.")

    @property
    def results(self):
        return self.root/"results"/"guardian_api_pilot"

    @property
    def raw(self):
        return self.root/"data"/"raw"/"news"/"guardian_api_pilot"

    @property
    def processed(self):
        return self.root/"data"/"processed"/"guardian_api_pilot"


def window_params(start,end):
    """UTC 날짜 검색에 끝 다음 날을 포함한 뒤, 실제 NY 날짜로 엄격히 자른다."""
    return {"from-date":start,"to-date":(date.fromisoformat(end)+timedelta(days=1)).isoformat()}


def validate_request(params,kind):
    """파일럿 제한을 유지하고, 별도 historical 목적에 승인된 전체 수집만 허용한다."""
    if kind=="historical":
        selected=[STRATEGIES[name] for name in ["broad_business_control","anchors_stock_market_tag"]]
        actual={k:params[k] for k in ["q","section","tag","production-office"] if k in params}
        dates=window_params(TARGET_START,TARGET_END)
        if actual not in selected or any(params.get(k)!=v for k,v in dates.items()):
            raise ValueError("전체 수집은 고정 목표 기간과 승인된 두 전략만 허용합니다.")
        if not 1<=params.get("page-size",0)<=200 or params.get("page",0)<1 or params.get("order-by")!="oldest":
            raise ValueError("전체 수집의 페이지 크기/번호/정렬 오류")
        return
    if kind=="catalog":
        return
    if kind=="count_only":
        expected=window_params(TARGET_START,TARGET_END)
        if any(params.get(k)!=v for k,v in expected.items()) or params.get("page-size")!=1 or params.get("page")!=1:
            raise ValueError("전체 기간에는 page-size=1, page=1 메타데이터 조회만 허용합니다.")
        return
    if kind!="pilot":
        raise ValueError("알 수 없는 요청 목적")
    allowed=[window_params(*window) for window in PILOT_WINDOWS.values()]
    if not any(all(params.get(k)==v for k,v in pair.items()) for pair in allowed):
        raise ValueError("본문/필드 수집은 사전 지정된 세 주간 파일럿만 허용합니다.")
