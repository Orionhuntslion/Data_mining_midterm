"""One cached public FRED CSV download; no price filling or access-control retries."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".packages"))
import requests

SOURCE_PAGE = "https://fred.stlouisfed.org/series/SP500"
DOWNLOAD_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
RAW = ROOT / "data/raw/market/fred_sp500_2017-01-01_2026-10-25.csv"
METADATA = RAW.with_suffix(".metadata.json")


def obtain_market_csv():
    if RAW.exists():
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        assert hashlib.sha256(RAW.read_bytes()).hexdigest() == metadata["sha256"]
        assert metadata["request_url"] == DOWNLOAD_URL
        return RAW, metadata
    RAW.parent.mkdir(parents=True, exist_ok=True)
    candidate = RAW.parent / "fred_simple_probe.response"
    if candidate.exists() and candidate.read_bytes().startswith(b"observation_date,SP500"):
        data = candidate.read_bytes()
        retrieved_at = datetime.fromtimestamp(candidate.stat().st_mtime, timezone.utc).isoformat()
        final_url, status, content_type = DOWNLOAD_URL, 200, "text/csv (validated header)"
    else:
        response = requests.get(DOWNLOAD_URL, timeout=(10, 30),
            headers={"User-Agent": "MidtermMarketStudy/1.0 (academic research; Python requests)"}, allow_redirects=True)
        data = response.content
        retrieved_at = datetime.now(timezone.utc).isoformat()
        final_url, status, content_type = response.url, response.status_code, response.headers.get("Content-Type")
    metadata = {"source": "FRED SP500", "original_provider": "S&P Dow Jones Indices LLC", "source_page": SOURCE_PAGE,
        "request_url": DOWNLOAD_URL, "final_url": final_url, "http_status": status,
        "retrieved_at_utc": retrieved_at, "content_type": content_type,
        "successful_data_request_count": 1, "automatic_retry": False, "authentication_or_access_bypass": False,
        "price_basis": "daily close of S&P 500 price index; excludes dividends; not total return",
        "adjusted_close_available": False, "analysis_period": ["2017-01-01", "2026-10-25"],
        "raw_download_scope": "FRED rolling daily history; analysis dates filtered locally"}
    if status != 200:
        METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"FRED HTTP {status}; stopped without retry or bypass")
    first_line = data.decode("utf-8-sig").splitlines()[0]
    if "SP500" not in first_line or "," not in first_line or "<" in first_line:
        raise RuntimeError("FRED did not return the expected CSV; no HTML fallback")
    metadata["sha256"] = hashlib.sha256(data).hexdigest()
    metadata["bytes"] = len(data)
    temporary = RAW.with_suffix(".csv.tmp")
    temporary.write_bytes(data)
    temporary.replace(RAW)
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return RAW, metadata


if __name__ == "__main__":
    path, metadata = obtain_market_csv()
    print(json.dumps({"saved": str(path.relative_to(ROOT)), **metadata}, ensure_ascii=False, indent=2))
