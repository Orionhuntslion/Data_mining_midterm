"""전체 수집의 중단/재개·전략 병합·ET 경계·Parquet 검증. 합성 자료는 임시 폴더만 사용."""
import hashlib
import json
import runpy
import tempfile
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
full=runpy.run_path(str(ROOT/"guardian_full_collection.py"))
from guardian_api_config import validate_request
from guardian_api_client import GuardianClient,PilotStopped,write_json
import pandas as pd


def article(identifier,stamp,title="NOT fucking bad!!! 🚀",description="<p>Raw &amp; description.</p>"):
    return {"id":identifier,"webPublicationDate":stamp,"webTitle":title,"webUrl":"https://fixture.invalid/"+identifier,
        "sectionId":"business","sectionName":"Business","type":"article","tags":[{"id":"business/stock-markets","type":"keyword"}],
        "fields":{"headline":title,"trailText":description}}


class FakeClient:
    def __init__(self,interrupt=False):
        self.used=0
        self.interrupt=interrupt

    def fetch(self,endpoint,params,kind,path):
        validate_request(params,kind)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")),True
        strict="tag" in params
        if self.interrupt and strict and params["page"]==2:
            raise PilotStopped("synthetic interrupted page")
        self.used+=1
        a=article("a","2017-01-01T05:00:00Z")
        b=article("b","2020-03-16T14:30:00Z",description=None)
        c=article("c","2026-10-26T03:00:00Z")
        d=article("d","2021-05-04T14:30:00Z",title=None)
        e=article("e","2026-10-26T05:00:00Z")
        items=([b,d] if strict else [a,b]) if params["page"]==1 else ([e] if strict else [c])
        envelope={"request_sha256":hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest(),
            "retrieved_at":"2026-10-27T00:00:00Z","response":{"status":"ok","total":3,"pages":2,"pageSize":2,"currentPage":params["page"],"results":items}}
        write_json(envelope,path)
        return envelope,False


def main():
    import pyarrow.parquet as pq
    settings=full["FullSettings"]()
    for name in full["SELECTED"]:
        validate_request(full["request_params"](name,1,settings),"historical")
    from guardian_api_config import STRATEGIES
    invalid=full["request_params"]("broad_business_control",1,settings)
    invalid.update(STRATEGIES["geography_and_equity"])
    try:
        validate_request(invalid,"historical")
    except ValueError:
        pass
    else:
        raise AssertionError("Unapproved strategy accepted")
    with tempfile.TemporaryDirectory(prefix="guardian_full_verify_") as directory:
        settings=full["FullSettings"](root=Path(directory),page_size=2)
        try:
            full["collect"](settings,client=FakeClient(interrupt=True))
        except RuntimeError:
            assert json.loads((settings.results/"collection_status.json").read_text())["status"]=="stopped"
        else:
            raise AssertionError("Interruption was not recorded")
        resumed=FakeClient()
        summary=full["collect"](settings,client=resumed)
        assert resumed.used==1
        assert summary["final_unique_article_count"]==4 and summary["overlap_between_strategies"]==1
        assert summary["duplicate_count_removed_in_et_target"]==1
        assert summary["out_of_target_et_records_excluded"]==1
        assert summary["missing_title_count"]==1 and summary["missing_description_count"]==1
        dataset=pd.read_parquet(settings.raw/"articles")
        assert len(dataset)==4 and dataset.guardian_id.is_unique
        b=dataset.loc[dataset.guardian_id.eq("b")].iloc[0]
        assert b.in_broad_business_control and b.in_anchors_stock_market_tag
        a=dataset.loc[dataset.guardian_id.eq("a")].iloc[0]
        assert a.web_title=="NOT fucking bad!!! 🚀" and a.description=="<p>Raw &amp; description.</p>"
        assert dataset.published_at_utc.dt.tz is not None and str(dataset.published_at_et.dt.tz)=="America/New_York"
        assert not any(col in dataset for col in ["compound","sentiment_label","WSI_count","fgi","relevant_us_equity_market"])
        assert pd.read_csv(settings.results/"yearly_article_counts.csv").unique_articles.sum()==4
        assert len(pd.read_csv(settings.results/"monthly_article_counts.csv"))==118
        cached=FakeClient()
        with patch("guardian_full_diagnostics.finalize",return_value=summary):
            full["collect"](settings,client=cached)
        assert cached.used==0
        # 과거 파일럿 로그도 전체 수집의 일일 예산에 포함되는지 검증.
        shared=settings.root/"results"/"guardian_api_pilot"/"http_log.jsonl"
        shared.parent.mkdir(parents=True,exist_ok=True)
        shared.write_text(json.dumps({"attempted_at":datetime.now(timezone.utc).isoformat(),"endpoint":"https://content.guardianapis.com/search"})+"\n")
        quota=GuardianClient(full["FullSettings"](root=settings.root,daily_request_limit=1),"fixture-key")
        try:
            quota.check_budget()
        except PilotStopped:
            pass
        else:
            raise AssertionError("Shared daily quota was ignored")
    print("PASS: approved strategies only; interrupted resume fetched one page; union flags; raw text; UTC/ET boundaries; Parquet types/counts; no sentiment/labels; shared quota")


if __name__=="__main__":
    main()
