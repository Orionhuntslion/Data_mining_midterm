"""2차 Guardian 파일럿 변경분 검증. API 요청과 합성 실증 결과 저장 없음."""
import json
import runpy
import tempfile
from pathlib import Path
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parent
api=runpy.run_path(str(ROOT/"guardian_api_pilot.py"))
import pandas as pd
import requests
import nbformat
from guardian_api_config import (PilotSettings,PILOT_WINDOWS,STRATEGIES,load_key,validate_request,window_params)
from guardian_api_client import GuardianClient,PilotStopped,write_json
from guardian_api_analysis import deduplicate,manual_sample,normalise,volume_estimates


def item(index,year=2017):
    start=PILOT_WINDOWS[year][0]
    return {"id":f"business/{year}/fixture-{index}","webTitle":f"US stocks NOT fucking bad!!! 🚀 {index}",
        "webPublicationDate":f"{start}T14:30:00Z","webUrl":f"https://fixture.invalid/{year}/{index}",
        "sectionId":"business","sectionName":"Business","type":"article",
        "fields":{"trailText":"Profits &amp; losses","productionOffice":"US"},
        "tags":[{"id":"business/stock-markets","type":"keyword","webTitle":"Stock markets"}]}


def response(payload,status=200):
    result=requests.Response()
    result.status_code=status
    result._content=json.dumps(payload).encode()
    result.url="https://content.guardianapis.com/search?api-key=fixture-secret"
    return result


def check_key_cache_and_stop(folder):
    settings=PilotSettings(root=folder)
    (folder/".env").write_text('GUARDIAN_API_KEY="fixture-secret"\n',encoding="utf-8")
    with patch.dict("os.environ",{"GUARDIAN_API_KEY":""}):
        assert load_key(folder)==("fixture-secret","project_dotenv")
    with patch.dict("os.environ",{"GUARDIAN_API_KEY":"environment-secret"}):
        assert load_key(folder)==("environment-secret","environment")
    client=GuardianClient(settings,"fixture-secret")
    params=api["search_params"]("anchors_global",*PILOT_WINDOWS[2017],200,1)
    path=settings.raw/"cache_test.json"
    payload={"response":{"status":"ok","total":1,"pages":1,"pageSize":200,"results":[item(1)]}}
    with patch.object(client,"get_official_api",return_value=response(payload)) as get:
        first,cached=client.fetch("/search",params,"pilot",path)
        second,cached=client.fetch("/search",params,"pilot",path)
        assert cached and get.call_count==1 and first==second
    assert "fixture-secret" not in path.read_text(encoding="utf-8")
    assert "api-key" not in first["params"]
    stopped=GuardianClient(settings,"fixture-secret")
    missing_robots=response({},404)
    denied=response({},429)
    with patch.object(stopped.public.session,"get",side_effect=[missing_robots,denied]) as get,patch("time.sleep"):
        try:
            stopped.fetch("/search",params,"pilot",settings.raw/"should_not_exist.json")
        except PilotStopped as exc:
            assert "fixture-secret" not in str(exc) and "429" in str(exc)
        else:
            raise AssertionError("429 did not stop")
        assert stopped.halted and get.call_count==2
    assert not (settings.raw/"should_not_exist.json").exists()
    assert "fixture-secret" not in stopped.log.read_text(encoding="utf-8")


def check_auxiliary_robots(folder):
    """robots 401과 실제 /search 401/403/429를 구분하는 회귀 검증."""
    params=api["search_params"]("anchors_global",*PILOT_WINDOWS[2017],200,1)
    payload={"response":{"status":"ok","total":1,"pages":1,"pageSize":200,"results":[item(1)]}}
    for robot_status in [401,403,404,429,500]:
        settings=PilotSettings(root=folder/f"robots_{robot_status}")
        client=GuardianClient(settings,"fixture-secret")
        with patch.object(client.public.session,"get",side_effect=[response({},robot_status),response(payload)]) as get,patch("time.sleep"):
            result,_=client.fetch("/search",params,"pilot",settings.raw/"page.json")
            assert len(result["response"]["results"])==1 and not client.halted and get.call_count==2
        events=[json.loads(x) for x in client.log.read_text().splitlines()]
        assert events[0]["request_role"]=="robots_auxiliary" and events[0]["status"]==robot_status
        assert events[1]["request_role"]=="content_api" and events[1]["status"]==200
        assert client.used==2
    for api_status in [401,403,429]:
        settings=PilotSettings(root=folder/f"api_{api_status}")
        client=GuardianClient(settings,"fixture-secret")
        with patch.object(client.public.session,"get",side_effect=[response({},401),response({},api_status)]) as get,patch("time.sleep"):
            try:
                client.fetch("/search",params,"pilot",settings.raw/"page.json")
            except PilotStopped:
                assert client.halted and get.call_count==2
            else:
                raise AssertionError("Content API denial did not stop")
    settings=PilotSettings(root=folder/"explicit_disallow")
    client=GuardianClient(settings,"fixture-secret")
    robots=response({},200)
    robots._content=b"User-agent: *\nDisallow: /search\n"
    with patch.object(client.public.session,"get",return_value=robots) as get:
        try:
            client.fetch("/search",params,"pilot",settings.raw/"page.json")
        except PilotStopped as exc:
            assert "명시적 금지" in str(exc) and get.call_count==1
        else:
            raise AssertionError("Explicit robots restriction was bypassed")


class FakeClient:
    def __init__(self):
        self.used=0
        self.halted=False

    def fetch(self,endpoint,params,kind,path):
        validate_request(params,kind)
        self.used+=1
        if kind=="catalog":
            results=[{"id":"business/stock-markets"}] if endpoint=="/tags" else [{"id":"business"}]
            payload={"status":"ok","results":results}
        elif kind=="count_only":
            payload={"status":"ok","results":[item(0)],"total":1200,"pages":1200,"pageSize":1}
        else:
            year=int(params["from-date"][:4])
            payload={"status":"ok","results":[item(0,year),item(1,year)],"total":2,"pages":1,"pageSize":200,"currentPage":1}
        envelope={"response":payload,"retrieved_at":"2026-10-26T00:00:00Z"}
        write_json(envelope,path)
        return envelope,False


def main():
    # 전체 기간의 페이지 순회를 차단하는지 확인한다.
    for kind,params in [("pilot",{"from-date":"2017-01-01","to-date":"2026-10-25"}),
                        ("count_only",{**window_params("2017-01-01","2026-10-25"),"page-size":200,"page":1}),
                        ("count_only",{**window_params("2017-01-01","2026-10-25"),"page-size":1,"page":2})]:
        try:
            validate_request(params,kind)
        except ValueError:
            pass
        else:
            raise AssertionError("full-collection guard failed")
    article=normalise(item(0),2017)
    assert article["published_at"].endswith("Z") and article["published_at_et"].endswith("-05:00")
    assert "fucking" in article["raw_text"] and "🚀" in article["title"] and "NOT" in article["title"]
    summer={**item(0,2026),"webPublicationDate":"2026-10-19T02:30:00Z"}
    assert normalise(summer,2026)["date_et"]=="2026-10-18"
    with tempfile.TemporaryDirectory(prefix="guardian_api_verify_") as directory:
        folder=Path(directory)
        check_key_cache_and_stop(folder)
        check_auxiliary_robots(folder)
        settings=PilotSettings(root=folder)
        result=api["run"](settings,client=FakeClient())
        assert len(result["articles"])==6 and len(result["sample"])==6
        assert result["sample"].relevant_us_equity_market.eq("").all()
        dedup=json.loads((settings.results/"deduplication_summary.json").read_text())
        assert dedup["raw_retrieved_records"]==36 and dedup["duplicate_count"]==30
        assert result["comparisons"].status.eq("complete").all()
        estimates=pd.read_csv(settings.results/"volume_estimates.csv")
        assert estimates.estimated_api_calls.eq(6).all()
        assert not (folder/"data"/"external").exists()
        # 150개 표본은 실제 충분한 기사 풀에서만 추출. 라벨 보존과 재현성 확인.
        many=[]
        for year in PILOT_WINDOWS:
            many.extend({**normalise(item(i,year),year),"query_strategy":"anchors_global"} for i in range(80))
        raw,unique,_=deduplicate(many)
        sample=manual_sample(unique,settings)
        assert len(sample)==150 and sample.groupby("year").size().eq(50).all()
        sample.loc[0,"relevant_us_equity_market"]="1"
        sample.to_csv(settings.results/"manual_review_sample.csv",index=False)
        again=manual_sample(unique,settings)
        assert again.guardian_id.tolist()==sample.guardian_id.tolist()
        assert str(again.loc[0,"relevant_us_equity_market"])=="1"
        with patch.dict("os.environ",{"GUARDIAN_API_KEY":""}):
            (folder/".env").write_text("GUARDIAN_API_KEY=\n")
            assert api["run"](settings)["status"]=="credentials_missing_cached_results_preserved"
    notebook=nbformat.read(ROOT/"guardian_api_pilot.ipynb",as_version=4)
    nbformat.validate(notebook)
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type=="code":
            compile(cell.source,f"guardian-api-cell-{i}","exec")
    print("PASS: credentials/cache/redaction/429 stop/full-collection guard/UTC-ET/OR dedup/150-item sampling/label preservation/count estimates/notebook")


if __name__=="__main__":
    main()
