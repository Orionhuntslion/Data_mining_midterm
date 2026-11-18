"""Guardian 공식 API 요청, 키 비노출, 페이지 캐시, 속도·요청 수 제한."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from urllib.robotparser import RobotFileParser

import requests
from guardian_api_config import validate_request

API="https://content.guardianapis.com"


def write_json(value,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(path)


class PilotStopped(RuntimeError):
    pass


class QueryRejected(RuntimeError):
    pass


class GuardianClient:
    def __init__(self,settings,key):
        # HTML 수집기의 인증 판단과 API 호스트의 보조 robots 확인을 분리한다.
        from midterm_common import PublicClient
        self.settings=settings
        self.key=key
        self.public=PublicClient(settings.interval_seconds)
        self.log=settings.results/"http_log.jsonl"
        self.used=0
        self.halted=False
        self.api_robots_checked=False
        self.api_robots_parser=None
        original=self.public._request

        def logged_request(url,**kwargs):
            self.check_budget()
            self.used+=1
            before=time.monotonic()
            # URL/예외 문자열에는 키가 들어갈 수 있으므로 저장하지 않는다.
            auxiliary=url==API+"/robots.txt"
            event={"attempted_at":datetime.now(timezone.utc).isoformat(),"endpoint":url.split("?")[0],
                   "request_role":"robots_auxiliary" if auxiliary else "content_api"}
            try:
                if auxiliary:
                    # 같은 속도/예산을 적용하되 robots의 HTTP 상태를 API 인증 오류로 변환하지 않는다.
                    time.sleep(max(0,self.public.interval-(time.monotonic()-self.public.last_request)))
                    self.public.last_request=time.monotonic()
                    response=self.public.session.get(url,timeout=60,allow_redirects=False,**kwargs)
                else:
                    response=original(url,**kwargs)
                event.update(status=response.status_code,bytes=len(response.content),elapsed_seconds=time.monotonic()-before)
                if auxiliary and response.status_code!=200:
                    event["inspection_status"]="unavailable; not evidence of Content API authentication failure"
                return response
            except Exception as exc:
                from midterm_common import AccessError
                if auxiliary:
                    event.update(status=None,error_type=type(exc).__name__,inspection_status="unavailable; official API authentication not tested")
                    return None
                if isinstance(exc,AccessError):
                    self.halted=True
                    # 코드만 추출하고 원래 예외/URL은 출력하지 않는다.
                    import re
                    match=re.search(r"HTTP (401|403|418|429)",str(exc))
                    event.update(status=int(match[1]) if match else None,error_type="AccessError")
                    raise PilotStopped(f"Guardian 접근/요청 제한: HTTP {event['status']}. 원문 응답/URL은 키 보호를 위해 생략.") from None
                event.update(status=None,error_type=type(exc).__name__)
                raise PilotStopped(f"Guardian 네트워크 오류: {type(exc).__name__}. 재시도 없이 중단.") from None
            finally:
                self.log.parent.mkdir(parents=True,exist_ok=True)
                with self.log.open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps(event,ensure_ascii=False)+"\n")
        self.public._request=logged_request

    def inspect_api_robots(self):
        """공식 API만의 보조 검사. 실제로 받은 명시적 금지 규칙은 계속 준수한다."""
        if self.api_robots_checked:
            return
        response=self.public._request(API+"/robots.txt")
        self.api_robots_checked=True
        record={"endpoint":API+"/robots.txt","role":"auxiliary_compliance_inspection",
                "checked_at":datetime.now(timezone.utc).isoformat(),"api_authentication":"not_determined_by_robots",
                "http_status":response.status_code if response is not None else None}
        if response is not None and response.status_code==200 and "<html" not in response.text[:500].lower():
            parser=RobotFileParser(API+"/robots.txt")
            parser.parse(response.text.splitlines())
            self.api_robots_parser=parser
            agent=self.public.session.headers["User-Agent"]
            delay=parser.crawl_delay(agent) or parser.crawl_delay("*")
            if delay:
                self.public.interval=max(self.public.interval,float(delay))
            record.update(status="parsed",rules=response.text.replace(self.key,"REDACTED"))
        else:
            record.update(status="unavailable",note="robots 확인 실패는 /search 인증 실패가 아니다. 명시적으로 허용된 공식 JSON API만 계속 사용하고 HTML로 대체하지 않는다.")
        write_json(record,self.settings.results/"robots_inspection.json")

    def get_official_api(self,endpoint,params):
        if endpoint not in {"/search","/sections","/editions","/tags"}:
            raise ValueError("허용된 Guardian Content API 엔드포인트만 사용할 수 있습니다.")
        self.inspect_api_robots()
        url=API+endpoint
        if self.api_robots_parser is not None:
            agent=self.public.session.headers["User-Agent"]
            checked_url=requests.Request("GET",url,params=params).prepare().url
            if not self.api_robots_parser.can_fetch(agent,checked_url):
                self.halted=True
                raise PilotStopped(f"robots.txt의 명시적 금지 규칙: {endpoint}. API 인증 실패와 구분하여 중단합니다.")
        response=self.public._request(url,params=params,allow_redirects=False)
        response.raise_for_status()
        return response

    def check_budget(self):
        if self.halted or self.used>=self.settings.max_requests_per_run:
            raise PilotStopped("파일럿 중단 또는 실행당 요청 한도 도달. 저장한 페이지부터 재개할 수 있습니다.")
        cutoff=datetime.now(timezone.utc)-timedelta(hours=24)
        count=0
        # 파일럿과 전체 수집이 같은 키의 일일 예산을 공유한다.
        logs=set((self.settings.root/"results").glob("*/http_log.jsonl"))|{self.log}
        for log in logs:
            if not log.exists():
                continue
            for line in log.read_text(encoding="utf-8").splitlines():
                event=json.loads(line)
                if event.get("endpoint","").startswith(API+"/") and event.get("attempted_at") and datetime.fromisoformat(event["attempted_at"])>=cutoff:
                    count+=1
        if count>=self.settings.daily_request_limit:
            raise PilotStopped("최근 24시간 로컬 공유 요청 한도 도달. 다른 앱의 동일 키 사용량은 포함되지 않습니다.")

    def fetch(self,endpoint,params,kind,cache):
        validate_request(params,kind)
        if "api-key" in params:
            raise ValueError("키를 공개 파라미터에 넣지 마세요.")
        fingerprint=hashlib.sha256(json.dumps({"endpoint":endpoint,"params":params},sort_keys=True).encode()).hexdigest()
        cache=Path(cache)
        if cache.exists():
            envelope=json.loads(cache.read_text(encoding="utf-8"))
            if envelope["request_sha256"]!=fingerprint:
                raise ValueError("캐시 조건 불일치. 기존 실행 폴더를 보존하고 새 폴더로 실행하세요.")
            return envelope,True
        try:
            response=self.get_official_api(endpoint,params={**params,"api-key":self.key})
        except requests.HTTPError as exc:
            status=exc.response.status_code
            if status==400:
                raise QueryRejected("HTTP 400: 검색 조건/필드/페이지 크기 지원 여부를 확인하세요.") from None
            self.halted=True
            raise PilotStopped(f"Guardian HTTP {status}: 저장된 페이지를 보존하고 중단합니다.") from None
        except PilotStopped:
            raise
        except Exception as exc:
            self.halted=True
            raise PilotStopped(f"Guardian 접근 규칙/요청 오류: {type(exc).__name__}") from None
        try:
            payload=response.json()["response"]
            if payload["status"]!="ok" or "results" not in payload:
                raise ValueError("invalid schema")
        except (ValueError,KeyError,TypeError):
            raise QueryRejected("Guardian JSON 스키마/오류 응답. 원문 키가 노출되지 않도록 메시지만 표시합니다.") from None
        # 키가 응답에 반사되어도 파일에 저장하지 않는다.
        text=json.dumps(payload,ensure_ascii=False).replace(self.key,"REDACTED")
        envelope={"endpoint":endpoint,"params":params,"request_sha256":fingerprint,
            "retrieved_at":datetime.now(timezone.utc).isoformat(),"response_bytes":len(response.content),"response":json.loads(text)}
        write_json(envelope,cache)
        return envelope,False
