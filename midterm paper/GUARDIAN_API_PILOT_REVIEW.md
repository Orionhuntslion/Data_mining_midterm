# Guardian API 2차 파일럿 비교

목표 기간: **2017-01-01~2026-10-25**. 상태: **completed**.
키 설정 위치 확인 결과: `project_dotenv`. 키 값은 기록하지 않는다.

## 실제 실행 상태

UTC 버퍼 포함 API 수신 레코드 425건.
ET 파일럿 구간 내 전략 소속 중복 포함 관측치 374건,
중복 218건, 고유 기사 156건.
키가 없으면 이 0은 실제 검색 결과 0건을 뜻하지 않는다. API total·규모 추정은 미조회로 남긴다.
이전 공개 HTML 9건은 공식 API 비교 자료에 넣지 않았다.

| 전략 | 2017 | 2020 | 2026 |
|---|---|---|---|
| anchors_global | complete; API total=36; ET unique=34 | complete; API total=53; ET unique=45 | complete; API total=33; ET unique=30 |
| anchors_business | complete; API total=9; ET unique=7 | complete; API total=15; ET unique=12 | complete; API total=6; ET unique=5 |
| geography_and_equity | complete; API total=36; ET unique=35 | complete; API total=90; ET unique=78 | complete; API total=42; ET unique=38 |
| broad_business_control | complete; API total=9; ET unique=8 | complete; API total=22; ET unique=18 | complete; API total=8; ET unique=7 |
| anchors_stock_market_tag | complete; API total=6; ET unique=6 | complete; API total=12; ET unique=11 | complete; API total=7; ET unique=5 |
| anchors_us_office | complete; API total=9; ET unique=9 | complete; API total=20; ET unique=16 | complete; API total=12; ET unique=10 |

## 검색 전략

- **anchors_global**: `{"q": "\"Wall Street\" OR \"S&P 500\" OR Nasdaq OR \"Dow Jones\" OR \"US stocks\" OR \"US markets\""}`. 정확한 미국시장 명칭 중심. 섹션 제한 없이 기술/경제 기사 포함. Wall Street의 비시장 용례는 수동 검토.
- **anchors_business**: `{"q": "\"Wall Street\" OR \"S&P 500\" OR Nasdaq OR \"Dow Jones\" OR \"US stocks\" OR \"US markets\"", "section": "business"}`. 동일 검색어 + Business. 정밀도 개선과 Technology 등 관련 기사 누락을 비교.
- **geography_and_equity**: `{"q": "(US OR \"U.S.\" OR American OR \"Wall Street\" OR \"Federal Reserve\") AND (stocks OR equities OR \"Wall Street\" OR \"S&P 500\" OR Nasdaq OR \"Dow Jones\")"}`. 미국 지리어 AND 주식시장 용어. 일반 미국주식 표현을 포괄하되 본문에 우연히 함께 나오는 용례 점검.
- **broad_business_control**: `{"q": "stocks OR equities OR \"Wall Street\" OR \"S&P 500\" OR Nasdaq OR \"Dow Jones\"", "section": "business"}`. stocks/equities를 포함한 넓은 Business 대조군. 비미국 주식시장 혼입을 점검.
- **anchors_stock_market_tag**: `{"q": "\"Wall Street\" OR \"S&P 500\" OR Nasdaq OR \"Dow Jones\" OR \"US stocks\" OR \"US markets\"", "tag": "business/stock-markets"}`. 같은 명칭 검색 + stock-markets 태그. 실제 태그 ID 확인 후만 요청; 옛 기사 태깅 누락 가능.
- **anchors_us_office**: `{"q": "\"Wall Street\" OR \"S&P 500\" OR Nasdaq OR \"Dow Jones\" OR \"US stocks\" OR \"US markets\"", "production-office": "us"}`. 같은 검색 + 미국 제작 오피스. US 에디션/미국시장 관련성과 동일하지 않음.


## 키 설정과 실행

같은 폴더의 **`.env`**에서 다음 등호 뒤에 개발자 키를 입력한다.

```dotenv
GUARDIAN_API_KEY=발급받은_키
```

파일은 git에서 제외된다. `.env.example`은 배포용 빈 예시이다.
현재 프로세스의 `GUARDIAN_API_KEY` 환경변수가 있으면 `.env`보다 우선한다.
VS Code 커널/터미널을 이미 실행 중이라면 파일 방식이 간단하다. 키를 채팅이나 노트북에 쓰지 않는다.

```powershell
python guardian_api_pilot.py
```

실제 환경 실행 파일이 필요하면 `C:\Users\rhffl\anaconda3\envs\hwenv\python.exe`를 사용한다.
등록: [Guardian Developer API](https://open-platform.theguardian.com/access/).

## 표본·시각·중복 검증

- 2017-01-09~15, 2020-03-16~22, 2026-10-19~25의 일주일 구간을 비교한다.
- 모든 원응답은 조건 해시+연도+전략+페이지별 저장이다. 성공한 캐시는 재사용한다.
- 기본 page-size=200, 주간별 최대 3페이지다. 응답의 pageSize/pages를 확인하며 실제 지원 여부는 키 설정 후 검증한다.
  400이면 지원되지 않은 파라미터/페이지 크기를 보고하고 조용히 조건을 바꾸지 않는다.
- `partial_page_cap`이면 검색 결과 전체를 받지 못한 것이다. 다운로드 순서는 oldest이므로 이때는 주간 전체의 무작위 표본이 아니다.
- 날짜 필터는 UTC 날짜 끝에 하루 버퍼를 두고, 원래 발행 문자열·UTC·America/New_York·ET 날짜를 저장한다.
  API total은 버퍼 구간의 값이다. 실제 ET 구간 문서 수는 별도 저장하며 둘을 같은 수로 표시하지 않는다.
- 안정적인 Guardian ID(없으면 URL)로 중복을 제거하고 전략별 소속을 모두 보존한다.
- `manual_review_sample.csv`: 다운로드된 고유 기사 합집합에서 연도별 최대 50건을 고정 시드로 무작위 추출한다.
  `relevant_us_equity_market`는 빈칸이다. 사람이 1/0을 입력하면 다시 실행 시 전략별 precision과 Wilson 구간을 계산한다.
  표본의 전략별 수는 `manual_precision.csv`에서 확인한다. 일부 전략에 검토 표본이 적으면 정밀도 결론을 유보한다.
- 기존 사람이 입력한 라벨은 ID로 재사용하고 `manual_review_labels.csv`에도 보존한다.
- sections·editions 목록과 keyword tags·productionOffice 분포를 저장한다. 미국 제작 오피스/US 에디션은 시장 관련성 라벨이 아니다.
- q 검색 범위와 반환한 headline/trailText의 범위가 같다고 가정하지 않는다. 반환한 관측 텍스트 자체의 관련성을 수동 검토한다.
- 캐시 시점이 다르면 검색 인덱스 변경으로 페이지가 달라질 수 있다. total 변화는 완전 수집으로 표시하지 않는다.

## 전체 기간 규모 추정

각 전략당 **page-size=1, page=1의 메타데이터 요청만** 전체 목표 기간에 보낸다.
full-period pages를 순회하는 기능은 없다. 이 total과 파일럿의 실제 페이지 크기·원응답 바이트를 사용한다.
UTC 경계 버퍼가 있어 수집량은 약간 넓은 범위의 추정값이며 전략별 total을 더해 합집합 수로 쓰지 않는다.

- API 호출 수 ≈ ceil(total / observed pageSize).
- 속도상 최소 실행 시간 ≈ (호출 수 − 1) × 1.1초. 통신 지연/일일 한도 대기는 제외한다.
- 일일 한도상 최소 일수 ≈ ceil(호출 수 / 500). 같은 키의 다른 사용량은 알 수 없다.
- 원응답 저장 용량 ≈ total × 파일럿에서 관측된 JSON 바이트/문서. 수정·태그 길이에 따라 달라진다.
- `volume_estimates.csv`에 근거와 수치를 저장한다. 키가 없거나 메타데이터 실패 시 숫자를 만들지 않는다.
- 세 주간만 10년으로 단순 외삽하지 않는다. COVID 주간은 보통 시기의 무작위 표본이 아니다.

공식 개발자 한도: 초당 1회·일 500회. 로컬 클라이언트는 1.1초 간격, 실행당 최대 90요청,
보수적인 최근 24시간 500요청 제한을 사용한다. robots 요청도 로컬 예산에 포함한다.
401/403/418/429는 자동 재시도하지 않고 전체 실행을 중단한다.
이는 **Content API 응답**의 처리이다. `/robots.txt`의 401은 별도 보조 검사 실패로 기록하고
`/search` 인증 실패로 판정하지 않는다. 실제로 받은 robots 금지 규칙은 준수하며 공식 JSON API 외 HTML 경로는 허용하지 않는다.

이전 중단 경로는 `fetch → PublicClient.get → robots 요청 → PublicClient._request`가 401을
`AccessError`로 바꾼 후, 래퍼 `logged_request`가 엔드포인트를 구분하지 않고
`halted=True`와 `PilotStopped`를 발생시킨 것이었다.
수정 후 robots 검사는 분리했고, 실제 API 401/403/429 중단·요청 간격·예산은 유지했다.

## 현재 추천과 채택 조건

**잠정 기준 후보는 `anchors_global`**이다. 명확한 미국시장 명칭을 쓰면서 섹션·제작 오피스 제한으로
Technology 등 관련 기사나 영국에서 작성한 미국시장 뉴스를 미리 제외하지 않는 접근이다.
이는 실측 정밀도 우승 전략이라는 뜻이 아니다. 키가 없거나 수동 검토가 끝나지 않았다면 최종 선택을 하지 않는다.
`geography_and_equity`의 추가 포괄성과 Business/태그/오피스 제한의 정밀도·누락을 세 연도별로 비교한 후 결정한다.
FGI 상관이 높다는 이유로 검색 전략을 고르지 않는다.

Guardian만 다룬다. Reddit/FNSPID/FGI/S&P 500 결합, 일별 감성 집계, 전체 수집은 실행하지 않는다.
Reddit은 전 기간 주 자료로 보지 않고, FNSPID는 향후 독립적인 2017~2023 강건성 자료로만 검토한다.

근거: [공식 검색/필드 문서](https://open-platform.theguardian.com/documentation/),
[공식 클라이언트의 검색·태그·페이지 처리](https://github.com/guardian/content-api-scala-client),
[개발자 키·한도](https://open-platform.theguardian.com/access/).
