# 데이터 수집·품질 진단에 사용한 라이브러리

실행 환경: **Python 3.10.21**, SQLite 엔진 **3.53.2**.
전체 원자료 수집은 Guardian 공식 Content API만 사용했다. 일별·주별 건수 진단은 수집 자료의 표본 규모 확인이다.

## 전체 수집·저장·품질 진단

| 패키지 | 확인 버전 | 용도 |
|---|---|---|
| requests | 2.34.2 | 공식 Guardian API GET, 인증·오류·요청 간격 처리 |
| pandas | 2.3.3 | 발행 시각 UTC/뉴욕 변환, 건수 집계, CSV, 품질 진단 |
| pyarrow | 25.0.1 | 연도별 Parquet 쓰기·배치 읽기·시간대 스키마 보존 |
| numpy | 2.2.5 | 고정 시드 무작위 검사 표본, 수치 계산 |
| matplotlib | 3.10.9 | 월별 기사량·주별 분포 그래프 |

`sqlite3`는 외부 설치 패키지가 아니라 Python 표준 라이브러리다. 페이지별 SQLite와 최종 Parquet는 역할이 다르다.
requests/pandas/pyarrow 버전은 전체 원자료 검증 당시 기록과 함께 CSV에 저장했다. 나머지 버전은 같은 실행 환경에서 현재 조회한 값이다.

## 초기 출처 탐색·노트북 도구

| 패키지 | 확인 버전 | 사용 범위 |
|---|---|---|
| beautifulsoup4 | 4.15.0 | 초기 공개 HTML 출처 파일럿의 제목/설명 메타데이터 추출. 전체 수집은 공식 API라 미사용 |
| nbformat | 5.11.0 | 주피터 노트북 생성·구조 검증 |
| nbclient | 0.11.0 | 노트북 자동 실행·검증 |
| ipykernel | 7.3.0 | Python 주피터 실행 커널 |
| jupyter_client | 8.10.0 | 노트북 커널 실행·관리 |
| ipython | 8.30.0 | 노트북 표·그림·Markdown 표시 |

BeautifulSoup를 전체 Guardian API 크롤링에 사용했다고 기술하지 않는다. HTML 출처 파일럿과 공식 API 전체 수집을 구분한다.

## 표준 라이브러리

- `sqlite3`: 원자료 ID 중복 병합, 페이지별 트랜잭션, 재개 기록; SQLite 엔진 3.53.2.
- `pathlib`: 상대/절대 파일 경로와 디렉터리 관리.
- `json`: 원응답과 메타데이터·진단 JSON 보존.
- `hashlib`: 요청 조건·원자료 SHA-256 검증.
- `datetime / time`: 시각·요청 간격·24시간 예산 관리.
- `urllib.robotparser / urllib.parse`: robots 규칙 확인과 공식 API URL 검사.
- `os`: 환경변수에서 API 키 읽기; .env는 직접 파싱하며 python-dotenv는 사용하지 않음.
- `argparse / dataclasses`: CLI 옵션과 설정 관리.
- `sys / runpy`: 실행 환경 경로와 기존 공통 모듈 재사용.
- `math`: 파일럿 요청량 추정과 precision 참고 구간 계산.
- `tempfile / unittest.mock`: 임시 합성 자료와 네트워크 없는 회귀 검증.
- `collections.Counter`: Parquet 날짜와 소속 플래그의 주간 건수 독립 교차 검증.

## 분석용 의존성과의 구분

저장소 requirements.txt에는 향후 분석용 VADER·SciPy·scikit-learn·wordcloud도 포함된다.
이들은 이번 전체 원자료 수집·주간 건수 진단에 사용한 도구 목록에 포함하지 않는다.
초기 별도 HTML 소표본의 VADER 시험과 달리, 최종 원자료 5,265건에는 감성점수·WSI·FGI·relevance 라벨을 붙이지 않았다.
별도 pip 패키지인 python-dotenv는 사용하지 않았다. .env는 표준 라이브러리로 읽고 비밀값을 출력하지 않는다.
