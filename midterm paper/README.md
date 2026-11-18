# 중간 페이퍼 코드 · 최종 WSI–S&P 500 수익률 탐색 분석

**최종 정량 단계:** `guardian_market_analysis_review.ipynb`와 `GUARDIAN_MARKET_HANDOFF.md`를 읽는다.
기존 전처리·VADER·WSI·FGI 자료/결과를 보존하고 FRED SP500 일별 종가로 주간 로그수익률을 만들었다.
Broad/strict count·continuous와 FGI 맥락 비교를 각각 절편+단일 설명변수 OLS로 계산했다.
동시점과 정확한 다음 달력 주 관계만 포함한다. 예측·ML·다변량·추가 분석은 수행하지 않는다.

```powershell
python guardian_market_analysis.py
python verify_guardian_market_analysis.py
python build_guardian_market_notebook.py --execute
# 현재 기본 stage도 guardian-market; upstream 분석을 재실행하지 않는다.
python "midterm paper.py" --stage guardian-market
```

코드: `guardian_market_data.py`(공개 CSV·캐시), `guardian_market_analysis.py`(검증·주간화·병합),
`guardian_market_stats.py`(승인된 단변량 상관·OLS), `guardian_market_outputs.py`(그림 2종·기술 인계).
원응답: `data/raw/market/`. 결과: `results/guardian_market_analysis/`, `figures/guardian_market_analysis/`.
기존 결과 폴더는 수정하지 않고 SHA-256을 대조한다. 이 단계 이후 추가 분석은 명시적 요청을 기다린다.
아래는 이전 단계 기록이다.

**WSI–FGI 검증 단계:** `guardian_fgi_validation_review.ipynb`와 `GUARDIAN_WSI_FGI_VALIDATION_REVIEW.md`를 읽는다.
기존 broad/strict의 count·continuous WSI와 보관한 일별 FGI를 사용한다. 전처리·채점·WSI 규칙은 수정하지 않는다.
FGI 주간 평균(주 명세)·마지막 값(강건성), 동시적 Pearson/Spearman, 정확한 달력 기준 -4~+4주 시차,
1차 차분 강건성까지 계산했다. 보간·전방 채움은 없다. 시장수익률 분석·회귀·예측·다변량 모형은 수행하지 않았다.

```powershell
python guardian_fgi_validation.py
python verify_guardian_fgi_validation.py
python build_guardian_fgi_notebook.py --execute
# 명시적 WSI–FGI 단계; 전처리나 VADER를 재실행하지 않는다.
python "midterm paper.py" --stage guardian-fgi-validate
```

코드는 `guardian_fgi_validation.py`(FGI 검증·주간화·병합), `guardian_fgi_stats.py`(상관·시차·차분),
`guardian_fgi_outputs.py`(그림·진단 보고)로 나누었다.
결과는 `results/guardian_fgi_validation/`와 `figures/guardian_fgi_validation/`에 저장했다.
거래일 대조는 exchange_calendars 4.13.2의 XNYS 기준이다. 새로운 시장 가격이나 FGI를 요청하지 않는다.
아래는 이전 단계의 기록이다.

**VADER·WSI 단계:** `guardian_sentiment_review.ipynb`와 `GUARDIAN_SENTIMENT_REVIEW.md`에서 결과를 검토한다.
확정된 `guardian_text_v2_final`의 text_sentiment를 수정 없이 VADER 3.3.2로 평가했다.
5,265건의 물리적 기사와 모든 원래 열을 보존하고 네 점수·sentiment_class만 추가했다.
Broad 주 자료와 strict 강건성 자료의 count WSI 및 continuous WSI를 각각 만들었다.
기사 0건 주의 감성·WSI는 결측이다. FGI 결합·시차·시장수익률·회귀·예측은 수행하지 않았다.

```powershell
python guardian_sentiment_analysis.py
python verify_guardian_sentiment.py
python build_guardian_sentiment_notebook.py --execute
# 같은 분석을 실행하는 명시적 진입점
python "midterm paper.py" --stage guardian-sentiment
```

코드는 `guardian_sentiment_analysis.py`(채점·저장), `guardian_weekly_sentiment.py`(주간 집계·WSI),
`guardian_sentiment_outputs.py`(그림·검사 표본·진단 보고)로 나누었다.
결과: `data/processed/guardian_vader/articles/`, `results/guardian_sentiment/`, `figures/guardian_sentiment/`.
전처리 정책·text_clean 재생성은 이 단계에서 실행하지 않는다. 아래는 이전 단계의 기록이다.

**Validating the Fear & Greed Index with Web-Based Financial Sentiment: Correlation and Lead-Lag Analysis**

목표 기간은 **2017-01-01~2026-10-25**, 정렬 기준은 `America/New_York`이다.
사용자 승인에 따라 `broad_business_control`(주 자료)과 `anchors_stock_market_tag`(엄격 강건성)의
전체 원자료 수집을 완료했다. `geography_and_equity`는 전체 수집하지 않았다.
전처리 완료 당시 CLI 동작은 `guardian-preprocess`였으며 감성 점수·WSI·FGI 결합을 실행하지 않았다.

**전처리 동결 단계:** `guardian_text_preprocessing_final.ipynb`, `FINAL_PREPROCESSING_REVIEW.md`,
`FINAL_PREPROCESSING_POLICY.json`/`.md`를 먼저 읽는다.
최종 추가 불용어는 happened/latest/coverage/rolling/live/news 여섯 개이며 text_clean 계열에만 적용한다.
new/year/business/says는 유지하고 표제어화·추가 문구 삭제는 비활성화했다.
최종 Parquet는 `data/processed/guardian_text_final/articles/`, 최종 진단은 `results/guardian_text_final/`이다.
이전 v1 자료와 결과는 그대로 보존했다. text_raw/text_sentiment는 모든 기사에서 변경 전후 동일하다.

```powershell
python guardian_preprocessing_finalize.py
python verify_guardian_text_rules.py
python verify_guardian_final_preprocessing.py
```

워드클라우드는 `figures/guardian_text_final/`에 모든 연도와 전체 기간·대표 연도 패널로 저장한다.
정책은 최종 명세로 동결했으며 변경 시 명시적으로 새 정책 버전을 만들어야 한다.
전처리 동결 시점에는 감성 점수·WSI·FGI 결합·시차·수익률 분석을 시작하지 않았다.

아래는 최초 전처리 단계의 기록이다.
`guardian_text_preprocessing.ipynb`와 `GUARDIAN_TEXT_PREPROCESSING_REVIEW.md`는 v1 비교 기준이다.
5,265건의 모든 원본 열을 보존하고 `text_raw`, `text_sentiment`, `text_clean`, 토큰과 주 ID를 별도로 만들었다.
처리 자료는 `data/processed/guardian_text/articles/`, 진단은 `results/guardian_text_preprocessing/`에 있다.
주 자료 broad와 엄격 강건성 strict는 같은 기사 레코드의 원래 소속 플래그로 각각 선택한다.
원본 파일은 덮어쓰지 않았다. 추가 불용어와 표제어화는 아직 적용하지 않았다.

```powershell
python guardian_text_preprocess.py
python verify_guardian_text_rules.py
python verify_guardian_text_data.py
```

NLTK stopwords/WordNet 자원은 프로젝트 `.nltk_data/corpora/`의 공식 ZIP으로 읽는다.
URL과 SHA-256은 `.nltk_data/resource_manifest.json` 및 진단의 `cleaning_policy.json`에 저장했다.
NLTK 다운로더가 이 환경의 공백 포함 경로를 거부하여 공식 문서의 수동 ZIP 저장 방식을 사용했다.
전처리 스크립트는 자원을 자동 다운로드하거나 외부 API를 요청하지 않는다.
이전 CSV용 `midterm paper_03.py`는 보존했지만 새 Guardian Parquet 전처리에는 별도 규칙을 사용한다.

아래는 수집 단계의 기록이다.

먼저 **`GUARDIAN_FULL_COLLECTION_REVIEW.md`**를 읽는다.
최종 자료는 `data/raw/news/guardian_full/articles/`의 연도별 Parquet이며,
`results/guardian_full/`에 품질 진단·연도별/월별/일별 건수·무작위 검사 표본이 있다.
같은 기사는 하나만 저장하고 `in_broad_business_control`, `in_anchors_stock_market_tag`를 함께 보존한다.

```powershell
python guardian_full_collection.py
# 완료 페이지 재사용. 아래는 API 없이 진단과 선택적 전체 CSV만 다시 생성한다.
python guardian_full_collection.py --diagnostics-only --export-csv
```

전체 수집 코드는 `guardian_full_collection.py`, 저장/진단은 `guardian_full_diagnostics.py`이다.
회귀 검증: `python verify_guardian_full_collection.py`.
공식 API 200건/페이지·1.1초 요청 간격과 파일럿/전체 수집의 공유 24시간 예산을 적용한다.
API 401/403/429는 중단하고 성공 페이지를 보존한다. 원응답과 SQLite의 페이지 트랜잭션으로 재개한다.
원문은 수정하지 않았으며 파일럿 relevance 라벨을 감성 라벨로 사용하지 않는다.
현재는 원자료와 품질 진단 검토 단계이다.

수집 단계 마지막 주별 진단: `python guardian_weekly_coverage.py`.
뉴욕 기준 월~일, 완전한 512주를 주 기준으로 사용하며 1일만 포함된 시작 경계 주는 별도 표시한다.
`results/guardian_full/weekly_article_counts.csv`, `weekly_distribution_summary.csv`에 주별 수와 통계를 저장했다.
설명은 `GUARDIAN_WEEKLY_COVERAGE.md`, 실제 사용 라이브러리와 버전은 `COLLECTION_LIBRARIES.md`에 정리했다.
원문이나 수집 범위를 바꾸지 않았고 텍스트 전처리·감성 분석·WSI는 시작하지 않았다.

아래 내용은 이전 파일럿의 기록이다.

**현재 작업: 공식 Guardian API 2차 파일럿.** `guardian_api_pilot.ipynb`과
`GUARDIAN_API_PILOT_REVIEW.md`를 먼저 읽는다. 키는 같은 폴더 `.env`의
`GUARDIAN_API_KEY=` 뒤에 넣는다. `.env`는 git 제외 대상이며 환경변수도 지원한다.
robots.txt의 401을 API 인증 실패로 잘못 처리하던 경로를 수정한 뒤 실제 API 파일럿을 완료했다.
6개 전략 × 3개 연도의 `/search`가 정상 응답했고, ET 구간 관측치 374건에서
중복 218건을 제거해 고유 기사 156건을 확보했다. 수동 검토 표본은 2017년 36건,
2020년 50건, 2026년 38건으로 총 124건이다.
검토 파일은 `results/guardian_api_pilot/manual_review_sample.csv`이다.
키 설정 후 `python guardian_api_pilot.py`를 실행한다.
검증은 `python verify_guardian_api_pilot.py`로 실행한다.
기존 HTML 9건을 새 API 자료에 넣지 않으며, 전체 수집은 여전히 차단한다.

**제공 라벨 반영:** `manual_review_sample_labeled.csv`와 `guardian_strategy_precision.csv`를 검증했다.
`python guardian_strategy_yield.py`는 추가 요청 없이 기사 수×precision·참고 구간·잡음량·연도별 차이를
`results/guardian_api_pilot/strategy_yield_comparison.csv`에 저장한다.
해석은 `GUARDIAN_STRATEGY_YIELD_REVIEW.md`를 참고한다. 최종 전략은 확정하지 않는다.

아래는 1차 출처 파일럿의 기록이다.
먼저 **`midterm_source_pilot.ipynb`**와 **`SOURCE_PILOT_REVIEW.md`**를 읽는다.
노트북은 저장한 실제 결과를 조회하며 기본 설정에서는 네트워크 요청을 다시 보내지 않는다.

## 현재 확인 상태

- Guardian 공개 아카이브: 2017/2020/2026 각 3건, 총 **9건**의 제목·설명·발행 시각·URL 확인.
- 미국시장 키워드 검출은 2/9건. 본 수집 전에 시장 관련성 필터를 추가 검증해야 한다.
- Guardian API: 키 없어 미시도. 키는 `GUARDIAN_API_KEY` 환경변수에서만 읽는다.
- GDELT: 연결 종료와 429. 제한 이후 해당 호스트 요청 중단.
- FNSPID: 파일 목록만 확인. 대용량 뉴스 행과 날짜별 표본은 미검증.
- Reddit: 공식 연구 접근·제공 범위 제약으로 게시물 수집 안 함.
- FGI: 기존 원응답에서 **2,443일분** 확인. 2021년 전후 자료 계보를 구분해 저장.

기존 `data/processed/fear_greed_data.csv`의 59일은 이전 실행 산출물이다.
이번 전체 범위 파일은 `data/external/fear_greed_2017-01-01_2026-10-25.csv`이다.
파일럿 9건으로 일별 WSI·상관을 추정하지 않았다. 보고서·연구 결론은 생성하지 않는다.

## 파일과 역할

| 파일 | 역할 |
|---|---|
| `midterm_source_pilot.ipynb` | 현재 출처·실제 응답·샘플·FGI 커버리지 리뷰 |
| `SOURCE_PILOT_REVIEW.md` | 성공/실패·접근 조건·다음 판단 사항 |
| `midterm paper.py` | 진입점, 기본 단계 `pilot` |
| `midterm paper_01.py` | 공통 설정·경로·HTTP/robots 규칙 |
| `midterm paper_02.py` | 기존 뉴스·FGI 수집/가져오기, 전체 수집 가드 |
| `midterm paper_03.py` | 원문 보존·전처리·중복 제거 |
| `midterm paper_04.py` | VADER, `neg,neu,pos,compound` 보존 |
| `midterm paper_05.py` | 기존 일별 지수·FGI 결합 |
| `midterm paper_06.py` | 기존 국면별 빈도·TF-IDF·단어 시점 |
| `midterm paper_07.py` | 기존 상관·달력 날짜 시차 |
| `midterm paper_08.py` | 그림, 주 비교를 count로 변경 |
| `midterm paper_09.py` | 소규모 출처 파일럿·진단 로그 |
| `review_source_pilot.py` | 저장한 표본의 VADER/필드 점검·노트북 정리 |
| `midterm_sentiment.ipynb` | 기존 **뉴스 단독** 분석, 출처 검토 후 사용 |
| `verify_pipeline.py` | 기존 분석의 합성 자료 검증 |
| `verify_source_pilot.py` | 파일럿 파서·시간대·원문·키 가림·수집 가드 검증 |

번호 파일은 진입점/노트북에서 불러온다. 족보 PDF는 실증 자료에 포함하지 않는다.

## 실행

```powershell
cd 'C:\Users\rhffl\data_mining\midterm paper'
python -m pip install -r requirements.txt
python 'midterm paper.py' --stage pilot
python review_source_pilot.py
python verify_source_pilot.py
```

실제 환경 경로는 `C:\Users\rhffl\anaconda3\envs\hwenv\python.exe`이다.
`python`이 Windows Store 별칭이면 해당 경로를 사용한다.
프로젝트 내부 `.packages/`도 자동 탐색한다. 패키지를 설치한 환경과 커널을 같게 선택한다.

공개 요청은 robots 규칙과 최소 6초 간격을 확인한다. 401/403/418/429는 우회하지 않는다.
실패/미시도/자료 없음은 구분해 저장하고 API 키는 로그에서 가린다.
계정 생성·신청 제출은 자동으로 하지 않았다.

## 저장 구조

```text
data/raw/news/<source>/<year>/<month>/          날짜별 원응답과 문서
data/processed/pilot/news/<source>/<year>/<month>/  문서 JSONL
data/external/                                 전체 범위 FGI
results/source_pilot/                          HTTP 로그·요약·표본 점검·연도별 커버리지
```

작은 파일럿은 월별 JSON/JSONL로 저장한다. 전체 기간 원문을 메모리에 합치지 않는다.
`news_sample_audit.csv`는 실제 9건을 검토하는 작은 표이다.
전체 수집의 Parquet·청크 처리와 별도 `data/raw/reddit/`은 출처 선택 후 구현한다.

## 수정된 분석 기준

- 주 지수: `WSI_count = 50 * (1 + (P-N)/(P+N+U))`.
- 강건성 비교: `WSI_continuous = 50 * (mean(compound)+1)`.
- 뉴스와 Reddit 각각 계산. 통합 지수는 미확정.
- 원문·욕설·은어·이모지·대문자 보존, 빈도 입력과 감성 입력 분리.
- 원래 VADER 점수를 FGI에 맞춰 바꾸지 않는다.
- binary 비율은 중립 제외. 정규근사는 추정 비율의 표집분포에 관한 것으로 Probit·지수 정의의 근거가 아니다.
- FGI/뉴스 없는 날짜는 보간하지 않고 공통 유효 날짜로 비교한다.
- `rho(k)=Corr(WSI_t, FGI_(t+k))`; 양의 k는 오늘 웹과 미래 FGI 비교. 인과/예측 주장 없음.
- 연도별 전체/선택/패널 워드클라우드·분할 수집·Reddit 분석은 출처 검토 후 확장한다.
- S&P 500 로그 수익률 OLS는 최종 탐색 단계로 남긴다. Probit은 추가하지 않는다.

기존 분석 경로는 작은 자료/입력 CSV용으로 보존했다. 이를 장기 전체 원문용 구조로 간주하지 않는다.
`--stage all`을 지정해도 장기 공개 수집은 기본 가드가 막는다.
수집 활성화 옵션이 있어도 GDELT의 과거 접근 한계를 해결하지는 않는다.
