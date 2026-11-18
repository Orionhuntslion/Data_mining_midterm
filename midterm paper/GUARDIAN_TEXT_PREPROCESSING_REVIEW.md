# Guardian 텍스트 전처리 진단 · 후속 분석 전 검토용

원문 **5,265건**을 한 번씩 저장하고 기존 메타데이터·전략 소속을 모두 보존했다.
주 자료는 broad_business_control, 엄격 강건성 자료는 anchors_stock_market_tag이며 집계 단위는 뉴욕 시간 월요일~일요일이다.
주간 감성 점수·WSI·FGI 결합·시차 분석·수익률 분석은 수행하지 않았다.

## 세 가지 텍스트

- `text_raw`: 원래 headline(없으면 web_title)과 description을 줄바꿈으로 결합한다. HTML도 원래 문자열대로 남긴다.
- 표시 텍스트가 정확히 같은 제목·설명은 description을 결합에서 한 번 제외한다. 원래 headline/description 열은 그대로 보존한다.
- `text_sentiment`: HTML/script/style와 비언어적 공백·엔티티만 정리한다. 대소문자·부정·욕설·은어·문장부호·이모지를 보존하고 불용어를 제거하지 않는다.
- `text_clean`: HTML 제거 후 lowercase/NFKC·URL 제거·Unicode 토큰화·기본 영어 불용어 제거를 적용한다.
- 금융 방향·지표·부정 표현은 불용어에서 보호한다. S&P 500→s&p_500, U.S.→us 등 문서화된 시장 표기를 정규화한다.
- 수치·내부 apostrophe/hyphen·$ticker·시장 약어·욕설·이모지는 일괄 삭제하지 않는다. 표제어화와 사용자 정의 불용어는 적용하지 않았다.

## 제목·설명 중복 및 아티팩트

- headline과 web_title이 다른 문서: 333건. headline을 우선했으며 두 원본 열은 유지했다.
- 원래 문자열이 완전히 같은 제목·설명: 0건.
- HTML/엔티티/공백 정리 후 정확히 같은 제목·설명: 0건.
- 명백한 near-duplicate 후보: 0건. token sequence similarity ≥0.90, 양쪽 5토큰 이상, 길이 비율 ≥0.80; 검출만 하고 제거하지 않았다.
- HTML 포함 문서: 135건. 엔티티 포함: 88건. URL 포함: 0건.
- 빈 text_sentiment: 0건. 빈 text_clean: 0건.
후보·정리 전후 사례는 headline_description_duplicates.csv와 html_web_artifact_examples.csv에 저장했다.

## 토큰 수

| 전략 | 기사 수 | raw 토큰 | stopword 제거 직전 | clean 토큰 | clean 어휘 수 |
|---|---:|---:|---:|---:|---:|
| broad_business_control | 5,054 | 155,246 | 154,614 | 112,409 | 12,556 |
| anchors_stock_market_tag | 2,512 | 79,634 | 79,216 | 57,888 | 7,338 |

raw는 HTML/URL을 제거하기 전 text_raw에 같은 Unicode 토큰 패턴을 적용한 수이다.
raw→clean 차이에는 HTML/URL/불용어 제거와 시장 표기 정규화가 함께 포함된다. 전부 불용어 제거 효과라고 해석하지 않는다.
양쪽 전략에 속한 기사는 물리적으로 한 번 저장하되 각 전략의 기술통계에 각각 포함한다. 두 전략의 토큰 수를 합집합 토큰 수로 더하지 않는다.

## 불용어·상용구 후보

기본 목록은 NLTK English stopwords이며 보호 단어를 제외한 정확한 적용 목록을 cleaning_policy.json에 저장했다.
candidate_custom_stopwords.csv에는 실제 빈도·문서 빈도·문서 비율·문맥 예시·applied=False를 저장한다.
as it happened, live updates 등의 문서 빈도도 별도 저장했으며 문구를 자동 삭제하지 않았다.

현재 표시할 후보(빈도 상위 10개씩, **미적용**):
- broad_business_control: happened (2,101), news (915), latest (859), coverage (854), rolling (839), says (631), new (479), year (425), business (389), live (229)
- anchors_stock_market_tag: happened (1,610), news (756), latest (701), coverage (697), rolling (689), new (280), says (258), year (245), business (232), live (153)

## 표제어화 평가

고유 기사 합집합의 clean 어휘 12,884개를 WordNet 명사형으로 비교했다.
보호 단어를 적용한 비교 어휘는 11,679개로 9.35% 감소한다.
바뀔 수 있는 토큰 질량 비중은 16.04%이다.
stocks→stock, losses→loss 같은 통합과 달리, 단순 WordNet 처리에는 us→u, 동사형 fed→feed 같은 약어 손상 가능성이 있다.
어휘 수 감소만으로 의미 보존·해석 개선이 입증되지는 않는다. POS/문맥 검토가 없는 현재는 전역 적용을 보류했다.
전체 mapping과 보호/비보호 risk probes를 CSV로 저장했다. 실제 text_clean은 표제어화하지 않았다.

## 주간 구조

weekly_article_token_counts.csv에는 두 전략 × 전체 경계 포함 513주, 총 1,026행이 있다.
article_count는 원자료의 주별 집계와 일치하며 token_count는 clean 토큰 수, unique_token_count는 해당 주·전략의 어휘 합집합 크기이다.
문서별 unique 수를 합하지 않는다. strict의 0건 주도 0으로 유지한다. 완전한 512주는 is_complete_week=True로 선택할 수 있다.

## 연도별 출력

2017~2026 모든 연도의 broad·strict 빈도표와 개별 워드클라우드 20개, 두 개의 전체 연도 패널을 저장했다.
WordCloud 자체 plural normalization·collocations·추가 stopwords는 끄고 이미 계산한 빈도를 입력했다.
이모지는 토큰/빈도표에 보존한다. 폰트 제약으로 wordcloud 렌더링에서만 제외한 경우 별도 CSV에 기록했다.
bigrams는 같은 문서·같은 제목/설명 필드·같은 문장 안의 원래 인접한 유효 토큰만 센다. 제거된 불용어 사이를 새로 이어 붙이지 않는다.

## 저장 위치

- data/processed/guardian_text/articles/year=YYYY/: 모든 원본 열 + 세 텍스트 + 토큰 + 주 ID의 Parquet.
- results/guardian_text_preprocessing/: raw/clean 상위 100개, 전체 빈도, 연도별 빈도, bigrams, 후보 불용어, lemma 비교, 주간 건수, 점검 표본, 정책·진단 JSON.
- figures/guardian_text_preprocessing/: 연도별 20개 워드클라우드와 broad/strict 패널.

## 검증과 작업 경계

원자료 Parquet SHA-256이 전처리 전후 동일하다. 원본의 모든 열과 한 기사당 한 레코드 원칙을 보존했다.
추가 Guardian 요청, VADER 점수, WSI, FGI, lead-lag 또는 회귀를 계산하지 않았다. 파일럿 relevance 라벨도 가져오지 않았다.
추가 불용어와 표제어화의 채택 여부는 이 결과 검토 후 결정한다. 어휘/워드클라우드의 경제적 해석은 작성하지 않았다.

규칙 근거: [NLTK WordNet lemmatizer](https://www.nltk.org/api/nltk.stem.wordnet.html), [NLTK resource installation](https://www.nltk.org/data.html).
