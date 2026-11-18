# Guardian VADER 및 주간 WSI 진단

VADER 3.3.2를 확정된 `guardian_text_v2_final`의 `text_sentiment` 전체에 기사별로 한 번씩 적용했다.
전처리 정책, text_raw/text_sentiment/text_clean, 원본 메타데이터와 소속 플래그는 수정하지 않았다.
동일 기사에는 한 점수만 부여하고 두 전략의 소속을 유지한다. 파일럿 relevance 라벨은 사용하지 않았다.

물리적 기사 5,265건; broad 5,054건; strict 2,512건; 중첩 2,301건.

## 기사별 분포

positive: compound >= 0.05; negative: compound <= -0.05; 나머지는 neutral. 수동 보정은 없다.
표준편차는 ddof=1, 왜도는 pandas 보정 표본 왜도, 첨도는 Fisher 초과첨도(정규분포 0)이다. Pearson 첨도도 CSV에 병기했다.

| query_strategy | article_count | mean_compound | median_compound | std_compound | min_compound | max_compound | skewness | excess_kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_business_control | 5054 | -0.0782 | -0.0516 | 0.5192 | -0.9705 | 0.9485 | 0.1008 | -1.1656 |
| anchors_stock_market_tag | 2512 | -0.0932 | -0.0772 | 0.5203 | -0.9705 | 0.9485 | 0.1376 | -1.1461 |

| query_strategy | positive_count | positive_share | neutral_count | neutral_share | negative_count | negative_share |
| --- | --- | --- | --- | --- | --- | --- |
| broad_business_control | 2009 | 0.3975 | 517 | 0.1023 | 2528 | 0.5002 |
| anchors_stock_market_tag | 955 | 0.3802 | 261 | 0.1039 | 1296 | 0.5159 |

## 주간 구조와 두 WSI

America/New_York 월요일~일요일 주를 그대로 사용한다. 513개 경계 포함 주 중 완전한 주는 512개이다.
2016-12-26~2017-01-01 경계 주는 연구기간 내 하루만 포함하는 부분 주이며 기사 0건이다. 삭제하지 않고 표시했다.
기본 기술통계는 complete_weeks 기준이며 all_calendar_weeks 결과도 저장했다.
기사 0건인 주는 count=0, 비율/평균/중앙값/표준편차/WSI/SE/구간은 NA이다. 단일 기사 주의 표준편차도 NA이다.

주 지표: B=(P-N)/(P+N+U), WSI_count=50*(1+B). 강건성 지표: WSI_continuous=50*(1+mean_compound).
두 지표 모두 0~100이다. count 지표에서 50은 긍정·부정 수가 균형이거나 전부 중립인 경우이다.

| query_strategy | metric | calendar_weeks | valid_weeks | missing_weeks | mean | median | std | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_business_control | WSI_count | 512 | 512 | 0 | 45.3951 | 45.0000 | 18.3332 | 0.0000 | 100.0000 |
| broad_business_control | WSI_continuous | 512 | 512 | 0 | 46.3176 | 46.8016 | 10.3776 | 17.0680 | 73.1830 |
| anchors_stock_market_tag | WSI_count | 512 | 504 | 8 | 43.6341 | 41.6667 | 26.3211 | 0.0000 | 100.0000 |
| anchors_stock_market_tag | WSI_continuous | 512 | 504 | 8 | 45.5278 | 45.8105 | 14.7636 | 8.8750 | 88.2500 |

## Broad와 strict의 기술적 비교

두 전략을 하나의 지표로 합치지 않았다. 아래 상관은 두 Guardian WSI 사이에서만 계산했으며 FGI와의 상관이 아니다.
두 표본에 중첩 기사가 있으므로 독립 표본 비교로 해석하지 않는다.

| scope | metric | paired_valid_weeks | broad_strict_pearson_correlation |
| --- | --- | --- | --- |
| complete_weeks | WSI_count | 504 | 0.6826 |
| complete_weeks | WSI_continuous | 504 | 0.7169 |
| all_calendar_weeks | WSI_count | 504 | 0.6826 |
| all_calendar_weeks | WSI_continuous | 504 | 0.7169 |

## 근사 표본 불확실성

SE(B)=sqrt([p_positive+p_negative-(p_positive-p_negative)^2]/n); SE(WSI)=50*SE(B).
구간은 WSI_count ± 1.96*SE(WSI)인 **근사 표본 구간**이다. 요청된 식 그대로 저장해 0~100 밖으로 나갈 수도 있다.
기사 간 독립성은 보장하지 않는다. 동일 사건·반복 설명·live-blog 등으로 실제 불확실성을 과소평가할 수 있다.
적은 기사 수와 한 범주만 관찰된 주에서는 정규근사가 부적절하거나 SE=0이 될 수 있다. 0폭 구간은 확실성의 증거가 아니다.
이 구간은 VADER 오분류, 표본 관련성, 기사 의존성을 포함한 전체 오차구간이 아니다. lt3/lt5 및 degenerate 플래그를 저장했다.

| query_strategy | weeks_lt3_articles | weeks_lt5_articles | degenerate_intervals | intervals_outside_0_100 |
| --- | --- | --- | --- | --- |
| broad_business_control | 0 | 18 | 4 | 115 |
| anchors_stock_market_tag | 71 | 251 | 76 | 288 |

## 수동 오류 검사 표본

고유 기사 56건을 `vader_inspection_sample.csv`로 저장했다.
강한 긍정/부정(±0.75), 중립 근처(|compound|<=0.10), 욕설·슬랭, 방향 단어와 점수의 의심 조합을 고정 시드로 추출했다.
의심 조합은 검사 후보를 찾는 휴리스틱이며 오류 판정이나 정답 라벨이 아니다. 원 점수·분류는 그대로 보존했다.
모든 특이 텍스트 후보는 `unusual_text_candidates.csv`에 추가 보존했다. 각 후보 규칙·전체 수·추출 수는 inventory CSV에 있다.
VADER는 일반 어휘·규칙 기반 모형이다. 금융 약어, 상승 맥락의 욕설, 풍자, 밈과 도메인 표현을 완벽히 이해한다고 가정하지 않는다.

## 결과물과 실행

- 기사 Parquet: `data/processed/guardian_vader/articles/year=YYYY/part-00000.parquet`
- 주간 broad/strict 감성·WSI: `results/guardian_sentiment/weekly_{broad,strict}_{sentiment,wsi}.{csv,parquet}`
- 기사·주간 통계, 클래스 분포, 전략 간 비교, 모델·입출력 해시: `results/guardian_sentiment/`
- 점수 히스토그램·클래스 분포·주간 WSI 두 그림: `figures/guardian_sentiment/`
- 검토 노트북: `guardian_sentiment_review.ipynb`

```powershell
python guardian_sentiment_analysis.py
python verify_guardian_sentiment.py
python build_guardian_sentiment_notebook.py --execute
```

원자료와 정책 해시, 모든 기존 열과 점수·분류, 집계·수식·결측·표본 보존을 별도 검증 스크립트로 확인한다.
FGI를 읽거나 병합하지 않았고 WSI-FGI 상관, 시차, 시장 수익률, 회귀 및 예측은 수행하지 않았다. 경제적 의미도 해석하지 않았다.

모형 및 임계값 근거: [VADER 공식 저장소](https://github.com/cjhutto/vaderSentiment/blob/master/README.rst).
