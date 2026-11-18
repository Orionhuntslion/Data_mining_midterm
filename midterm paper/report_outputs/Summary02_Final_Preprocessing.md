# 최종 전처리 검토 · 정책 동결

정책 버전: **guardian_text_v2_final**. 물리적 기사 **5,265건**.
text_raw·text_sentiment와 모든 원본 메타데이터/전략 플래그를 보존하고 text_clean 계열에서 승인된 여섯 단어만 제거했다.
추가 불용어: **happened, latest, coverage, rolling, live, news**.
**new, year, business, says는 추가하지 않았고 기존 출현 횟수도 그대로 보존했다. 표제어화는 비활성화했다.**

## 토큰 변경 전후

| 전략 | clean 토큰 전 | 후 | 제거 | 감소율 | 고유 어휘 전 | 후 |
|---|---:|---:|---:|---:|---:|---:|
| broad_business_control | 112,409 | 106,612 | 5,797 | 5.16% | 12,556 | 12,550 |
| anchors_stock_market_tag | 57,888 | 53,282 | 4,606 | 7.96% | 7,338 | 7,332 |

단어 빈도만 기준으로 여섯 항목을 제외했으며 나머지 모든 토큰의 출현 횟수는 동일하다.
이 결과는 빈도 입력의 변경 진단이며 실제 감성점수나 감성모델 성능 비교가 아니다.

## Top 30 신규 진입

- broad_business_control: hit (새 순위 26, 414회), global (새 순위 27, 402회), oil (새 순위 28, 402회), up (새 순위 29, 400회), record (새 순위 30, 394회)
- anchors_stock_market_tag: says (새 순위 26, 258회), over (새 순위 27, 252회), rate (새 순위 28, 251회), year (새 순위 29, 245회), oil (새 순위 30, 240회)

변경 전 top30_before.csv와 변경 후 top30_after.csv에는 전체 30개씩의 순위와 빈도가 있다.
새 단어를 삽입한 것이 아니라 기존 단어가 상위 30위로 올라온 것이다.

## 금융 어휘 보존

financial_vocabulary_preservation.csv에서 stock/stocks, share/shares, equities, market, Fed/Federal Reserve,
interest/rates, inflation, oil, tariffs/trade, banks, technology, s&p_500, Nasdaq, Dow 등을 확인했다.
원래 존재한 항목은 빈도가 그대로이며 없는 항목은 not_present; not_inserted로 기록했다.
검증은 단순 예시 목록에만 한정하지 않고 모든 비불용어 토큰의 빈도 동일성도 검사했다.

## 남아 있는 상용구 · 추가 규칙 미적용

원문과 text_sentiment는 보존하므로 해당 텍스트에서 상용구의 출현 횟수가 줄어들지는 않는다.
빈도 텍스트에서 as it happened의 happened와 rolling coverage 두 토큰은 제거되었다.
live updates에서는 live만 제거하며 updates는 추가 불용어로 승인되지 않아 남을 수 있다.
반복 설명 템플릿에 포함된 경제 어휘도 자동 삭제하지 않았다.

| 전략 | 같은 설명 템플릿 기사 수 | 삭제 후 남는 텍스트 |
|---|---:|---|
| broad_business_control | 418 | economic financial |
| anchors_stock_market_tag | 381 | economic financial |

이 템플릿의 economic/financial 빈도 잔류는 알려진 한계로 보고한다. 추가 단어/문구 삭제나 템플릿 제거는 하지 않았다.
bigrams는 원래 인접한 단어 중 둘 다 허용된 경우만 계산하므로 제거한 단어를 건너뛴 새 bigram은 만들지 않는다.
상용구별 원문 출현 수, 살아남는 토큰, 잔류 문서 수는 phrase_artifact_review.csv에 있다.
반복 설명의 원문과 전후 텍스트는 repeated_description_templates.csv에 있다.

## 최종 산출물

- data/processed/guardian_text_final/articles/year=YYYY/: 최종 Parquet; 원본 열 + 기존 text_raw/text_sentiment + 최종 text_clean.
- data/processed/guardian_text/articles/: 이전 v1 baseline. 덮어쓰지 않았다.
- results/guardian_text_final/: 전체/연도별 빈도, top100, bigrams, 대표 연도 표, 전후 비교, 주간 기술통계, 어휘 보존 점검.
- figures/guardian_text_final/: 연도별 20개, 전체 연도 패널 2개, broad/strict 전체 기간 2개, 대표 연도 패널 1개.
- 대표 연도는 2017·2020·2026이며 모든 연도 결과도 보존했다. 2026은 10월 25일까지의 부분 연도다.
- FINAL_PREPROCESSING_POLICY.json / .md: 최종 명세. 변경하려면 새 버전의 명시적 정책 수정이 필요하다.

## 검증과 중단 지점

5,265개 ID와 원본/이전 Parquet 해시, text_raw/text_sentiment 불변, 전략 소속, 비불용어 빈도 불변을 검증했다.
감성분석에 대한 불용어 적용은 없고, 표제어화·추가 문구 삭제도 없다.
VADER·WSI·FGI·시차·수익률 분석은 시작하지 않았다. 최종 어휘와 워드클라우드의 사용자 검토를 기다린다.
