# 최종 제출 노트북 점검

## 노트북 섹션 목록

- 0. Environment and Project Setup
- 1. Guardian Data · Data Collection Reference
- 2. Pilot Retrieval Strategy
- 3. Text Preprocessing · Frozen Specification
- 4. VADER Sentiment Analysis
- 5. Weekly Web Sentiment Index
- 6. Fear & Greed Index Validation
- 7. Lead-Lag Analysis · Pearson Only
- 8. S&P 500 Market Analysis
- 9. Final Results Summary
- 10. Validation Checks

## 사용한 입력 파일

- `data/raw/news/guardian_full/articles/year=YYYY/part-00000.parquet` (2017–2026, 10개).
- `data/processed/guardian_text_final/articles/year=YYYY/part-00000.parquet` (확정 텍스트 대조, 10개).
- `data/processed/guardian_vader/articles/year=YYYY/part-00000.parquet` (확정 점수 대조, 10개).
- `FINAL_PREPROCESSING_POLICY.json`, `guardian_text_rules.py`, `.nltk_data/corpora/stopwords.zip`.
- `guardian_api_config.py`, `guardian_full_collection.py` (쿼리 설정/수집 코드 참조만), `guardian_weekly_sentiment.py`.
- `results/guardian_api_pilot/strategy_yield_comparison.csv`.
- `results/guardian_text_final/full_period_word_frequencies.csv`, `yearly_word_frequencies.csv`, `weekly_article_token_counts.csv`.
- `results/guardian_sentiment/article_descriptive_statistics.csv`, `weekly_descriptive_statistics.csv`, `weekly_broad_wsi.csv`, `weekly_strict_wsi.csv`.
- `data/external/fear_greed_2017-01-01_2026-10-25.csv`.
- `results/guardian_fgi_validation/fgi_weekly_validated.csv`, `contemporaneous_correlations.csv`, `lead_lag_correlations.csv`, `first_difference_robustness.csv`, `scatter_visual_fit.json` (필요한 Pearson 열만 읽음).
- `data/raw/market/fred_sp500_2017-01-01_2026-10-25.csv`.
- `results/guardian_market_analysis/sp500_weekly_returns.csv`, `market_OLS_results.csv`, `market_correlations.csv`.
- `figures/guardian_text_final/wordcloud_broad_{2017,2020,2026}.png`.
- `figures/guardian_sentiment/sentiment_class_distribution.png`, `weekly_wsi_count.png`.
- `figures/guardian_fgi_validation/wsi_fgi_overlay_0_100.png`, `wsi_fgi_scatter.png`.
- `figures/guardian_market_analysis/market_broad_count_scatter.png`.

## 셀 실행 여부

- 전체 25개 셀; 코드 13개 모두 실행 성공; 오류 출력 0개.
- `20210673_KimDaeho_Midterm_Final.ipynb` 자체가 출력이 포함된 실행 완료본이다. 별도 중복 노트북 없음.
- HTML: 생성 완료: 20210673_KimDaeho_Midterm_Final.html.
- 자동 API 수집·가격/FGI 다운로드·패키지 설치 없음.

## 검증 결과

- 최종 검증 21개 모두 PASS. 중간 단계의 저장 결과 대조도 모두 통과.
- Guardian 5,265; broad 5,054; strict 2,512; 중첩 2,301.
- 최종 clean 토큰/어휘: broad 106,612/12,550; strict 53,282/7,332.
- 감성 분류 건수 및 4개 VADER 점수: 전 기사 저장 값과 일치.
- 완전한 주 512; strict 결측 8주; FGI 2,443개·완전한 주 중 관측 없는 주 4개.
- WSI–FGI Pearson 0.2248/0.2286/0.1419/0.1511; broad count R² 0.0505.
- 최고 절대상관 시차: broad 0주, strict +1주; broad 차분 Pearson 0.1103/0.1301.
- 시장 일별 종가 2,466개·주간 수익률 511개. 주 명세 동시점 Pearson 0.1773·R² 0.0314, 다음 주 Pearson −0.0109·R² 약 0.0001.
- 모든 실제 입력 파일의 실행 전후 SHA-256 일치. 기존 결과·정책·원자료 불변.

## 남은 재현성 한계

- 노트북 단독 파일만으로는 재실행할 수 없다. 위 상대 경로의 로컬 원자료·확정 결과·규칙 모듈과 NLTK stopwords ZIP이 필요하다.
- 새 환경에서는 0번 섹션의 패키지 버전에 맞춰 별도 설치해야 한다. 노트북이 설치·다운로드를 자동 수행하지 않는다.
- FGI는 보관된 제3자 자료 스냅샷이며 계보 차이와 누락 주를 유지한다. 시장 자료는 배당 미포함 가격지수이다.
- 기존 두 패널 WSI·감성 분포 그림은 Broad와 Strict를 함께 포함한다. 시차 그림은 기존 Pearson 수치만 노트북 안에서 표시했다.
- Conventional OLS p값은 자기상관·이분산에 강건하지 않다. 새 추정·설계 변경 없이 기존 단변량 결과만 재현했다.
