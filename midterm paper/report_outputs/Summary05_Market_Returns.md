# 최종 탐색 분석 · 기술 인계

## 데이터 출처

- FRED `SP500`, 원출처 S&P Dow Jones Indices LLC: [자료 설명](https://fred.stlouisfed.org/series/SP500).
- 다운로드: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500; 원응답 SHA-256 `269b80fef05c69061cb7f9c42ac6f12c03b3a1074e21da3c64b501036c7a8193`.
- 일별 종가 가격지수. 배당 포함 총수익지수가 아니며 별도 adjusted close 필드는 없다. 임의 수정종가를 만들지 않았다.
- FRED 기본 공개 CSV를 보관하고 2017-01-01~2026-10-25를 로컬 선택했다. 다른 가격 출처를 이어 붙이지 않았다.

## 방법·수식

- 뉴욕 월요일~일요일 달력 유지. 분석은 완전한 주만 사용하며, 부분 경계 주는 파일에 유지한다.
- P_w = 주 안의 마지막 유효 거래일 종가; r_w = ln(P_w/P_(w-1)). 단위는 소수 로그수익률이다.
- 직전 **달력 주** 종가가 없으면 수익률은 NA. 기간 밖 초기 가격·보간·전방 채움은 사용하지 않는다.
- 동시점: r_w = alpha + beta*x_w + epsilon_w. 탐색적 다음 주: r_(w+1) = alpha + beta*x_w + epsilon_w.
- 다음 주는 정확한 week_start+7일로 연결하고 그 후 결측 쌍을 제외한다. 각 적합은 설명변수 하나와 절편만 사용한다.
- x는 broad/strict count·continuous WSI 또는 맥락 비교 FGI_weekly_mean. 기존 값·생성 규칙은 변경하지 않는다.
- SE(beta)=sqrt([SSE/(n-2)]/Sxx); t=beta/SE(beta); p=2*t_df=n-2.sf(|t|). 모두 conventional OLS 기준이다.
- R²=1-SSE/SST; adjusted R²=1-(1-R²)*(n-1)/(n-2). beta는 지수 1포인트당 소수 로그수익률이다.
- Pearson 및 동률 평균 순위 Spearman을 각 명세의 유효 쌍에서 계산한다. CSV round_trip 정밀도를 사용한다.
- 그림만 수익률에 100을 곱한다. 시계열 WSI z=(x-mean)/std(ddof=1), 동시점 유효 쌍 표본 기준이다.

## 사용 패키지

- requests 2.34.2; pandas 2.3.3; numpy 2.2.5; pyarrow 25.0.1; scipy 1.15.3; matplotlib 3.10.9; exchange_calendars 4.13.2.
- 노트북: nbformat, nbclient, ipykernel, jupyter_client, IPython. 검증 이미지 읽기: Pillow.
- 표준 라이브러리: pathlib, hashlib, json, datetime, sys, importlib.metadata; 검증용 statistics/math.
- 통계 함수 근거: [SciPy 단순 선형 적합](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html).

## 표본 크기·자료 진단

- 유효 일별 종가 2,466개; 2017-01-03~2026-10-23.
- 기간 내 날짜 중복 0; 유효 가격 자료의 결측 0.
- 원응답의 기간 내 행 2,560개 중 가격 공란 94개 (비거래일 공란 94). 예상 XNYS 거래일 2,466일; 누락 거래일 가격 0개.
- 달력 513주, 완전한 주 512주. 첫 완전 주는 기간 내 직전 주 종가가 없어 수익률이 결측이다.

| predictor | relationship | valid_paired_weeks | predictor_only_weeks | return_only_weeks | first_matched_predictor_week | last_matched_predictor_week |
| --- | --- | --- | --- | --- | --- | --- |
| broad_WSI_count | contemporaneous | 511 | 1 | 0 | 2017-01-09 | 2026-10-19 |
| broad_WSI_count | exploratory_next_week | 511 | 1 | 0 | 2017-01-02 | 2026-10-12 |
| broad_WSI_continuous | contemporaneous | 511 | 1 | 0 | 2017-01-09 | 2026-10-19 |
| broad_WSI_continuous | exploratory_next_week | 511 | 1 | 0 | 2017-01-02 | 2026-10-12 |
| strict_WSI_count | contemporaneous | 503 | 1 | 8 | 2017-01-09 | 2026-10-19 |
| strict_WSI_count | exploratory_next_week | 503 | 1 | 8 | 2017-01-02 | 2026-10-12 |
| strict_WSI_continuous | contemporaneous | 503 | 1 | 8 | 2017-01-09 | 2026-10-19 |
| strict_WSI_continuous | exploratory_next_week | 503 | 1 | 8 | 2017-01-02 | 2026-10-12 |
| FGI_weekly_mean | contemporaneous | 507 | 1 | 4 | 2017-01-09 | 2026-10-19 |
| FGI_weekly_mean | exploratory_next_week | 507 | 1 | 4 | 2017-01-02 | 2026-10-12 |

## 수치 결과

| predictor | relationship | n_pairs | pearson | spearman |
| --- | --- | --- | --- | --- |
| broad_WSI_count | contemporaneous | 511 | 0.1773379 | 0.1466998 |
| broad_WSI_count | exploratory_next_week | 511 | -0.01093186 | -0.04889375 |
| broad_WSI_continuous | contemporaneous | 511 | 0.2083721 | 0.1788712 |
| broad_WSI_continuous | exploratory_next_week | 511 | -0.01055181 | -0.03796855 |
| strict_WSI_count | contemporaneous | 503 | 0.18355 | 0.1724186 |
| strict_WSI_count | exploratory_next_week | 503 | -0.009004833 | -0.03354711 |
| strict_WSI_continuous | contemporaneous | 503 | 0.1997655 | 0.1961231 |
| strict_WSI_continuous | exploratory_next_week | 503 | -0.0208366 | -0.04295452 |
| FGI_weekly_mean | contemporaneous | 507 | 0.1880219 | 0.1709214 |
| FGI_weekly_mean | exploratory_next_week | 507 | -0.0428334 | -0.06591152 |

| predictor | relationship | n_observations | alpha | beta | se_beta | t_beta | p_value_beta_two_sided | R_squared | adjusted_R_squared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_WSI_count | contemporaneous | 511 | -0.008185427 | 0.0002327561 | 5.725349e-05 | 4.065362 | 5.555081e-05 | 0.03144874 | 0.02954589 |
| broad_WSI_count | exploratory_next_week | 511 | 0.003014294 | -1.430177e-05 | 5.798439e-05 | -0.2466487 | 0.8052796 | 0.0001195055 | -0.001844896 |
| broad_WSI_continuous | contemporaneous | 511 | -0.02001297 | 0.0004835859 | 0.0001006088 | 4.806595 | 2.023844e-06 | 0.04341893 | 0.0415396 |
| broad_WSI_continuous | exploratory_next_week | 511 | 0.003495647 | -2.44045e-05 | 0.0001025085 | -0.238073 | 0.8119203 | 0.0001113407 | -0.001853077 |
| strict_WSI_count | contemporaneous | 503 | -0.005005055 | 0.0001684031 | 4.029349e-05 | 4.179412 | 3.449004e-05 | 0.03369061 | 0.03176185 |
| strict_WSI_count | exploratory_next_week | 503 | 0.002663217 | -8.225117e-06 | 4.080656e-05 | -0.2015636 | 0.8403397 | 8.108702e-05 | -0.001914759 |
| strict_WSI_continuous | contemporaneous | 503 | -0.0125413 | 0.0003270241 | 7.166338e-05 | 4.563336 | 6.340341e-06 | 0.03990624 | 0.03798989 |
| strict_WSI_continuous | exploratory_next_week | 503 | 0.003848683 | -3.392433e-05 | 7.27229e-05 | -0.4664876 | 0.6410693 | 0.0004341641 | -0.001560977 |
| FGI_weekly_mean | contemporaneous | 507 | -0.008868477 | 0.0002318594 | 5.389581e-05 | 4.301993 | 2.032971e-05 | 0.03535223 | 0.03344204 |
| FGI_weekly_mean | exploratory_next_week | 507 | 0.004881556 | -5.306513e-05 | 5.507851e-05 | -0.9634452 | 0.3357852 | 0.0018347 | -0.0001418646 |

### Broad·FGI 비교표

| predictor | relationship | n_pairs | pearson | spearman | beta | p_value_beta_two_sided | R_squared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| broad_WSI_count | contemporaneous | 511 | 0.1773379 | 0.1466998 | 0.0002327561 | 5.555081e-05 | 0.03144874 |
| broad_WSI_count | exploratory_next_week | 511 | -0.01093186 | -0.04889375 | -1.430177e-05 | 0.8052796 | 0.0001195055 |
| broad_WSI_continuous | contemporaneous | 511 | 0.2083721 | 0.1788712 | 0.0004835859 | 2.023844e-06 | 0.04341893 |
| broad_WSI_continuous | exploratory_next_week | 511 | -0.01055181 | -0.03796855 | -2.44045e-05 | 0.8119203 | 0.0001113407 |
| FGI_weekly_mean | contemporaneous | 507 | 0.1880219 | 0.1709214 | 0.0002318594 | 2.032971e-05 | 0.03535223 |
| FGI_weekly_mean | exploratory_next_week | 507 | -0.0428334 | -0.06591152 | -5.306513e-05 | 0.3357852 | 0.0018347 |

## 생성 파일

- `data/raw/market/fred_sp500_2017-01-01_2026-10-25.csv`, `.metadata.json`, `source_access_log.json`: 원응답·출처·접근 기록.
- `results/guardian_market_analysis/sp500_daily_validated.{csv,parquet}`: 유효 일별 종가.
- `results/guardian_market_analysis/sp500_weekly_returns.{csv,parquet}`: 주간 종가·로그수익률·관측 수.
- `results/guardian_market_analysis/merged_market_WSI_weekly.{csv,parquet}`: 분리된 WSI 열·FGI·동시/다음 주 수익률.
- `market_correlations.csv`, `market_OLS_results.csv`, `market_benchmark_comparison.csv`: 상관·OLS·맥락 비교.
- `market_merge_diagnostics.csv`, `market_pair_status.csv`: 유효/비매칭 표본과 날짜.
- `market_daily_audit.json`, `market_calendar_audit.csv`, `market_missing_source_rows.csv`, `market_missing_XNYS_sessions.csv`: 자료 진단.
- `market_timeseries_plot_data.csv`, `market_plot_standardization.csv`: 그림 자료와 표준화 기준.
- `market_method.json`, `validation_summary.json`, `independent_validation.json`: 방법·패키지·입력 해시·독립 검증.
- 위 확장자 없는 결과 파일명 기준 폴더: `results/guardian_market_analysis/`.
- `figures/guardian_market_analysis/market_broad_count_scatter.png`.
- `figures/guardian_market_analysis/market_broad_count_scatter.svg`.
- `figures/guardian_market_analysis/market_weekly_returns_wsi.png`.
- `figures/guardian_market_analysis/market_weekly_returns_wsi.svg`.
- `guardian_market_analysis_review.ipynb`: 실행된 저장 결과 검토 노트북.
- 코드: `guardian_market_data.py`, `guardian_market_analysis.py`, `guardian_market_stats.py`, `guardian_market_outputs.py`, `verify_guardian_market_analysis.py`, `build_guardian_market_notebook.py`.

## 한계·주의

- 탐색적 동시점/다음 주 연관성만 계산했다. 인과·예측 능력·거래 수익성·시장 효율성 위반의 증거가 아니다.
- Conventional 표준오차·p값은 자기상관·이분산에 강건하지 않다. 여러 명세를 비교했으며 다중검정 보정은 하지 않았다.
- 통계적 유의성과 경제적 중요성은 같지 않다. 작은 R²를 예측력으로 해석하지 않는다.
- WSI 측정오차·기사 밀도 차이·두 전략의 중첩이 있다. FGI는 정답이 아니며 기존 자료 계보·누락을 유지했다.
- 모델마다 결측 제거 후 표본 날짜·수가 다를 수 있다. 위 비교표는 공통 표본으로 강제하지 않았으며 n을 병기했다.
- 가격지수 수익률은 배당 재투자를 포함하지 않는다. 원자료 재배포 조건은 FRED 출처 페이지를 확인한다.
- 전처리·VADER·WSI·FGI 자료와 이전 결과를 수정하지 않았다. 예측·ML·다변량·추가 모형은 수행하지 않았다.
