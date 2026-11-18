| 파일명 | 내용 | 원본 분석 단계 | 권장 위치 |
| --- | --- | --- | --- |
| Fig01_WordCloud_2017.png | 최종 text_clean의 Broad 2017년 워드클라우드 | 최종 전처리 · guardian_text_v2_final | 본문 |
| Fig02_WordCloud_2020.png | 최종 text_clean의 Broad 2020년 워드클라우드 | 최종 전처리 · guardian_text_v2_final | 본문 |
| Fig03_WordCloud_2026.png | 최종 text_clean의 Broad 2026년 워드클라우드; 연구기간 종료일 10월 25일 | 최종 전처리 · guardian_text_v2_final | 본문 |
| Fig04_Sentiment_Distribution.png | Broad·Strict의 VADER 긍정·중립·부정 기사 수와 비율; 기존 두 패널 유지 | VADER / 주간 WSI | 본문 |
| Fig05_Weekly_WSI.png | Broad 위쪽·Strict 아래쪽의 주간 count WSI와 근사 표본 구간; 결측 유지 | VADER / 주간 WSI | 본문 |
| Fig06_WSI_FGI_Overlay.png | Broad count WSI와 주간 평균 FGI의 0–100 중첩 시계열 | WSI–FGI 검증 | 본문 |
| Fig07_WSI_FGI_Scatter.png | Broad count WSI와 주간 평균 FGI 산점도·동시적 직선·Pearson r·R² | WSI–FGI 검증 | 본문 |
| Fig08_LeadLag.png | Broad count WSI와 주간 평균 FGI의 −4~+4주 Pearson·Spearman 상관; Corr(WSI_w, FGI_(w+k)) | WSI–FGI 시차 분석 | 본문 |
| Fig09_WSI_SP500_Scatter.png | Broad count WSI와 S&P 500 주간 로그수익률 산점도·OLS 직선·Pearson r·R² | 최종 시장수익률 탐색 분석 | 본문 |
| Fig09_WSI_SP500_Scatter.svg | Fig09와 동일한 기존 SVG 벡터 원본 | 최종 시장수익률 탐색 분석 | 부록 |
| Fig10_WSI_SP500_TimeSeries.png | S&P 500 주간 로그수익률과 표준화 Broad count WSI의 두 패널 시계열 | 최종 시장수익률 탐색 분석 | 본문 |
| Fig10_WSI_SP500_TimeSeries.svg | Fig10과 동일한 기존 SVG 벡터 원본 | 최종 시장수익률 탐색 분석 | 부록 |
| Table01_Filter_Strategy.csv | 전략별 파일럿 precision·기사 수·조건부 관련 기사 수 비교; 파일럿 추정치와 기존 not_finalized 표기 유지 | Guardian 필터 전략 파일럿 / 수율 비교 | 부록 |
| Table01a_Final_Collection_Counts.csv | 최종 선택 Broad·Strict의 실제 API 수집 건수·기간 내 고유 기사 수; UTC 버퍼 구분 | Guardian 전체 원자료 수집 | 본문 |
| Table02_Sentiment_Summary.csv | 전략별 compound 기술통계·왜도·첨도와 감성 클래스별 수·비율 | VADER / 주간 WSI | 본문 |
| Table03_WSI_Summary.csv | 전략별 count·continuous WSI 기술통계; complete_weeks와 경계 포함 범위 모두 유지 | VADER / 주간 WSI | 본문; 기본 비교는 complete_weeks |
| Table04_WSI_FGI_Correlation.csv | Broad·Strict count·continuous와 FGI 주간 평균·마지막 값의 동시적 Pearson·Spearman 상관 | WSI–FGI 검증 | 본문 |
| Table05_LeadLag.csv | 네 WSI 명세와 주간 평균 FGI의 −4~+4주 유효 쌍 수·Pearson·Spearman 상관 | WSI–FGI 시차 분석 | 부록 |
| Table06_FirstDifference.csv | 네 WSI 명세와 주간 평균 FGI의 1차 차분 동시점(k=0)·시차 상관 | WSI–FGI 차분 강건성 | 부록 |
| Table07_Market_Return_OLS.csv | WSI 네 명세·FGI의 동시점 및 탐색적 다음 주 OLS; alpha·beta·SE·t·p·R²·수정 R²·n | 최종 시장수익률 탐색 분석 | 부록 |
| Table07a_Market_Return_Correlation.csv | WSI 네 명세·FGI와 동시점·다음 주 수익률의 Pearson·Spearman 상관 및 n | 최종 시장수익률 탐색 분석 | 부록 |
| Table07b_Market_Benchmark_Comparison.csv | Broad count·continuous와 FGI의 동시점·다음 주 상관·beta·p·R²·n 비교 | 최종 시장수익률 탐색 분석 | 본문 |
| Summary01_Guardian_Collection.md | 전체 Guardian 원자료 수집·중복 제거·자료 품질 진단의 최종 요약 | Guardian 전체 원자료 수집 | 부록 |
| Summary02_Final_Preprocessing.md | 승인된 여섯 불용어 적용 전후·금융 어휘 보존·잔여 상용구·최종 정책 동결 요약 | 최종 전처리 · guardian_text_v2_final | 부록 |
| Summary03_VADER_WSI.md | VADER 분포·Broad/Strict WSI·근사 표본 구간·검사 표본의 최종 진단 | VADER / 주간 WSI | 부록 |
| Summary04_WSI_FGI_Validation.md | FGI 날짜·누락·병합·동시적 상관·시차·차분의 최종 진단 | WSI–FGI 검증 / 시차 분석 | 부록 |
| Summary05_Market_Returns.md | 가격 출처·방법·패키지·표본·수치·파일명·한계만 담은 최종 기술 인계 | 최종 시장수익률 탐색 분석 | 부록 |
| Policy_Final_Preprocessing.md | 프로젝트 최종 전처리 명세의 읽기용 문서 | 최종 전처리 · guardian_text_v2_final | 부록 |
| Policy_Final_Preprocessing.json | 최종 전처리 규칙·불용어·보호 어휘·자원/패키지 버전·정책 SHA-256 | 최종 전처리 · guardian_text_v2_final | 부록 |
| Packages_Collection_Summary.md | 수집·품질 진단 및 노트북 도구의 라이브러리·버전·용도 요약; 수집 단계 기록 | Guardian 수집 / 주간 밀도 진단 | 부록 |
| Packages_Collection_Versions.csv | 수집 단계 패키지 버전·용도·표준 라이브러리 구분의 기존 목록 | Guardian 수집 / 주간 밀도 진단 | 부록 |
| Packages_Final_Analysis.json | 최종 시장분석 방법과 requests·pandas·numpy·pyarrow·scipy·matplotlib·거래일 달력 버전 기록 | 최종 시장수익률 탐색 분석 | 부록 |
