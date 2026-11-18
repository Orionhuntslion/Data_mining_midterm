"""최종 전처리의 정책 명세와 변경 전후 진단 문서. 경제적 해석/감성 분석 없음."""
import json


def write_review(root,policy,comparison,entrants,phrases,templates,validation):
    lines=["# 최종 전처리 검토 · 정책 동결", "",
        f"정책 버전: **{policy['version']}**. 물리적 기사 **{validation['physical_articles']:,}건**.",
        "text_raw·text_sentiment와 모든 원본 메타데이터/전략 플래그를 보존하고 text_clean 계열에서 승인된 여섯 단어만 제거했다.",
        "추가 불용어: **happened, latest, coverage, rolling, live, news**.",
        "**new, year, business, says는 추가하지 않았고 기존 출현 횟수도 그대로 보존했다. 표제어화는 비활성화했다.**", "",
        "## 토큰 변경 전후", "", "| 전략 | clean 토큰 전 | 후 | 제거 | 감소율 | 고유 어휘 전 | 후 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in comparison:
        lines.append(f"| {r['query_strategy']} | {r['clean_tokens_before']:,} | {r['clean_tokens_after']:,} | {r['tokens_removed']:,} | {r['removed_pct']:.2f}% | {r['unique_tokens_before']:,} | {r['unique_tokens_after']:,} |")
    lines.extend(["", "단어 빈도만 기준으로 여섯 항목을 제외했으며 나머지 모든 토큰의 출현 횟수는 동일하다.",
        "이 결과는 빈도 입력의 변경 진단이며 실제 감성점수나 감성모델 성능 비교가 아니다.","",
        "## Top 30 신규 진입", ""])
    for strategy in ["broad_business_control","anchors_stock_market_tag"]:
        values=[f"{r['term']} (새 순위 {r['rank']}, {r['count']:,}회)" for r in entrants if r["query_strategy"]==strategy]
        lines.append(f"- {strategy}: "+(", ".join(values) if values else "없음"))
    lines.extend(["", "변경 전 top30_before.csv와 변경 후 top30_after.csv에는 전체 30개씩의 순위와 빈도가 있다.",
        "새 단어를 삽입한 것이 아니라 기존 단어가 상위 30위로 올라온 것이다.","",
        "## 금융 어휘 보존", "",
        "financial_vocabulary_preservation.csv에서 stock/stocks, share/shares, equities, market, Fed/Federal Reserve,",
        "interest/rates, inflation, oil, tariffs/trade, banks, technology, s&p_500, Nasdaq, Dow 등을 확인했다.",
        "원래 존재한 항목은 빈도가 그대로이며 없는 항목은 not_present; not_inserted로 기록했다.",
        "검증은 단순 예시 목록에만 한정하지 않고 모든 비불용어 토큰의 빈도 동일성도 검사했다.","",
        "## 남아 있는 상용구 · 추가 규칙 미적용", "",
        "원문과 text_sentiment는 보존하므로 해당 텍스트에서 상용구의 출현 횟수가 줄어들지는 않는다.",
        "빈도 텍스트에서 as it happened의 happened와 rolling coverage 두 토큰은 제거되었다.",
        "live updates에서는 live만 제거하며 updates는 추가 불용어로 승인되지 않아 남을 수 있다.",
        "반복 설명 템플릿에 포함된 경제 어휘도 자동 삭제하지 않았다.","",
        "| 전략 | 같은 설명 템플릿 기사 수 | 삭제 후 남는 텍스트 |", "|---|---:|---|"])
    for row in templates:
        if row["description_visible"].lower()=="rolling coverage of the latest economic and financial news":
            lines.append(f"| {row['query_strategy']} | {row['identical_description_articles']:,} | {row['text_clean_after']} |")
    lines.extend(["", "이 템플릿의 economic/financial 빈도 잔류는 알려진 한계로 보고한다. 추가 단어/문구 삭제나 템플릿 제거는 하지 않았다.",
        "bigrams는 원래 인접한 단어 중 둘 다 허용된 경우만 계산하므로 제거한 단어를 건너뛴 새 bigram은 만들지 않는다.",
        "상용구별 원문 출현 수, 살아남는 토큰, 잔류 문서 수는 phrase_artifact_review.csv에 있다.",
        "반복 설명의 원문과 전후 텍스트는 repeated_description_templates.csv에 있다.","",
        "## 최종 산출물", "",
        "- data/processed/guardian_text_final/articles/year=YYYY/: 최종 Parquet; 원본 열 + 기존 text_raw/text_sentiment + 최종 text_clean.",
        "- data/processed/guardian_text/articles/: 이전 v1 baseline. 덮어쓰지 않았다.",
        "- results/guardian_text_final/: 전체/연도별 빈도, top100, bigrams, 대표 연도 표, 전후 비교, 주간 기술통계, 어휘 보존 점검.",
        "- figures/guardian_text_final/: 연도별 20개, 전체 연도 패널 2개, broad/strict 전체 기간 2개, 대표 연도 패널 1개.",
        "- 대표 연도는 2017·2020·2026이며 모든 연도 결과도 보존했다. 2026은 10월 25일까지의 부분 연도다.",
        "- FINAL_PREPROCESSING_POLICY.json / .md: 최종 명세. 변경하려면 새 버전의 명시적 정책 수정이 필요하다.","",
        "## 검증과 중단 지점", "",
        "5,265개 ID와 원본/이전 Parquet 해시, text_raw/text_sentiment 불변, 전략 소속, 비불용어 빈도 불변을 검증했다.",
        "감성분석에 대한 불용어 적용은 없고, 표제어화·추가 문구 삭제도 없다.",
        "VADER·WSI·FGI·시차·수익률 분석은 시작하지 않았다. 최종 어휘와 워드클라우드의 사용자 검토를 기다린다."])
    (root/"FINAL_PREPROCESSING_REVIEW.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    spec=["# 프로젝트 최종 전처리 명세", "", f"**FINAL · {policy['version']}**", "",
        f"정책 SHA-256: `{policy['policy_sha256']}`", "",
        "## 기준과 불용어", "",
        "- 기본 목록: NLTK English stopwords. 실제 단어 전체와 공개 자원의 URL·SHA-256을 JSON에 고정했다.",
        "- 보호 어휘: "+", ".join(policy["protected_terms"]),
        "- 추가 불용어(빈도 입력만): "+", ".join(policy["custom_stopwords_applied"]),
        "- 추가하지 않는 단어: new, year, business, says.",
        "- Lemmatization: **disabled**. Stemming: **disabled**.","",
        "## 원문과 감성 입력", "",
        "headline을 우선하고 없으면 web_title을 사용한다. description과 줄바꿈으로 결합하며 표시상 정확한 중복만 한 번 제외한다.",
        "모든 원본 열과 API JSON을 보존한다. near-duplicate는 표시만 한다.",
        "text_sentiment는 HTML/script/style·엔티티·비언어적 공백만 처리한다. 대소문자·부정·문장부호·강조·욕설·은어·이모지는 보존한다.",
        "감성 입력에는 기본/추가 불용어, URL 삭제, 소문자화, 표제어화를 적용하지 않는다.","",
        "## 빈도 입력", "",
        "text_sentiment에서 NFKC·소문자화·curly apostrophe 표준화·URL 제거를 한 뒤 Unicode 단어/숫자/이모지를 토큰화한다.",
        "U.S.→us, U.K.→uk, S&P 500→s&p_500, FTSE 100/250→ftse_100/250, Nasdaq 100→nasdaq_100을 사용한다.",
        "내부 apostrophe/hyphen, $ticker, 시장 약어와 수치를 일괄 삭제하지 않는다.",
        "효과적인 불용어 집합은 (NLTK 기본 목록 − 보호 어휘) ∪ 승인된 여섯 단어이다.",
        "추가 phrase-level deletion은 없다. WordCloud의 plural normalization·collocations·별도 stopword 처리는 끈다.",
        "토큰화 정규식과 실제 모든 불용어·보호 단어는 JSON 파일에 기록되어 있다.","",
        "## 기술통계와 분리", "",
        "주간은 America/New_York 날짜의 월요일~일요일. 경계 주를 보존하고 is_complete_week로 구분한다.",
        "한 Guardian ID는 물리적으로 하나이며 broad/strict membership flags로 각각 집계한다. 통합 감성지수는 만들지 않는다.",
        "week token_count는 최종 clean 토큰 수, unique_token_count는 주간 어휘 집합의 크기이다.",
        "bigrams는 같은 문서·같은 필드·같은 문장 내 원래 인접한 두 유효 토큰만 센다.","",
        "이 명세는 전처리만 동결한다. 감성 모델 실행이나 후속 분석을 승인하는 명세는 아니다."]
    (root/"FINAL_PREPROCESSING_POLICY.md").write_text("\n".join(spec)+"\n",encoding="utf-8")
