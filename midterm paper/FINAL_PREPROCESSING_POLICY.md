# 프로젝트 최종 전처리 명세

**FINAL · guardian_text_v2_final**

정책 SHA-256: `2874fd2b32c45097b161ccc62931a899777a0c7b9711d98efd587d7d1b1dff5d`

## 기준과 불용어

- 기본 목록: NLTK English stopwords. 실제 단어 전체와 공개 자원의 URL·SHA-256을 JSON에 고정했다.
- 보호 어휘: above, against, amount, aren't, back, bear, bearish, bears, below, bill, bull, bullish, bulls, bullshit, call, can't, cannot, couldn't, didn't, doesn't, don't, dow, down, fed, fomo, ftse, fuck, fucking, hadn't, hasn't, haven't, hodl, interest, isn't, less, more, mustn't, nasdaq, needn't, neither, never, no, nobody, none, nor, not, nothing, off, out, over, put, rekt, shit, shorts, shouldn't, spx, stonks, system, tendies, uk, under, up, us, vix, wasn't, weren't, without, won't, wouldn't, wsb, yolo
- 추가 불용어(빈도 입력만): coverage, happened, latest, live, news, rolling
- 추가하지 않는 단어: new, year, business, says.
- Lemmatization: **disabled**. Stemming: **disabled**.

## 원문과 감성 입력

headline을 우선하고 없으면 web_title을 사용한다. description과 줄바꿈으로 결합하며 표시상 정확한 중복만 한 번 제외한다.
모든 원본 열과 API JSON을 보존한다. near-duplicate는 표시만 한다.
text_sentiment는 HTML/script/style·엔티티·비언어적 공백만 처리한다. 대소문자·부정·문장부호·강조·욕설·은어·이모지는 보존한다.
감성 입력에는 기본/추가 불용어, URL 삭제, 소문자화, 표제어화를 적용하지 않는다.

## 빈도 입력

text_sentiment에서 NFKC·소문자화·curly apostrophe 표준화·URL 제거를 한 뒤 Unicode 단어/숫자/이모지를 토큰화한다.
U.S.→us, U.K.→uk, S&P 500→s&p_500, FTSE 100/250→ftse_100/250, Nasdaq 100→nasdaq_100을 사용한다.
내부 apostrophe/hyphen, $ticker, 시장 약어와 수치를 일괄 삭제하지 않는다.
효과적인 불용어 집합은 (NLTK 기본 목록 − 보호 어휘) ∪ 승인된 여섯 단어이다.
추가 phrase-level deletion은 없다. WordCloud의 plural normalization·collocations·별도 stopword 처리는 끈다.
토큰화 정규식과 실제 모든 불용어·보호 단어는 JSON 파일에 기록되어 있다.

## 기술통계와 분리

주간은 America/New_York 날짜의 월요일~일요일. 경계 주를 보존하고 is_complete_week로 구분한다.
한 Guardian ID는 물리적으로 하나이며 broad/strict membership flags로 각각 집계한다. 통합 감성지수는 만들지 않는다.
week token_count는 최종 clean 토큰 수, unique_token_count는 주간 어휘 집합의 크기이다.
bigrams는 같은 문서·같은 필드·같은 문장 내 원래 인접한 두 유효 토큰만 센다.

이 명세는 전처리만 동결한다. 감성 모델 실행이나 후속 분석을 승인하는 명세는 아니다.
