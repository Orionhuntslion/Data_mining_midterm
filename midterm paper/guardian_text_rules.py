"""Guardian 최종 전처리 규칙. 승인된 불용어 6개는 빈도 텍스트에만 적용한다."""
from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import regex
import nltk
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parent
nltk.data.path.insert(0,str(ROOT/".nltk_data"))
from nltk.corpus import stopwords

POLICY_VERSION="guardian_text_v2_final"
PROTECTED={"no","not","nor","never","neither","without","nothing","nobody","none","cannot",
    "can't","won't","don't","doesn't","didn't","isn't","aren't","wasn't","weren't","haven't","hasn't","hadn't",
    "wouldn't","couldn't","shouldn't","mustn't","needn't","us","uk","fed","ftse","nasdaq","dow","spx","vix",
    "up","down","above","below","over","under","against","off","out","back","more","less",
    "call","put","interest","bill","amount","system","bull","bear","bulls","bears","shorts",
    "bullish","bearish","hodl","yolo","fomo","stonks","rekt","tendies","wsb","fucking","shit","fuck","bullshit"}
BASE_STOPWORDS=set(stopwords.words("english"))
ACTIVE_STOPWORDS=BASE_STOPWORDS-PROTECTED
CUSTOM_STOPWORDS=frozenset({"happened","latest","coverage","rolling","live","news"})

TOKEN_PATTERN=regex.compile(
    r"\$?\p{L}[\p{L}\p{M}\p{N}_]*(?:[&.'’\-][\p{L}\p{M}\p{N}_]+)*"
    r"|[$£€]?\d+(?:[.,]\d+)*(?:%|[a-zA-Z]+)?"
    r"|\p{Regional_Indicator}{2}"
    r"|\p{Extended_Pictographic}(?:\uFE0F|\p{Emoji_Modifier}|\u200D\p{Extended_Pictographic})*")
HTML_TAG=re.compile(r"</?[A-Za-z][^>]*>|<!--.*?-->",re.S)
URL=re.compile(r"https?://\S+|www\.\S+",re.I)
EMOJI=regex.compile(r"\p{Extended_Pictographic}|\p{Regional_Indicator}")
BOILERPLATE={
    "as it happened":r"\bas it happened\b","live updates":r"\blive updates\b",
    "sign up":r"\bsign up\b","read more":r"\bread more\b","subscribe":r"\bsubscribe\b",
    "the guardian":r"\bthe guardian\b","follow the latest":r"\bfollow the latest\b"}
CANDIDATES={
    **{w:"possible web/UI boilerplate; inspect context" for w in ["guardian","subscribe","newsletter","click","email","sign","read","follow","updates","live","happened"]},
    **{w:"possible rolling-news template; may be substantive, inspect examples" for w in ["rolling","coverage","news"]},
    **{w:"generic reporting/time term; may be substantive, requires review" for w in ["said","says","say","new","year","years","week","weeks","day","days","latest","business"]}}


def visible_text(text):
    """대문자·부정·욕설·문장부호·이모지를 보존하고 HTML/비언어적 공백만 정리."""
    text=html.unescape(text or "")
    if HTML_TAG.search(text):
        soup=BeautifulSoup(text,"html.parser")
        for node in soup(["script","style","noscript"]):
            node.decompose()
        # block/line-break는 분리하되 inline markup 제거로 문장부호 앞에 공백을 새로 만들지 않는다.
        for node in soup.find_all(["br","p","div","li","ul","ol"]):
            node.insert_before(" ")
            node.insert_after(" ")
        text=soup.get_text()
    text=text.replace("\u200b"," ").replace("\ufeff","").replace("\u00ad","")
    return re.sub(r"\s+"," ",text).strip()


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def frequency_input(sentiment_text):
    text=unicodedata.normalize("NFKC",sentiment_text).lower().replace("’","'").replace("‘","'")
    text=URL.sub(" ",text)
    text=re.sub(r"\bu\.s\.?(?!\w)","us",text)
    text=re.sub(r"\bu\.k\.?(?!\w)","uk",text)
    text=re.sub(r"\bs\s*&\s*p(?:\s*-?\s*(\d{2,4}))?\b",lambda m:"s&p"+("_"+m[1] if m[1] else ""),text)
    text=re.sub(r"\bftse\s+(100|250)\b",r"ftse_\1",text)
    text=re.sub(r"\bnasdaq\s+(100)\b",r"nasdaq_\1",text)
    return text


def clean_tokens(text):
    tokens=tokenize(frequency_input(text))
    return tokens,[t for t in tokens if t not in ACTIVE_STOPWORDS and t not in CUSTOM_STOPWORDS]


def article_texts(record):
    headline=record.get("headline") or record.get("web_title") or ""
    description=record.get("description") or ""
    h,d=visible_text(headline),visible_text(description)
    exact_raw=bool(headline and description and headline==description)
    exact_visible=bool(h and d and h==d)
    # near-duplicate 검출은 별도 진단. 이 결과로 원문이나 결합 텍스트를 자동 삭제하지 않는다.
    left=tokenize(unicodedata.normalize("NFKC",h).casefold())
    right=tokenize(unicodedata.normalize("NFKC",d).casefold())
    similarity=SequenceMatcher(None,left,right,autojunk=False).ratio() if left and right else 0.0
    near=bool(not exact_visible and min(len(left),len(right))>=5 and similarity>=.90 and min(len(left),len(right))/max(len(left),len(right))>=.80)
    raw=headline if exact_visible else headline+("\n"+description if description else "")
    sentiment=visible_text(raw)
    before,tokens=clean_tokens(sentiment)
    # 문서·headline/description·문장 경계를 넘지 않고, 원래 인접한 유효 토큰만 bigram으로 센다.
    pairs=[]
    for field in [h]+([] if exact_visible else [d]):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+",frequency_input(field)):
            words=tokenize(sentence)
            pairs.extend((a,b) for a,b in zip(words,words[1:]) if a not in ACTIVE_STOPWORDS and b not in ACTIVE_STOPWORDS and a not in CUSTOM_STOPWORDS and b not in CUSTOM_STOPWORDS)
    return {
        "headline_for_text":headline,"headline_source":"headline" if record.get("headline") else "web_title",
        "text_raw":raw,"text_sentiment":sentiment,"text_clean":" ".join(tokens),"tokens_clean":tokens,
        "raw_token_count":len(tokenize(raw)),"token_count_before_stopwords":len(before),
        "clean_token_count":len(tokens),"clean_unique_token_count":len(set(tokens)),
        "headline_description_exact_raw":exact_raw,"headline_description_exact_visible":exact_visible,
        "headline_description_near_duplicate":near,"headline_description_similarity":similarity,
        "text_policy_version":POLICY_VERSION},pairs
