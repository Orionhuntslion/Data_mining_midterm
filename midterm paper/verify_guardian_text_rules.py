"""전처리 규칙의 금융 표현·원문·중복·문장 경계 보존을 합성 문자열로 검증."""
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_full_collection.py"))
import guardian_text_rules as rules


def main():
    original={"headline":"US stocks are NOT fucking bad!!! 🚀","web_title":"Different web title","description":"US stocks are NOT fucking bad!!! 🚀"}
    snapshot=original.copy()
    result,pairs=rules.article_texts(original)
    assert original==snapshot and result["headline_description_exact_raw"]
    assert result["text_raw"]==original["headline"]
    assert "NOT fucking bad!!! 🚀" in result["text_sentiment"]
    assert {"not","fucking","us","🚀"}<=set(result["tokens_clean"])
    html={"headline":"US stocks SOAR!!!","description":"<p>US stocks SOAR!!!</p>"}
    result,_=rules.article_texts(html)
    assert result["headline_description_exact_visible"] and result["text_raw"]==html["headline"]
    near={"headline":"US stocks surge after strong earnings report","description":"US stocks surge after very strong earnings report"}
    result,_=rules.article_texts(near)
    assert result["headline_description_near_duplicate"] and "\n" in result["text_raw"]
    assert near["description"] in result["text_raw"]
    minimal=rules.visible_text("<p>NOT bad!!! 🚀 👩‍💻 &amp; love &lt;3</p>\u00a0Really?")
    assert minimal=="NOT bad!!! 🚀 👩‍💻 & love <3 Really?"
    before,tokens=rules.clean_tokens("US UK FED S&P 500 FTSE 100 Nasdaq 100 $GME BRK.B AT&T HODL YOLO never sell-off!!! 🚀 https://example.invalid/path")
    assert {"us","uk","fed","s&p_500","ftse_100","nasdaq_100","$gme","brk.b","at&t","hodl","yolo","never","sell-off","🚀"}<=set(tokens)
    assert "https" not in tokens and "example.invalid" not in tokens
    result,pairs=rules.article_texts({"headline":"Stocks fall. Bonds rise.","description":"Profits and stocks"})
    assert ("stocks","fall") in pairs and ("bonds","rise") in pairs
    assert ("fall","bonds") not in pairs and ("rise","profits") not in pairs and ("profits","stocks") not in pairs
    assert rules.CUSTOM_STOPWORDS=={"happened","latest","coverage","rolling","live","news"}
    result,pairs=rules.article_texts({"headline":"Live news latest stocks business says new year","description":"Rolling coverage happened!!!"})
    assert "Live news latest" in result["text_sentiment"] and "Rolling coverage happened!!!" in result["text_sentiment"]
    assert not rules.CUSTOM_STOPWORDS.intersection(result["tokens_clean"])
    assert {"new","year","business","says"}<=set(result["tokens_clean"])
    assert all(not rules.CUSTOM_STOPWORDS.intersection(pair) for pair in pairs)
    print("PASS: raw/sentiment preserved; exactly six frequency-only custom stopwords; finance/negation/emoji retained; bigram boundaries; new/year/business/says retained")


if __name__=="__main__":
    main()
