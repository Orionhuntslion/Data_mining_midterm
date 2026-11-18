"""파일럿 변경분만 네트워크 없이 검증한다."""
import json
import runpy
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
pipeline=runpy.run_path(str(ROOT/"midterm paper.py"))
pipeline["load_module"](2,"midterm_stage_02")
pilot=pipeline["load_module"](9,"midterm_stage_09")


def main():
    import nbformat
    import pandas as pd

    assert pipeline["Config"]().start=="2017-01-01"
    assert pipeline["Config"]().end=="2026-10-25"
    assert pipeline["common"].PRIMARY_INDEX=="WSI_count"
    assert pilot.WINDOWS==["2017-01-03","2020-03-16","2026-10-23"]
    assert "secret123" not in pilot.safe_url("https://example.invalid/?api-key=secret123&from-date=2017-01-03")
    html='''<meta property="og:title" content="FUCKING BULLISH 🚀!!!"><meta name="description" content="NOT bad &amp; strong."><meta property="article:published_time" content="2020-03-16T14:30:00Z">'''
    one=pilot.metadata_from_article(html,"https://fixture.invalid/a")
    assert one["title_raw"]=="FUCKING BULLISH 🚀!!!" and "NOT bad" in one["raw_text"]
    two={**one,"doc_id":"second","url":"https://fixture.invalid/b","published_at":"2020-03-16T02:30:00Z"}
    with tempfile.TemporaryDirectory(prefix="source_pilot_verify_") as directory:
        folder=Path(directory)
        row=pilot.inspect_and_save(folder,"fixture","2020-03-16",[one,one.copy(),two],{})
        assert row["n_raw"]==3 and row["n_deduplicated"]==2
        assert row["n_exact_timestamps"]==2 and row["n_in_requested_et_date"]==1
        raw=json.loads((folder/"data/raw/news/fixture/2020/03/pilot_2020-03-16.json").read_text(encoding="utf-8"))
        assert raw["records"][0]["raw_text"]==one["raw_text"]
        cfg=pipeline["Config"](root=folder)
        try:
            pipeline["load_module"](2,"midterm_stage_02").collect_news(cfg)
        except ValueError as exc:
            assert "파일럿 검토" in str(exc)
        else:
            raise AssertionError("검토 전 전체 수집 가드 실패")
    for filename in ["midterm_source_pilot.ipynb","midterm_sentiment.ipynb"]:
        notebook=nbformat.read(ROOT/filename,as_version=4)
        nbformat.validate(notebook)
        for i,cell in enumerate(notebook.cells):
            if cell.cell_type=="code":
                compile(cell.source,f"{filename}:{i}","exec")
    summary=pd.read_csv(ROOT/"results/source_pilot/pilot_summary.csv")
    audit=pd.read_csv(ROOT/"results/source_pilot/news_sample_audit.csv")
    obtained=summary.loc[summary.status.eq("sample_obtained")]
    assert len(audit)==int(obtained.n_deduplicated.sum())==9
    assert audit.published_at.notna().all()
    assert audit.requested_day.eq(audit.date_et).all()
    assert audit[["neg","neu","pos","compound"]].notna().all().all()
    assert not audit.doc_id.duplicated().any()
    print("PASS: new target dates, primary index, metadata/raw preservation, ET alignment, deduplication, key redaction, full-collection guard, both notebooks, 9 actual sample records")


if __name__=="__main__":
    main()
