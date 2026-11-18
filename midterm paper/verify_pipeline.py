"""네트워크 없이 수식·시차·전체 산출물을 검증. 합성 자료는 임시 폴더에만 둔다."""
from __future__ import annotations

import json
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".packages").is_dir():
    sys.path.insert(0, str(ROOT / ".packages"))

import nbformat
import numpy as np
import pandas as pd

pipeline = runpy.run_path(str(ROOT / "midterm paper.py"))
Config = pipeline["Config"]
load = pipeline["load_module"]


def verify_formulas():
    daily_module = load(5, "midterm_stage_05")
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-09-01"]*3+["2026-09-02"]*2),
        "compound": [0.5,-0.5,0,0,0], "sentiment_label": ["positive","negative","neutral","neutral","neutral"]})
    cfg = Config(start="2026-09-01",end="2026-09-03",min_daily_docs=1)
    result = daily_module.build_daily(frame,cfg)
    assert result.loc[0, ["WSI_continuous","WSI_count","WSI_binary"]].eq(50).all()
    assert result.loc[0,"n_binary"] == 2
    assert pd.isna(result.loc[1,"WSI_binary"])
    assert result.loc[1,"WSI_count"] == 50
    assert result.loc[2,"n_docs"] == 0
    assert result.loc[2, ["WSI_continuous","WSI_count","WSI_binary"]].isna().all()
    positive_only = frame.iloc[:1]
    edge = daily_module.build_daily(positive_only,cfg)
    assert edge.loc[0,"WSI_count"] == 100 and edge.loc[0,"WSI_binary"] == 100
    assert edge.loc[0,"WSI_continuous"] == 75
    assert 0 <= edge.loc[0,"binary_wilson_low_iid"] <= edge.loc[0,"binary_wilson_high_iid"] <= 100


def verify_calendar_lags():
    module = load(7,"midterm_stage_07")
    dates = pd.date_range("2026-08-01",periods=40)
    x = pd.Series(np.random.default_rng(42).normal(size=40), index=dates)
    fgi = x.shift(3)  # FGI at t+3 equals WSI at t.
    fgi = fgi.drop(dates[[9,10,16]])  # 欠측 날짜를 실제로 제거한 비정규 인덱스
    out = module.lag_table(x,fgi,7)
    peak = out.loc[out.pearson.idxmax()]
    assert peak.lag == 3 and np.isclose(peak.pearson,1)
    assert out.loc[out.lag.eq(3),"n_overlap"].iloc[0] == 34
    reverse = module.lag_table(fgi,x,7)
    assert reverse.loc[reverse.pearson.idxmax(),"lag"] == -3
    assert module.pair_stats(pd.Series([1,1,1]),pd.Series([1,2,3]))["status"] == "constant_series"
    assert module.pair_stats(pd.Series([1]),pd.Series([2]))["status"] == "fewer_than_3_pairs"


def verify_access_and_text():
    common = pipeline["common"]
    from unittest.mock import Mock, patch
    client = common.PublicClient(interval=0)
    response = Mock(status_code=429,url="https://example.invalid/",headers={"Retry-After":"60"})
    with patch.object(client.session,"get",return_value=response) as get:
        try:
            client._request("https://example.invalid/")
        except common.AccessError:
            pass
        else:
            raise AssertionError("429가 중단되지 않았습니다.")
        assert get.call_count == 1
    module = load(3,"midterm_stage_03")
    cleaned = module.sentiment_text("<p>NOT fucking bad!!! &amp; BULLISH $GME</p>")
    assert "NOT" in cleaned and "!!!" in cleaned
    tokens = module.tokens_for_frequency(cleaned)
    assert "fucking" in tokens and "bullish" in tokens and "$gme" in tokens and "not" in tokens
    try:
        module.parse_timestamp("2026-09-01 12:00:00")
    except ValueError:
        pass
    else:
        raise AssertionError("시간대 없는 시각이 통과했습니다.")


def create_fixture(folder):
    rows=[]
    calendar=pd.date_range("2026-09-01","2026-09-20")
    for day_idx,day in enumerate(calendar):
        for j in range(32):
            phrase = ("AMAZING profits, strong growth and wonderful gains!!!" if j < 6 + day_idx % 20 else
                      "Terrible losses, awful crash and fear!!!" if j < 27 else "Company announces quarterly meeting")
            rows.append({"published_at": f"{day:%Y-%m-%d}T16:00:00Z",
                         "raw_text": f"{phrase} report {j}", "source":"fixture.invalid",
                         "url":f"https://fixture.invalid/{day:%Y%m%d}/{j}"})
    rows.append(rows[0].copy())
    rows.append({**rows[0],"url":"https://fixture.invalid/syndicated-copy"})
    rows.append({"published_at":"2026-09-01T02:00:00Z","raw_text":"Previous New York day",
                 "source":"fixture.invalid","url":"https://fixture.invalid/out-of-range"})
    news = folder / "fixture_news.csv"
    pd.DataFrame(rows).to_csv(news,index=False)
    fgi = folder / "fixture_fgi.csv"
    pd.DataFrame({"date":[d.date() for d in calendar if d.weekday()<5],
                  "fgi":[float((i*19)%101) for i,d in enumerate(calendar) if d.weekday()<5]}).to_csv(fgi,index=False)
    return news,fgi


def verify_integration():
    with tempfile.TemporaryDirectory(prefix="midterm_verify_") as directory:
        folder=Path(directory)
        news,fgi=create_fixture(folder)
        cfg=Config(root=folder,start="2026-09-01",end="2026-09-20",text_input=str(news),
                   fgi_input=str(fgi),fgi_source_url="https://fixture.invalid/FGI-SYNTHETIC-TEST-ONLY")
        pipeline["run_all"](cfg)
        processed=pd.read_csv(cfg.processed / "processed_text_data.csv")
        assert len(processed) == 640
        diag=json.loads((cfg.results/"preprocessing_diagnostics.json").read_text(encoding="utf-8"))
        assert diag["duplicate_urls"] == 1 and diag["duplicate_headlines"] == 1 and diag["out_of_range"] == 1
        merged=pd.read_csv(cfg.processed / "merged_daily_data.csv")
        assert len(merged)==20 and merged.fgi.isna().sum()==6
        assert (cfg.results/"lag_correlations.csv").exists()
        assert len(list(cfg.figures.glob("*.png")))==18
        assert pd.read_csv(cfg.results/"human_review_template.csv").human_sentiment.isna().all()
        assert pd.read_csv(cfg.results/"word_daily_patterns.csv").groupby("term").size().eq(20).all()


def verify_notebook():
    notebook=nbformat.read(ROOT/"midterm_sentiment.ipynb",as_version=4)
    nbformat.validate(notebook)
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type=="code":
            compile(cell.source, f"notebook-cell-{i}", "exec")
    # Windows PowerShell의 ASCII 파이프 때문에 한글이 손상되지 않았는지 확인한다.
    assert "중간" in notebook.cells[0].source
    assert "???" not in notebook.cells[0].source


def verify_notebook_execution():
    """실제 노트북 셀을 새 커널에서 실행하되 자료/출력은 임시 폴더에만 둔다."""
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    with tempfile.TemporaryDirectory(prefix="midterm_notebook_verify_") as directory:
        folder=Path(directory)
        news,fgi=create_fixture(folder)
        notebook=nbformat.read(ROOT/"midterm_sentiment.ipynb",as_version=4)
        config_cells=[c for c in notebook.cells if c.cell_type=="code" and "# CONFIGURATION" in c.source]
        assert len(config_cells)==1
        config_cells[0].source=(
            f"cfg=Config(root=Path({str(folder)!r}),start='2026-09-01',end='2026-09-20',"
            f"text_input={str(news)!r},fgi_input={str(fgi)!r},"
            "fgi_source_url='https://fixture.invalid/FGI-SYNTHETIC-TEST-ONLY')\ncfg.setup()"
        )
        manager=KernelManager(kernel_name="python3")
        manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        client=NotebookClient(notebook,km=manager,timeout=180,resources={"metadata":{"path":str(ROOT)}})
        client.on_cell_start=lambda cell,cell_index,**kwargs: print(f"Notebook cell {cell_index+1}/{len(notebook.cells)}",flush=True)
        client.execute()
        assert not any(o.output_type=="error" for c in notebook.cells if c.cell_type=="code" for o in c.outputs)
        assert len(list((folder/"figures").glob("*.png")))==18
    print("PASS verify_notebook_execution (synthetic data only; original notebook unchanged)")


def main():
    if "--notebook" in sys.argv:
        verify_notebook_execution()
        return
    for check in [verify_formulas,verify_calendar_lags,verify_access_and_text,verify_notebook,verify_integration]:
        check()
        print(f"PASS {check.__name__}")
    print("검증 완료. 합성 자료/그림은 임시 폴더에서 삭제됐으며 실증 폴더에 저장하지 않았습니다.")


if __name__=="__main__":
    main()
