"""금융 웹 감성 연구의 단계별 실행 진입점. 보고서/연구 주장은 생성하지 않는다.

python "midterm paper.py" --stage all
python "midterm paper.py" --stage collect-fgi
python "midterm paper.py" --stage preprocess
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = ROOT / ".packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))


def load_module(number, name):
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / f"midterm paper_{number:02d}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


common = load_module(1, "midterm_common")
Config = common.Config
STAGES = {"preprocess": 3, "sentiment": 4, "daily": 5, "words": 6, "association": 7, "figures": 8}


def run_stage(cfg, stage):
    cfg.setup()
    print(f"\n[{stage}]", flush=True)
    if stage == "guardian-market":
        if str(ROOT) not in sys.path:
            sys.path.insert(0,str(ROOT))
        from guardian_market_analysis import main as analyze_market
        return analyze_market()
    if stage == "guardian-fgi-validate":
        if str(ROOT) not in sys.path:
            sys.path.insert(0,str(ROOT))
        from guardian_fgi_validation import main as validate_fgi
        return validate_fgi()
    if stage == "guardian-sentiment":
        if str(ROOT) not in sys.path:
            sys.path.insert(0,str(ROOT))
        from guardian_sentiment_analysis import main as score_guardian
        return score_guardian()
    if stage == "guardian-preprocess":
        if str(ROOT) not in sys.path:
            sys.path.insert(0,str(ROOT))
        from guardian_text_preprocess import main as preprocess_guardian
        return preprocess_guardian()
    if stage == "guardian-full":
        if str(ROOT) not in sys.path:
            sys.path.insert(0,str(ROOT))
        from guardian_full_collection import collect,FullSettings
        return collect(FullSettings(root=cfg.root))
    if stage == "guardian-api-pilot":
        if str(ROOT) not in sys.path:
            sys.path.insert(0,str(ROOT))
        from guardian_api_pilot import run
        from guardian_api_config import PilotSettings
        return run(PilotSettings(root=cfg.root))
    if stage == "pilot":
        load_module(2, "midterm_stage_02")
        return load_module(9, "midterm_stage_09").run(cfg)
    if stage in {"collect", "collect-news", "collect-fgi"}:
        module = load_module(2, "midterm_stage_02")
        return {"collect": module.run, "collect-news": module.collect_news, "collect-fgi": module.collect_fgi}[stage](cfg)
    if stage not in STAGES:
        raise ValueError(f"알 수 없는 단계: {stage}")
    number = STAGES[stage]
    return load_module(number, f"midterm_stage_{number:02d}").run(cfg)


def run_all(cfg, collect=True):
    if collect:
        run_stage(cfg, "collect")
    for stage in STAGES:
        run_stage(cfg, stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="guardian-market", choices=["guardian-market", "guardian-fgi-validate", "guardian-sentiment", "guardian-preprocess", "guardian-full", "guardian-api-pilot", "pilot", "all", "analysis", "collect", "collect-news", "collect-fgi", *STAGES])
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-10-25")
    parser.add_argument("--min-daily-docs", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=7)
    parser.add_argument("--text-input", help="published_at,raw_text,source,url 열의 CSV")
    parser.add_argument("--fgi-input", help="date,fgi[,category] 열의 CSV")
    parser.add_argument("--fgi-source", default="archive", choices=["archive","official","local"])
    parser.add_argument("--fgi-source-url", default="")
    parser.add_argument("--refresh", action="store_true", help="기존 캐시를 갱신하여 다시 요청")
    parser.add_argument("--allow-full-collection", action="store_true", help="출처 파일럿 검토 후 전체 수집을 명시적으로 활성화")
    args = parser.parse_args()
    cfg = Config(start=args.start,end=args.end,min_daily_docs=args.min_daily_docs,max_lag=args.max_lag,
        text_input=args.text_input,fgi_input=args.fgi_input,fgi_source=args.fgi_source,
        fgi_source_url=args.fgi_source_url,refresh=args.refresh,allow_full_collection=args.allow_full_collection)
    if args.stage in {"all","analysis"}:
        run_all(cfg,collect=args.stage=="all")
    else:
        run_stage(cfg,args.stage)


if __name__ == "__main__":
    main()
