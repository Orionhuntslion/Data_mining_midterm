"""제공 라벨을 검증하고 기사 수 × precision을 비교한다. 네트워크/수집 호출 없음."""
from __future__ import annotations

import hashlib
import json
import math
import runpy
from pathlib import Path

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/"guardian_api_pilot.py"))
import numpy as np
import pandas as pd
from guardian_api_config import STRATEGIES
from guardian_api_analysis import precision_table


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k,n):
    if n==0:
        return np.nan,np.nan
    if not 0<=k<=n:
        raise ValueError("Wilson 구간: 0 <= k <= n 필요")
    p=k/n
    z=1.959963984540054
    denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0,center-half),min(1,center+half)


def validate_inputs(root):
    results=root/"results"/"guardian_api_pilot"
    paths={"labels":root/"manual_review_sample_labeled.csv","supplied_precision":root/"guardian_strategy_precision.csv",
           "original_sample":results/"manual_review_sample.csv","volume":results/"volume_estimates.csv",
           "comparison":results/"strategy_comparison.csv"}
    inputs={key:pd.read_csv(path,keep_default_na=False) for key,path in paths.items()}
    labels=inputs["labels"]
    original=inputs["original_sample"]
    if not labels.guardian_id.is_unique or set(labels.guardian_id)!=set(original.guardian_id):
        raise ValueError("제공 라벨과 원래 표본의 Guardian ID가 일치하지 않습니다.")
    labels["relevant_us_equity_market"]=pd.to_numeric(labels.relevant_us_equity_market,errors="raise")
    if not labels.relevant_us_equity_market.isin([0,1]).all():
        raise ValueError("제공 라벨은 결측 없는 0/1이어야 합니다.")
    original_by_id=original.set_index("guardian_id")
    for row in labels.to_dict("records"):
        base=original_by_id.loc[row["guardian_id"]]
        for col in ["year","published_at","title","description","url"]:
            if str(row[col])!=str(base[col]):
                raise ValueError(f"제공 라벨 파일에서 원래 {col} 값이 변경되었습니다: {row['guardian_id']}")
        if set(row["query_strategy"].split(" | "))!=set(base.query_strategy.split(" | ")):
            raise ValueError("원래 전략 소속과 제공 라벨의 소속이 다릅니다.")
        previous=str(base.relevant_us_equity_market).strip()
        if previous and float(previous)!=float(row["relevant_us_equity_market"]):
            raise ValueError("기존 수동 라벨과 새 라벨이 충돌합니다. 자동 덮어쓰기를 중단합니다.")
    supplied=inputs["supplied_precision"]
    if not supplied.strategy.is_unique or set(supplied.strategy)!=set(STRATEGIES):
        raise ValueError("precision 파일의 전략 목록 오류")
    parts=labels.query_strategy.str.split(" | ",regex=False)
    for strategy in STRATEGIES:
        subset=labels.loc[parts.map(lambda x:strategy in x)]
        row=supplied.loc[supplied.strategy.eq(strategy)].iloc[0]
        if len(subset)!=row.reviewed or subset.relevant_us_equity_market.sum()!=row.relevant or not np.isclose(subset.relevant_us_equity_market.mean(),row.precision):
            raise ValueError(f"제공 precision과 라벨 재계산 불일치: {strategy}")
    return inputs,paths


def calculate(labels,supplied,comparison,volume):
    annual=precision_table(labels)
    annual=annual.merge(comparison[["query_strategy","year","unique_articles","status"]],on=["query_strategy","year"],validate="one_to_one")
    if not annual.status.eq("complete").all() or (annual.n_reviewed>annual.unique_articles).any():
        raise ValueError("완료된 파일럿과 검토 표본 크기를 확인하세요.")
    annual["pilot_review_is_census"]=annual.n_reviewed.eq(annual.unique_articles)
    annual["pilot_relevant_estimate"]=annual.unique_articles*annual.precision
    annual["known_relevant"]=annual.n_reviewed*annual.precision
    annual["unreviewed"]=annual.unique_articles-annual.n_reviewed
    annual["pilot_relevant_sharp_lower"]=annual.known_relevant
    annual["pilot_relevant_sharp_upper"]=annual.known_relevant+annual.unreviewed
    rows=[]
    for row in supplied.to_dict("records"):
        strategy=row["strategy"]
        n=int(row["reviewed"])
        k=int(row["relevant"])
        p=k/n
        low,high=wilson(k,n)
        counts=volume.loc[volume.query_strategy.eq(strategy)]
        if len(counts)!=1:
            raise ValueError("전략별 기사 수 메타데이터 중복/누락")
        N=int(counts.api_total_with_utc_buffer.iloc[0])
        year=annual.loc[annual.query_strategy.eq(strategy)]
        pilot_estimate=year.pilot_relevant_estimate.sum()
        pilot_N=year.unique_articles.sum()
        rows.append({"query_strategy":strategy,"reviewed":n,"relevant":k,"precision":p,
            "precision_wilson_95_low":low,"precision_wilson_95_high":high,
            "full_period_api_count_with_utc_buffer":N,"scenario_relevant_count":N*p,
            "scenario_relevant_count_low":N*low,"scenario_relevant_count_high":N*high,
            "scenario_noise_count":N*(1-p),"noise_per_relevant":(1-p)/p if p else np.nan,
            "pilot_unique_articles":pilot_N,"pilot_stratified_relevant_estimate":pilot_estimate,
            "pilot_population_weighted_precision":pilot_estimate/pilot_N if pilot_N else np.nan,
            "pilot_relevant_sharp_lower":year.pilot_relevant_sharp_lower.sum(),
            "pilot_relevant_sharp_upper":year.pilot_relevant_sharp_upper.sum(),
            "min_observed_year_precision":year.precision.min(),"max_observed_year_precision":year.precision.max(),
            "scenario_using_min_year_precision":N*year.precision.min(),
            "scenario_using_max_year_precision":N*year.precision.max(),
            "interpretation":"conditional scenario: pilot precision transfers to full target; Wilson interval is a binomial working-model reference, not temporal validation",
            "decision_status":"not_finalized"})
    return pd.DataFrame(rows),annual


def plot(summary,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y=np.arange(len(summary))
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),sharey=True)
    for ax,col,lo,hi,title in [
        (axes[0],"precision","precision_wilson_95_low","precision_wilson_95_high","Provided-label precision (reference Wilson 95%)"),
        (axes[1],"scenario_relevant_count","scenario_relevant_count_low","scenario_relevant_count_high","API count × pooled precision: conditional scenario")]:
        x=summary[col].to_numpy()
        errors=np.array([x-summary[lo].to_numpy(),summary[hi].to_numpy()-x])
        ax.errorbar(x,y,xerr=errors,fmt="o",capsize=4,color="#236899")
        ax.set_title(title,fontsize=11)
        ax.set_yticks(y,summary.query_strategy)
        ax.grid(axis="x",alpha=.25)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set(xlim=(0,1),xlabel="Precision")
    axes[1].set(xlabel="Estimated relevant articles (not validated for ten years)")
    axes[0].invert_yaxis()
    fig.suptitle("Strategy comparison — selection remains open",fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,bbox_inches="tight")
    plt.close(fig)


def write_review(root,summary,annual):
    lines=["# 기사 수 × precision · 전략 선택 보류", "", "제공 라벨 124건과 제공 precision 6개 전략을 대조했고 집계가 모두 일치했다.",
        "이 precision은 **수집된 기사 중 제공 라벨 기준 관련 기사 비율**이다. 분류기의 전체 accuracy나 recall과는 다르다.","",
        "| 전략 | 검토 n | precision | 전체 API 기사 수* | 관련 기사 수 시나리오 | 참고 95% 범위** |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.query_strategy} | {r.reviewed} | {r.precision:.1%} | {r.full_period_api_count_with_utc_buffer:,} | {r.scenario_relevant_count:,.0f} | {r.scenario_relevant_count_low:,.0f}–{r.scenario_relevant_count_high:,.0f} |")
    lines.extend(["", "* 기존 2017-01-01~2026-10-25 API 건수 조회의 UTC 경계 버퍼를 포함한 값. 추가 수집·요청은 하지 않았다.",
        "** precision의 Wilson 이항 작업모형 구간에 같은 기사 수를 곱한 참고 범위다. 시간 외삽·연도별 이질성·기사 의존성·라벨 오류·경계 건수 불확실성을 포함한 10년 모집단 신뢰구간이 아니다.","",
        "## 결정 시 읽는 방식", "",
        "- `기사 수 × precision`은 예상 관련 기사 수를 비교하는 시나리오다. precision만 높은 전략이나 수량만 많은 전략을 바로 최종 선택하지 않는다.",
        "- 전체 기사 수가 많아도 관련 문서를 정확히 선별할 수 있다는 뜻은 아니다. 각 전략의 예상 잡음 수와 관련 기사 1건당 잡음 수도 CSV에 저장했다.",
        "- 전략 간 같은 기사가 겹치므로 표의 관련 기사 수를 더해 합집합의 관련 기사 수로 해석하지 않는다. recall을 계산할 관련 기사 전체 모집단도 없다.",
        "- 작은 표본은 불확실성을 키우지만, 낮은 precision이 단순한 표본 오차라는 근거는 아니다.",
        "- 2017년과 2026년은 해당 파일럿 합집합 전수를 라벨링했다. 해당 주간의 낮은 precision은 연도 내 표본 추출만으로 설명할 수 없다. 다만 한 주가 해당 연도/10년을 대표하는지는 미검증이다.",
        "- 2020년은 82개 중 50개 표본이다. 연도별 표본 비율이 달라 pooled precision과 파일럿 모집단 가중 precision을 구분했다.",
        "- `strategy_yield_by_year.csv`에는 각 연도 N·검토 n·precision·전수 여부를 저장한다. 파일럿 관련 기사 수는 연도별 N×precision을 더해 추정하며, 미검토 기사가 모두 관련/비관련일 때의 논리적 상하한도 저장한다.","",
        "## 다음 검토 기준", "",
        "1. 현재 6개 전략을 유지하고 기간별 정밀도 차이, 관련 기사 수, 잡음량, 일별 분포를 함께 본다.",
        "2. 다른 주간/추가 연도로 표본을 확장할 때 동일한 라벨 기준을 먼저 고정하고, 제목+설명에 드러나는 미국 주식시장 관련성을 검토한다.",
        "3. broad 전략의 관련 기사 수 이점이 기간을 바꿔도 남는지, narrow 전략의 표본 부족·일별 결측이 어느 정도인지 비교한다.",
        "4. 추가 검증이 끝날 때까지 최종 전략과 전체 기간 크롤링은 보류한다. FGI와 높은 상관을 만드는 방향으로 전략을 선택하지 않는다.","",
        "제공 원본 두 CSV는 수정하지 않았다. 검증된 라벨은 기존 검토 CSV/라벨 보관 파일에 ID로 반영해 재실행 시 보존한다."])
    (root/"GUARDIAN_STRATEGY_YIELD_REVIEW.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(root=ROOT):
    root=Path(root)
    inputs,paths=validate_inputs(root)
    labels=inputs["labels"]
    summary,annual=calculate(labels,inputs["supplied_precision"],inputs["comparison"],inputs["volume"])
    results=root/"results"/"guardian_api_pilot"
    # 원래 표본의 문서/순서는 보존하고 라벨·메모만 ID로 반영한다.
    updated=inputs["original_sample"].copy()
    provided=labels.set_index("guardian_id")
    for col in ["relevant_us_equity_market","review_notes"]:
        updated[col]=updated.guardian_id.map(provided[col])
    archive=pd.read_csv(results/"manual_review_labels.csv",keep_default_na=False)
    archive=pd.concat([archive,updated],ignore_index=True).drop_duplicates("guardian_id",keep="last")
    updated.to_csv(results/"manual_review_sample.csv",index=False,encoding="utf-8-sig")
    archive.to_csv(results/"manual_review_labels.csv",index=False,encoding="utf-8-sig")
    precision_table(updated).to_csv(results/"manual_precision.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(results/"strategy_yield_comparison.csv",index=False,encoding="utf-8-sig")
    annual.to_csv(results/"strategy_yield_by_year.csv",index=False,encoding="utf-8-sig")
    write_review(root,summary,annual)
    plot(summary,root/"figures"/"guardian_strategy_yield.png")
    manifest={"label_file":{"path":str(paths["labels"]),"sha256":sha(paths["labels"])},
        "provided_precision_file":{"path":str(paths["supplied_precision"]),"sha256":sha(paths["supplied_precision"])},
        "provided_precision_matches_recalculation":True,"n_labeled":len(labels),
        "label_counts":labels.relevant_us_equity_market.value_counts().to_dict(),
        "network_requests":0,"full_collection":False,"strategy_selection":"not_finalized"}
    (results/"strategy_yield_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary[["query_strategy","reviewed","precision","full_period_api_count_with_utc_buffer","scenario_relevant_count"]].to_string(index=False))
    return summary,annual


if __name__=="__main__":
    run()
