"""Module 2: difficulty analysis (aggregates only)."""
import config
from utils import helpers

LEVELS=["Easy","Medium","Hard"]


def compute(df,years):
    overall={l:helpers.safe_pct(int((df["difficulty"]==l).sum()),len(df)) for l in LEVELS}
    by_year={l:[] for l in LEVELS}
    for y in years:
        sub=df[df["year"]==y]
        for l in LEVELS:
            by_year[l].append(helpers.safe_pct(int((sub["difficulty"]==l).sum()),len(sub)))
    trend={l:helpers.trend_label(by_year[l],config.TREND_PP_THRESHOLD,config.TREND_MIN_YEARS) for l in LEVELS}
    return dict(overall=overall,by_year=by_year,trend=trend)


def build_report(res,years):
    out=["Difficulty levels are estimates; the papers carry no official difficulty ratings.",""]
    out.append("Overall: "+"   ".join(f"{l} {res['overall'][l]:.1f}%" for l in LEVELS))
    out.append("")
    rows=[[y]+[f"{res['by_year'][l][i]:.0f}%" for l in LEVELS] for i,y in enumerate(years)]
    out.append(helpers.format_table(["Year"]+LEVELS,rows))
    out.append("")
    if len(years)>=2:
        d=res["by_year"]["Hard"][-1]-res["by_year"]["Hard"][0]
        out.append(f"Change over time: Hard share moved from {res['by_year']['Hard'][0]:.0f}% ({years[0]}) to "
                   f"{res['by_year']['Hard'][-1]:.0f}% ({years[-1]}), a change of {d:+.0f} points.")
    if len(years)>=config.TREND_MIN_YEARS:
        out.append("Trend: "+", ".join(f"{l} {res['trend'][l]}" for l in LEVELS))
    else:
        out.append(f"Trend direction: insufficient data (need >= {config.TREND_MIN_YEARS} years).")
    return "\n".join(out)
