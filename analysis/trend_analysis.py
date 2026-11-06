"""Topic trend by year."""
import config
from utils import helpers


def _yearly_pct(df,column,values,years):
    out={v:[] for v in values}
    for y in years:
        sub=df[df["year"]==y]
        for v in values:
            out[v].append(helpers.safe_pct(int((sub[column]==v).sum()),len(sub)))
    return out


def _classify(series_map):
    groups={"increasing":[],"decreasing":[],"stable":[],"insufficient data":[]}
    for k,s in series_map.items():
        groups[helpers.trend_label(s,config.TREND_PP_THRESHOLD,config.TREND_MIN_YEARS)].append(k)
    return groups


def compute_topic_trend(df,years,topic_order):
    topics=[t for t in topic_order if t!=config.UNMAPPED]
    pct=_yearly_pct(df,"topic",topics,years)
    recent=set(years[-config.RECENT_WINDOW:])
    recently_active=[t for t in topics if (df[(df["topic"]==t)]["year"].isin(recent)).any()]
    return dict(pct=pct,groups=_classify(pct),recent=recently_active,topics=topics)


def _group_lines(groups):
    out=[]
    for key in ("increasing","decreasing","stable"):
        items=groups[key]
        out.append(f"{key.capitalize():<11}: "+(", ".join(items) if items else "none"))
    return out


def build_topic_trend_report(tp,years,topic_rows):
    order=[r["topic"] for r in topic_rows if r["topic"]!=config.UNMAPPED][:config.TOP_TOPICS_IN_TREND]
    rows=[[y]+[f"{tp['pct'][t][i]:.0f}%" for t in order] for i,y in enumerate(years)]
    out=[f"Percentage of each year's questions (top {len(order)} topics by question count)",
         helpers.format_table(["Year"]+[t[:14] for t in order],rows),""]
    if len(years)<config.TREND_MIN_YEARS:
        out.append(f"Trend direction: insufficient data ({len(years)} year(s); need >= {config.TREND_MIN_YEARS}).")
    else:
        out+=_group_lines(tp["groups"])
    out.append("Recently active: "+(", ".join(tp["recent"]) if tp["recent"] else "none"))
    return "\n".join(out)
