"""Module 3: aggregate topic analysis."""
import config
from utils import helpers


def compute(df,years):
    total_q=len(df)
    total_m=float(df["marks"].fillna(0).sum())
    recent=years[-config.RECENT_WINDOW:]
    rows=[]
    for topic,g in df.groupby("topic"):
        ys=sorted(g["year"].unique().tolist())
        m=float(g["marks"].fillna(0).sum())
        rows.append(dict(topic=topic,count=len(g),pct=helpers.safe_pct(len(g),total_q),marks=m,
                         marks_pct=helpers.safe_pct(m,total_m),years=len(ys),year_list=ys,
                         recent=int(g["year"].isin(recent).sum()),
                         recurrence=helpers.safe_pct(len(ys),len(years))))
    rows.sort(key=lambda r:-r["count"])
    return rows


def build_report(rows,years,has_marks):
    out=[f"Recent = questions in the last {config.RECENT_WINDOW} paper(s). Recurrence = share of years the topic appeared.",""]
    table=[[r["topic"][:34],r["count"],f"{r['pct']:.1f}%",f"{r['marks']:g}" if has_marks else "n/a",
            f"{r['marks_pct']:.1f}%" if has_marks else "n/a",f"{r['years']}/{len(years)}",r["recent"],
            f"{r['recurrence']:.0f}%"] for r in rows]
    out.append(helpers.format_table(["Topic","Qs","%Qs","Marks","%Marks","Years","Recent","Recurrence"],table))
    return "\n".join(out)
