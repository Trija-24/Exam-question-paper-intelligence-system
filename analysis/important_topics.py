"""Module 9: transparent weighted importance score (analytical, NOT a prediction)."""
import numpy as np

import config
from utils import helpers


def _minmax(values):
    arr=np.asarray(values,dtype=float)
    if len(arr)==0:
        return arr
    lo,hi=arr.min(),arr.max()
    if hi==lo:
        return np.ones_like(arr) if hi>0 else np.zeros_like(arr)
    return (arr-lo)/(hi-lo)


def compute(df,years):
    topics=[t for t in df["topic"].unique() if t!=config.UNMAPPED]
    if not topics:
        return dict(rows=[],marks_used=False)
    marks_used=bool(df["marks"].notna().any())
    w=np.array([config.RECENCY_DECAY**(len(years)-1-i) for i in range(len(years))])
    freq,marks,rec,cov,meta=[],[],[],[],[]
    for t in topics:
        g=df[df["topic"]==t]
        present=np.array([float((g["year"]==y).any()) for y in years])
        freq.append(len(g))
        marks.append(float(g["marks"].fillna(0).sum()) if marks_used else float(len(g)))
        rec.append(float((w*present).sum()/w.sum()))
        cov.append(present.sum()/len(years))
        meta.append((g,int(present.sum())))
    W=config.IMPORTANCE_WEIGHTS
    score=(W["frequency"]*_minmax(freq)+W["marks"]*_minmax(marks)+W["recency"]*_minmax(rec)+W["coverage"]*_minmax(cov))
    rows=[]
    for i,t in enumerate(topics):
        g,yrs=meta[i]
        rows.append(dict(topic=t,score=float(score[i]),freq=freq[i],
                         marks_pct=helpers.safe_pct(marks[i],sum(marks)),recency=rec[i]*100,years=yrs))
    rows.sort(key=lambda r:-r["score"])
    return dict(rows=rows,marks_used=marks_used)


def build_report(imp,years):
    out=[]
    if not imp["marks_used"]:
        out.append("Marks were unavailable, so question counts were used for the marks component.")
        out.append("")
    rows=[[i+1,r["topic"][:34],f"{r['score']:.2f}",r["freq"],f"{r['marks_pct']:.1f}%",f"{r['recency']:.0f}%",
           f"{r['years']}/{len(years)}"] for i,r in enumerate(imp["rows"])]
    out.append(helpers.format_table(["Rank","Topic","Score","Freq","Marks%","Recent activity","Years"],rows))
    return "\n".join(out)
