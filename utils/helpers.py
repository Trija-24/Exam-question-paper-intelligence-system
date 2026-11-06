"""Small helper functions shared by many modules (formatting, trends)."""
import numpy as np


def safe_pct(part,total):
    """Percentage that never divides by zero."""
    return 100.0*part/total if total else 0.0


def truncate(text,limit=90):
    text=" ".join(str(text).split())
    return text if len(text)<=limit else text[:limit-3]+"..."


def format_table(headers,rows):
    """Return a readable fixed-width table (first column left aligned)."""
    rows=[[str(c) for c in r] for r in rows]
    widths=[len(h) for h in headers]
    for r in rows:
        for i,c in enumerate(r):
            widths[i]=max(widths[i],len(c))

    def fmt(r):
        return "  ".join(c.ljust(widths[i]) if i==0 else c.rjust(widths[i]) for i,c in enumerate(r))

    lines=[fmt(headers),"  ".join("-"*w for w in widths)]
    lines+=[fmt(r) for r in rows]
    return "\n".join(lines)


def slope(values):
    """Least-squares slope of a list of numbers against 0,1,2,..."""
    if len(values)<2:
        return 0.0
    x=np.arange(len(values))
    return float(np.polyfit(x,np.asarray(values,dtype=float),1)[0])


def trend_label(values,threshold,min_points):
    """Classify a series of percentages as increasing / decreasing / stable."""
    if len(values)<min_points:
        return "insufficient data"
    s=slope(values)
    if s>threshold:
        return "increasing"
    if s<-threshold:
        return "decreasing"
    return "stable"
