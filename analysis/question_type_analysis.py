"""Module 5: aggregate question-type distribution (no per-question output)."""
import config
from utils import helpers


def compute(df):
    total=len(df)
    return {t:helpers.safe_pct(int((df["qtype"]==t).sum()),total) for t in config.QUESTION_TYPES}


def build_report(dist):
    return "\n".join(f"{t:<30}{v:5.1f}%" for t,v in sorted(dist.items(),key=lambda x:-x[1]))
