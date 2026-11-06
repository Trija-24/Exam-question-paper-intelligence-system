"""Module 1: dataset summary."""
from utils import helpers


def compute(syllabus,topics,papers,n_requested,issues,df,years):
    per_paper=[]
    for p in papers:
        q=p["questions"]
        marks=[x["marks"] for x in q if x["marks"] is not None]
        per_paper.append(dict(name=p["name"],year=p["year"],pages=p["pages"],questions=len(q),
                              marks=sum(marks),with_marks=len(marks),stated=p.get("stated_total")))
    total_marks=sum(p["marks"] for p in per_paper)
    return dict(syllabus=syllabus["name"],n_topics=len(topics),n_requested=n_requested,n_loaded=len(papers),
                years=years,pages=sum(p["pages"] for p in per_paper),questions=len(df),
                total_marks=total_marks,has_marks=bool(df["marks"].notna().any()),
                marks_cov=helpers.safe_pct(int(df["marks"].notna().sum()),len(df)),
                per_paper=per_paper,issues=issues,syllabus_pages=syllabus["pages"])


def build_report(ds):
    out=[]
    n=max(ds["n_loaded"],1)
    out.append(f"Syllabus file          : {ds['syllabus']} ({ds['syllabus_pages']} pages, {ds['n_topics']} topics detected)")
    out.append(f"Question papers        : {ds['n_loaded']} loaded of {ds['n_requested']} provided")
    out.append(f"Years covered          : {', '.join(str(y) for y in ds['years'])}")
    out.append(f"Total pages (papers)   : {ds['pages']}")
    out.append(f"Total questions        : {ds['questions']}")
    if ds["has_marks"]:
        out.append(f"Total marks (parsed)   : {ds['total_marks']:g}   (marks found for {ds['marks_cov']:.0f}% of questions)")
        out.append(f"Average marks / paper  : {ds['total_marks']/n:.1f}")
    else:
        out.append("Total marks (parsed)   : not available (no marks detected in the papers)")
    out.append(f"Avg questions / paper  : {ds['questions']/n:.1f}")
    out.append("")
    rows=[[p["year"],p["name"][:28],p["pages"],p["questions"],f"{p['marks']:g}" if p["with_marks"] else "n/a",
           f"{p['stated']:g}" if p["stated"] else "n/a"] for p in ds["per_paper"]]
    out.append(helpers.format_table(["Year","File","Pages","Questions","Marks parsed","Marks stated"],rows))
    out.append("")
    out.append("Extraction issues      :"+(" none" if not ds["issues"] else ""))
    for i in ds["issues"]:
        out.append(f"  - {i}")
    return "\n".join(out)
