"""Module 4: concept analysis (semantic matching + knowledge graph queries)."""
import config
from knowledge_graph import graph_builder
from utils import helpers


def compute(G,years):
    recs=graph_builder.concept_records(G,years)
    by_topic={}
    for r in recs:
        by_topic.setdefault(r["topic"],[]).append(r)
    for t in by_topic:
        by_topic[t].sort(key=lambda r:(-r["freq"],-len(r["years"])))
    return by_topic


def build_report(by_topic,topic_rows,years,has_marks):
    if not by_topic:
        return ("No syllabus concepts were matched to questions (the syllabus may list no concept phrases, or the\n"
                "similarity threshold was not reached).")
    out=[]
    order=[r["topic"] for r in topic_rows if r["topic"] in by_topic][:config.MAX_TOPICS_IN_CONCEPT_REPORT]
    for t in order:
        if out:
            out.append("")
        out.append(f"Topic: {t}")
        rows=[[r["concept"][:30],r["freq"],f"{len(r['years'])}/{len(years)}",r["recent"],
               f"{r['marks']:g}" if has_marks else "n/a",r["repeated"],", ".join(r["related"])[:34] or "-"]
              for r in by_topic[t][:config.TOP_CONCEPTS_PER_TOPIC]]
        out.append(helpers.format_table(["Concept","Freq","Years","Recent","Marks","Repeated Qs","Related concepts"],rows))
    return "\n".join(out)
