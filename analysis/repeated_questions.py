"""Repeated / rephrased questions and hidden question groups."""
import config
from utils import helpers


def enrich(pairs,df):
    for p in pairs:
        a=df.iloc[p["i"]]
        b=df.iloc[p["j"]]
        p["a"]=a
        p["b"]=b
        p["topic"]=a["topic"] if a["topic"]==b["topic"] else f"{a['topic']} / {b['topic']}"
        shared=[c for c in a["concepts"] if c in b["concepts"]]
        p["concept"]=", ".join(shared) if shared else ", ".join(sorted(set(a["concepts"])|set(b["concepts"])))
    return pairs


def pair_counts_per_question(pairs,n):
    counts=[0]*n
    for p in pairs:
        counts[p["i"]]+=1
        counts[p["j"]]+=1
    return counts


def build_similar_report(pairs):
    high=sum(1 for p in pairs if p["level"]=="Highly similar")
    out=[f"Pairs found: {len(pairs)} ({high} highly similar, {len(pairs)-high} similar). Each pair is listed once."]
    for p in pairs[:config.MAX_PAIRS_SHOWN]:
        a,b=p["a"],p["b"]
        out.append("")
        out.append(f"{p['level']}  score {p['score']:.2f}   {a['year']} Q{a['qid']} <-> {b['year']} Q{b['qid']}")
        out.append(f"  Topic  : {p['topic']}")
        if p["concept"]:
            out.append(f"  Concept: {p['concept']}")
        out.append(f"  A: {helpers.truncate(a['text'],100)}")
        out.append(f"  B: {helpers.truncate(b['text'],100)}")
    if len(pairs)>config.MAX_PAIRS_SHOWN:
        out.append(f"\n... {len(pairs)-config.MAX_PAIRS_SHOWN} more pair(s) not shown (see config.MAX_PAIRS_SHOWN).")
    return "\n".join(out)


def build_groups_report(clusters,df):
    if not clusters["ok"]:
        return clusters["reason"]
    out=[f"Question groups found: {clusters['k']}. Groups are numbered, not named."]
    for c in clusters["clusters"]:
        out.append(f"Group {c['id']}: {c['size']} question(s); dominant topic: {c['topic']}; dominant type: {c['qtype']}")
        for r in c["reps"]:
            row=df.iloc[r]
            out.append(f"    e.g. ({row['year']} Q{row['qid']}) {helpers.truncate(row['text'],85)}")
    return "\n".join(out)
