from collections import Counter

import networkx as nx

import config


def build_graph(df,topics,pairs):
    G=nx.DiGraph()
    for t in topics:
        tn=f"topic:{t['name']}"
        G.add_node(tn,kind="topic",label=t["name"])
        for c in t["concepts"]:
            cn=f"concept:{t['name']}:{c}"
            G.add_node(cn,kind="concept",label=c,topic=t["name"])
            G.add_edge(tn,cn,relation="has_concept")
    for i,r in df.iterrows():
        qn=f"q:{r.paper_idx}:{r.qid}"
        G.add_node(qn,kind="question",label=f"{r.year} Q{r.qid}",year=int(r.year),
                   marks=None if r.marks!=r.marks else float(r.marks),qtype=r.qtype,topic=r.topic,
                   repeated=False,text=r.text)
        yn=f"year:{r.year}"
        G.add_node(yn,kind="year",label=str(r.year))
        G.add_edge(qn,yn,relation="in_year")
        tyn=f"type:{r.qtype}"
        G.add_node(tyn,kind="type",label=r.qtype)
        G.add_edge(qn,tyn,relation="has_type")
        if r.marks==r.marks:
            mn=f"marks:{r.marks:g}"
            G.add_node(mn,kind="marks",label=f"{r.marks:g}")
            G.add_edge(qn,mn,relation="carries")
        if f"topic:{r.topic}" in G:
            G.add_edge(f"topic:{r.topic}",qn,relation="covers")
        for c in r.concepts:
            cn=f"concept:{r.topic}:{c}"
            if cn in G:
                G.add_edge(cn,qn,relation="asked_in")
    for p in pairs:
        a=df.iloc[p["i"]]
        b=df.iloc[p["j"]]
        qa=f"q:{a.paper_idx}:{a.qid}"
        qb=f"q:{b.paper_idx}:{b.qid}"
        G.add_edge(qa,qb,relation="similar_to",score=p["score"])
        G.nodes[qa]["repeated"]=True
        G.nodes[qb]["repeated"]=True
    return G


def concept_records(G,years):
    """One record per syllabus concept that appears in at least one question."""
    recent=set(years[-config.RECENT_WINDOW:])
    records=[]
    for node,d in G.nodes(data=True):
        if d["kind"]!="concept":
            continue
        qs=[q for q in G.successors(node) if G.nodes[q]["kind"]=="question"]
        if not qs:
            continue
        yrs=sorted({G.nodes[q]["year"] for q in qs})
        co=Counter()
        for q in qs:
            for pred in G.predecessors(q):
                if pred!=node and G.nodes[pred]["kind"]=="concept":
                    co[G.nodes[pred]["label"]]+=1
        records.append(dict(topic=d["topic"],concept=d["label"],freq=len(qs),years=yrs,
                            recent=sum(1 for q in qs if G.nodes[q]["year"] in recent),
                            marks=sum(G.nodes[q]["marks"] or 0 for q in qs),
                            repeated=sum(1 for q in qs if G.nodes[q]["repeated"]),
                            related=[c for c,_ in co.most_common(2)]))
    return records
