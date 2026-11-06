"""Cosine-similarity search for repeated / rephrased questions."""
import numpy as np

import config


def find_similar_pairs(df,emb):
    """Return each similar pair exactly once (earlier year first), best first."""
    n=len(df)
    if n<2:
        return []
    sims=emb@emb.T
    iu,ju=np.triu_indices(n,k=1)
    scores=sims[iu,ju]
    years=df["year"].to_numpy()
    pairs=[]
    for i,j,s in zip(iu,ju,scores):
        if s<config.SIM_MID:
            continue
        if years[i]==years[j] and not config.INCLUDE_SAME_YEAR_PAIRS:
            continue
        if years[i]>years[j]:
            i,j=j,i
        pairs.append(dict(i=int(i),j=int(j),score=float(s),
                          level="Highly similar" if s>=config.SIM_HIGH else "Similar"))
    pairs.sort(key=lambda p:-p["score"])
    return pairs


def semantic_search(query,embedder,emb,df,top_k=5):
    """Return the top_k questions most similar in meaning to a free-text query."""
    q=embedder.encode([query])[0]
    sims=emb@q
    order=np.argsort(-sims)[:top_k]
    return [(int(i),float(sims[i])) for i in order]
