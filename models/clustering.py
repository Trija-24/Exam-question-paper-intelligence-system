"""K-Means clustering of Sentence-BERT embeddings (unsupervised pattern discovery)."""
from collections import Counter

import numpy as np

import config


def cluster_questions(df,emb):
    n=len(df)
    if n<config.MIN_QUESTIONS_FOR_KMEANS:
        return dict(ok=False,reason=f"Too few questions ({n}) for K-Means (need >= {config.MIN_QUESTIONS_FOR_KMEANS}).")
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        if config.KMEANS_FIXED_K:
            ks=[min(config.KMEANS_FIXED_K,n-1)]
        else:
            ks=range(config.KMEANS_MIN_K,min(config.KMEANS_MAX_K,n-1)+1)
        best=None
        for k in ks:
            km=KMeans(n_clusters=k,n_init=10,random_state=config.RANDOM_STATE).fit(emb)
            if len(set(km.labels_))<2:
                continue
            sil=float(silhouette_score(emb,km.labels_))
            if best is None or sil>best[0]:
                best=(sil,km)
        if best is None:
            return dict(ok=False,reason="K-Means could not form at least two clusters.")
        sil,km=best
        clusters=[]
        for c in range(km.n_clusters):
            idx=np.where(km.labels_==c)[0]
            dist=np.linalg.norm(emb[idx]-km.cluster_centers_[c],axis=1)
            reps=[int(idx[i]) for i in np.argsort(dist)[:2]]
            topics=Counter(t for t in df["topic"].iloc[idx] if t!=config.UNMAPPED)
            types=Counter(df["qtype"].iloc[idx])
            clusters.append(dict(id=c+1,size=len(idx),reps=reps,
                                 topic=topics.most_common(1)[0][0] if topics else "n/a",
                                 qtype=types.most_common(1)[0][0] if types else "n/a"))
        return dict(ok=True,k=km.n_clusters,silhouette=sil,clusters=clusters,labels=km.labels_.tolist())
    except Exception as exc:
        return dict(ok=False,reason=f"K-Means failed ({str(exc)[:80]}).")
