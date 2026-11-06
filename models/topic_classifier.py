from collections import Counter

import numpy as np

import config


def classify_topics(df,scores,lex,topic_names):
    n=len(df)
    rows=np.arange(n)
    order=np.argsort(-scores,axis=1)
    top=scores[rows,order[:,0]]
    second=scores[rows,order[:,1]] if scores.shape[1]>1 else np.zeros(n)
    margin=top-second
    sem_idx=order[:,0]
    sem_topics=np.array([topic_names[idx] if top[r]>=config.TOPIC_MIN_SIM else config.UNMAPPED
                         for r,idx in enumerate(sem_idx)],dtype=object)
    topics=list(sem_topics)

    confident=lex[rows,sem_idx]&(margin>=config.PSEUDO_LABEL_MARGIN)&(top>=config.TOPIC_MIN_SIM)
    counts=Counter(sem_topics[confident])
    eligible={t for t,c in counts.items() if c>=config.MIN_SVM_PER_CLASS}
    train_mask=confident&np.array([t in eligible for t in sem_topics])
    n_train=int(train_mask.sum())

    if n_train<config.MIN_SVM_LABELS or len(eligible)<config.MIN_SVM_CLASSES:
        return topics

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        texts=df["tfidf_text"].to_numpy()
        y=sem_topics[train_mask]
        pipe=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
                       ("svm",LinearSVC(C=1.0,class_weight="balanced",random_state=config.RANDOM_STATE))])
        pipe.fit(texts[train_mask],y)
        dec=pipe.decision_function(texts)
        classes=pipe.classes_
        for r in range(n):
            if train_mask[r]:
                continue
            if dec.ndim==1:
                best=classes[1] if dec[r]>0 else classes[0]
                gap=abs(dec[r])
            else:
                srt=np.sort(dec[r])[::-1]
                best=classes[int(np.argmax(dec[r]))]
                gap=srt[0]-srt[1]
            if gap>=config.SVM_MIN_MARGIN and topics[r]!=config.UNMAPPED:
                topics[r]=best
    except Exception:
        return list(sem_topics)
    return topics
