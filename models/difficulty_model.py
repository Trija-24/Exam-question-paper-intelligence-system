from collections import Counter

import numpy as np
import pandas as pd

import config

TYPE_WEIGHT={"Definition":0.15,"Theory / Explanation":0.35,"Application":0.50,"Comparison":0.40,
             "Numerical / Problem Solving":0.70,"Derivation":0.80,"Proof":0.85,"Other":0.30}
DIFF_NUM={"Easy":0,"Medium":1,"Hard":2}


def heuristic_labels(df):
    """Weak labels: 0.35 type + 0.25 relative marks + 0.20 length + 0.10 numeric + 0.10 concepts."""
    mx=df.groupby("paper_idx")["marks"].transform("max")
    marks_rel=(df["marks"]/mx).where(mx>0).fillna(0.5)
    type_w=df["qtype"].map(TYPE_WEIGHT).fillna(0.3)
    length=(df["words"]/40.0).clip(upper=1.0)
    numeric=df["numeric_flag"].astype(float)
    conc=(df["concepts"].apply(len)/3.0).clip(upper=1.0)
    score=0.35*type_w+0.25*marks_rel+0.20*length+0.10*numeric+0.10*conc
    labels=np.where(score<config.DIFF_EASY_MAX,"Easy",np.where(score>=config.DIFF_HARD_MIN,"Hard","Medium"))
    return score,labels


def build_features(df,emb,recurrence):
    f=pd.DataFrame(index=df.index)
    f["words"]=df["words"]
    f["chars"]=df["text"].str.len()
    median=df["marks"].median() if df["marks"].notna().any() else 0.0
    f["marks"]=df["marks"].fillna(median)
    f["marks_missing"]=df["marks"].isna().astype(int)
    f["n_concepts"]=df["concepts"].apply(len)
    f["numeric_flag"]=df["numeric_flag"].astype(int)
    f["is_subpart"]=df["is_sub"].astype(int)
    f["recurrence"]=recurrence
    f["topic_avg_marks"]=df.groupby("topic")["marks"].transform("mean").fillna(0)
    for t in config.QUESTION_TYPES:
        f["type_"+config.TYPE_SHORT[t].lower()]=(df["qtype"]==t).astype(int)
    n_comp=min(3,emb.shape[1],len(df)-1)
    if n_comp>=1:
        from sklearn.decomposition import PCA
        comps=PCA(n_components=n_comp,random_state=config.RANDOM_STATE).fit_transform(emb)
        for c in range(n_comp):
            f[f"semantic_pc{c+1}"]=comps[:,c]
    return f


def run_difficulty(df,emb,recurrence):
    """Adds the difficulty columns to df (Random Forest on weak labels; heuristic labels as fallback)."""
    score,weak=heuristic_labels(df)
    df["difficulty_score"]=score
    df["diff_heur"]=weak
    df["diff_heur_num"]=df["diff_heur"].map(DIFF_NUM)
    df["difficulty"]=weak
    if len(df)<config.MIN_RF_SAMPLES or len(Counter(weak))<2:
        return
    try:
        from sklearn.ensemble import RandomForestClassifier
        X=build_features(df,emb,recurrence)
        rf=RandomForestClassifier(**config.RF_PARAMS)
        rf.fit(X,weak)
        df["difficulty"]=rf.predict(X)
    except Exception:
        df["difficulty"]=weak
