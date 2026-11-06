import numpy as np
import pandas as pd

import config
from utils import helpers

FEATURES=["freq_total","freq_share","years_present","year_coverage","recent_freq","appeared_last_year",
          "marks_total","marks_share","avg_marks","marks_trend","freq_trend","years_since_last",
          "consecutive","concept_recurrence","concept_count","repeated_q","avg_difficulty",
          "numerical_share","theory_share","n_visible_years"]


def _make_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(**config.XGB_PARAMS)
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=config.RANDOM_STATE)


class History:
    """Topic-by-year statistics; features(t,k) only sees years[:k]."""

    def __init__(self,df,topic_names,years,pairs):
        self.years=years
        self.topics=topic_names
        self.qtot={y:int((df["year"]==y).sum()) for y in years}
        self.mtot={y:float(df.loc[df["year"]==y,"marks"].fillna(0).sum()) for y in years}
        self.has_marks=bool(df["marks"].notna().any())
        self.sub={t:df[df["topic"]==t] for t in topic_names}
        self.pair_years={t:[] for t in topic_names}
        for p in pairs:
            a=df.iloc[p["i"]]
            b=df.iloc[p["j"]]
            if a["topic"]==b["topic"] and a["topic"] in self.pair_years:
                self.pair_years[a["topic"]].append(max(a["year"],b["year"]))

    def features(self,t,k):
        ys=self.years[:k]
        sub=self.sub[t]
        sub=sub[sub["year"].isin(ys)]
        counts=[int((sub["year"]==y).sum()) for y in ys]
        marks=[float(sub.loc[sub["year"]==y,"marks"].fillna(0).sum()) for y in ys]
        qshare=[100.0*c/self.qtot[y] if self.qtot[y] else 0.0 for c,y in zip(counts,ys)]
        mshare=[100.0*m/self.mtot[y] if self.mtot[y] else 0.0 for m,y in zip(marks,ys)]
        present=[c>0 for c in counts]
        freq=sum(counts)
        last=max([i for i,p in enumerate(present) if p],default=-1)
        since=len(ys)-1-last if last>=0 else len(ys)+1
        streak=0
        for p in reversed(present):
            if not p:
                break
            streak+=1
        cy={}
        for _,r in sub.iterrows():
            for c in r["concepts"]:
                cy.setdefault(c,set()).add(r["year"])
        total_q=sum(self.qtot[y] for y in ys)
        total_m=sum(self.mtot[y] for y in ys)
        n_sub=len(sub)
        return dict(
            freq_total=freq,
            freq_share=100.0*freq/total_q if total_q else 0.0,
            years_present=sum(present),
            year_coverage=sum(present)/len(ys) if ys else 0.0,
            recent_freq=sum(counts[-config.RECENT_WINDOW:]),
            appeared_last_year=int(present[-1]) if present else 0,
            marks_total=sum(marks),
            marks_share=100.0*sum(marks)/total_m if total_m else 0.0,
            avg_marks=float(np.mean([m for m in sub["marks"] if m==m])) if sub["marks"].notna().any() else 0.0,
            marks_trend=helpers.slope(mshare),
            freq_trend=helpers.slope(qshare),
            years_since_last=since,
            consecutive=streak,
            concept_recurrence=sum(1 for v in cy.values() if len(v)>=2),
            concept_count=len(cy),
            repeated_q=sum(1 for y in self.pair_years[t] if y in ys),
            avg_difficulty=float(sub["diff_heur_num"].mean()) if n_sub else 0.0,
            numerical_share=float((sub["qtype"]=="Numerical / Problem Solving").mean()) if n_sub else 0.0,
            theory_share=float(sub["qtype"].isin(["Theory / Explanation","Definition"]).mean()) if n_sub else 0.0,
            n_visible_years=len(ys))

    def matrix(self,k):
        return pd.DataFrame([self.features(t,k) for t in self.topics],index=self.topics)[FEATURES]

    def labels(self,k):
        sub_year=self.years[k]
        return np.array([int((self.sub[t]["year"]==sub_year).any()) for t in self.topics])

    def recency_rate(self):
        """Fallback score: recency-weighted share of years in which the topic appeared."""
        n=len(self.years)
        w=np.array([config.RECENCY_DECAY**(n-1-i) for i in range(n)])
        out={}
        for t in self.topics:
            pres=np.array([float((self.sub[t]["year"]==y).any()) for y in self.years])
            out[t]=float((w*pres).sum()/w.sum())
        return out


def _eligible(y):
    return len(y)>=config.MIN_XGB_ROWS and y.sum()>=config.MIN_XGB_PER_CLASS and (len(y)-y.sum())>=config.MIN_XGB_PER_CLASS


def _walk_forward(hist):
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score
    folds=[]
    skipped=[]
    for k in range(config.MIN_TRAIN_YEARS,len(hist.years)):
        Xtr=pd.concat([hist.matrix(j) for j in range(1,k)])
        ytr=np.concatenate([hist.labels(j) for j in range(1,k)])
        test_year=hist.years[k]
        if not _eligible(ytr):
            skipped.append((test_year,f"only {len(ytr)} training rows / too few examples of one class"))
            continue
        Xte=hist.matrix(k)
        yte=hist.labels(k)
        try:
            model=_make_model()
            model.fit(Xtr,ytr)
            proba=model.predict_proba(Xte)[:,1]
            pred=(proba>=0.5).astype(int)
            p,r,f,_=precision_recall_fscore_support(yte,pred,average="binary",zero_division=0)
            auc=float(roc_auc_score(yte,proba)) if len(set(yte))>1 else None
            baseline=hist.matrix(k)["appeared_last_year"].to_numpy()
            folds.append(dict(test_year=test_year,train_years=hist.years[:k],n_train=len(ytr),n_test=len(yte),
                              accuracy=float(accuracy_score(yte,pred)),precision=float(p),recall=float(r),f1=float(f),
                              auc=auc,baseline=float(accuracy_score(yte,baseline))))
        except Exception as exc:
            skipped.append((test_year,f"training failed ({str(exc)[:50]})"))
    return folds,skipped


def _weighted_counter(sub,years,column,is_list=False):
    out={}
    for _,r in sub.iterrows():
        age=len(years)-1-years.index(r["year"])
        w=config.RECENCY_DECAY**age
        for v in (r[column] if is_list else [r[column]]):
            out[v]=out.get(v,0.0)+w
    return out


def _profile(hist,t):
    """Likely concepts / question type / difficulty from recency-weighted history (statistical)."""
    sub=hist.sub[t]
    if len(sub)==0:
        return dict(concepts=[],qtype=None,qtype_share=0.0,difficulty=None)
    cw=_weighted_counter(sub,hist.years,"concepts",True)
    tw=_weighted_counter(sub,hist.years,"qtype")
    dw=_weighted_counter(sub,hist.years,"difficulty")
    qtype=max(tw,key=tw.get)
    return dict(concepts=[c for c,_ in sorted(cw.items(),key=lambda x:-x[1])[:4]],qtype=qtype,
                qtype_share=100.0*tw[qtype]/sum(tw.values()),difficulty=max(dw,key=dw.get))


def _evidence(hist,t,feat):
    sub=hist.sub[t]
    n=len(hist.years)
    ys=sorted(sub["year"].unique().tolist())
    ev=[f"Appeared in {len(ys)} of {n} year(s)"+(f" ({', '.join(str(y) for y in ys)})" if ys else "")]
    if ys:
        gap=n-1-hist.years.index(ys[-1])
        ev.append(f"Appeared in the latest paper ({hist.years[-1]})" if gap==0 else f"Last appeared in {ys[-1]} ({gap} paper(s) ago)")
    else:
        ev.append("Never appeared in the supplied papers")
    if hist.has_marks:
        ev.append(f"Contributed {feat['marks_share']:.1f}% of historical marks")
    else:
        ev.append(f"Contributed {feat['freq_share']:.1f}% of historical questions (marks unavailable)")
    if feat["concept_recurrence"]>0:
        ev.append(f"{int(feat['concept_recurrence'])} concept(s) recurred in 2+ years")
    if feat["repeated_q"]>0:
        ev.append(f"{int(feat['repeated_q'])} repeated/rephrased question pair(s)")
    lab=helpers.trend_label(_share_series(hist,t),config.TREND_PP_THRESHOLD,config.TREND_MIN_YEARS)
    ev.append(f"Share trend: {lab}")
    return ev


def _share_series(hist,t):
    sub=hist.sub[t]
    return [100.0*int((sub["year"]==y).sum())/hist.qtot[y] if hist.qtot[y] else 0.0 for y in hist.years]


def predict_future(df,topics,years,pairs):
    names=[t["name"] for t in topics]
    hist=History(df,names,years,pairs)
    n_years=len(years)
    result=dict(trained=False,next_label=f"next examination (after {years[-1]})",
                folds=[],skipped=[],predictions=[],notes=[],reduced_confidence=True)

    final_X=pd.concat([hist.matrix(j) for j in range(1,n_years)]) if n_years>=2 else None
    final_y=np.concatenate([hist.labels(j) for j in range(1,n_years)]) if n_years>=2 else np.array([])
    pred_X=hist.matrix(n_years)

    model=None
    if n_years<3:
        result["notes"].append(f"Only {n_years} year(s) of papers: accuracy on past years cannot be measured "
                               f"(needs >= {config.MIN_TRAIN_YEARS+1} years).")
    else:
        result["folds"],result["skipped"]=_walk_forward(hist)
        if not result["folds"]:
            result["notes"].append("Too few years/questions to measure accuracy on past years reliably.")
    if final_X is not None and _eligible(final_y):
        try:
            model=_make_model()
            model.fit(final_X,final_y)
            proba=model.predict_proba(pred_X)[:,1]
            result["trained"]=True
        except Exception:
            model=None
    elif final_X is None:
        result["notes"].append("Only one paper supplied: there is no history to learn from.")

    if model is None:
        rate=hist.recency_rate()
        proba=np.array([min(max(rate[t],0.01),0.99) for t in names])
        result["notes"].append("Likelihoods are based on how recently and how often each topic appeared "
                               "(too little history for a learned estimate).")
    else:
        result["reduced_confidence"]=len(result["folds"])<2 or len(final_y)<30
        if len(result["folds"])<2:
            result["notes"].append("Fewer than 2 past-year checks were possible: prediction confidence is reduced.")

    order=np.argsort(-proba)[:config.TOP_N_PREDICTIONS]
    for idx in order:
        t=names[idx]
        feat=pred_X.iloc[idx].to_dict()
        result["predictions"].append(dict(topic=t,likelihood=float(proba[idx]),evidence=_evidence(hist,t,feat),
                                          **_profile(hist,t)))
    return result
