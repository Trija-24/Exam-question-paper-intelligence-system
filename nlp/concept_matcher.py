import re

import numpy as np

import config


def _contains(phrase,text_lower):
    if len(phrase)<4:
        return False
    return re.search(r"\b"+re.escape(phrase.lower())+r"\b",text_lower) is not None


class ConceptMatcher:
    def __init__(self,topics,embedder):
        self.topics=topics
        self.names=[t["name"] for t in topics]
        topic_texts=[t["name"]+": "+", ".join(t["concepts"][:20]) if t["concepts"] else t["name"] for t in topics]
        self.topic_emb=embedder.encode(topic_texts)
        self.name_emb=embedder.encode(self.names)
        self.concept_names=[]
        self.concept_emb=[]
        for t in topics:
            if t["concepts"]:
                self.concept_names.append(t["concepts"])
                self.concept_emb.append(embedder.encode(t["concepts"]))
            else:
                self.concept_names.append([])
                self.concept_emb.append(None)

    def topic_scores(self,q_emb,texts):
        """Return (scores matrix [questions x topics], literal-match matrix)."""
        n=len(texts)
        T=len(self.topics)
        scores=np.zeros((n,T))
        lex=np.zeros((n,T),dtype=bool)
        low=[t.lower() for t in texts]
        sim_topic=q_emb@self.topic_emb.T
        sim_name=q_emb@self.name_emb.T
        for t in range(T):
            best=np.maximum(sim_topic[:,t],sim_name[:,t])
            if self.concept_emb[t] is not None:
                best=np.maximum(best,(q_emb@self.concept_emb[t].T).max(axis=1))
            phrases=[self.names[t]]+self.topics[t]["concepts"]
            for i in range(n):
                lex[i,t]=any(_contains(p,low[i]) for p in phrases)
            scores[:,t]=best+config.LEXICAL_TOPIC_BONUS*lex[:,t]
        return scores,lex

    def match_concepts(self,topic_name,q_vec,text):
        """Concepts of the assigned topic that the question is about."""
        if topic_name not in self.names:
            return []
        t=self.names.index(topic_name)
        if self.concept_emb[t] is None:
            return []
        sims=self.concept_emb[t]@q_vec
        low=text.lower()
        found=[]
        for name,s in zip(self.concept_names[t],sims):
            s=float(s)
            if _contains(name,low):
                s=max(s,config.LEXICAL_CONCEPT_SCORE)
            if s>=config.CONCEPT_THRESHOLD:
                found.append((name,s))
        found.sort(key=lambda x:-x[1])
        return [c for c,_ in found[:config.MAX_CONCEPTS_PER_QUESTION]]
