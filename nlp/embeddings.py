"""Sentence-BERT embedding layer (with a clearly labelled TF-IDF fallback)."""
import numpy as np

import config


class Embedder:
    """Turns text into L2-normalised vectors so a dot product equals cosine similarity."""

    def __init__(self,model_name=config.SBERT_MODEL_NAME):
        self.model=None
        self.vectorizer=None
        self.load_error=None
        try:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(model_name)
        except Exception as exc:
            self.load_error=str(exc)[:150]

    @property
    def is_sbert(self):
        return self.model is not None

    def prepare(self,corpus):
        """Fit the fallback vectoriser on the whole corpus (no-op for SBERT)."""
        if self.is_sbert:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,stop_words="english").fit(corpus)

    def encode(self,texts):
        texts=list(texts)
        if self.is_sbert:
            vecs=self.model.encode(texts,show_progress_bar=False,convert_to_numpy=True)
        else:
            vecs=self.vectorizer.transform(texts).toarray()
        norms=np.linalg.norm(vecs,axis=1,keepdims=True)
        norms[norms==0]=1.0
        return vecs/norms
