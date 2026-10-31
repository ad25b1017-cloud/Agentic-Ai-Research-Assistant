"""Hybrid retrieval: BM25 + dense, fused with RRF, then cross-encoder rerank."""
import re
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import EMBED_MODEL, RERANK_MODEL

def tokenize(t): return re.findall(r"\w+", t.lower())

@dataclass
class Hit:
    chunk: object
    score: float

class HybridIndex:
    def __init__(self):
        self.chunks, self.emb, self.bm25 = [], None, None
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)

    @property
    def papers(self): return sorted({c.paper for c in self.chunks})

    def add(self, chunks):
        if not chunks: return
        new = self.embedder.encode([c.text for c in chunks], normalize_embeddings=True, batch_size=32)
        self.emb = new if self.emb is None else np.vstack([self.emb, new])
        self.chunks += chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def search(self, query, top_k=6, pool=30, mode="hybrid_rerank", paper=None, section=None):
        """mode: bm25 | dense | hybrid | hybrid_rerank (used for ablations)."""
        if not self.chunks: return []
        bm = np.array(self.bm25.get_scores(tokenize(query)), dtype=float)
        de = self.emb @ self.embedder.encode(query, normalize_embeddings=True)
        mask = np.array([(paper is None or c.paper == paper) and (section is None or c.section == section)
                         for c in self.chunks])
        bm[~mask], de[~mask] = -1e9, -1e9
        n = int(mask.sum())
        if n == 0: return []
        if mode == "bm25":
            return [Hit(self.chunks[i], float(bm[i])) for i in np.argsort(-bm)[:min(top_k, n)]]
        if mode == "dense":
            return [Hit(self.chunks[i], float(de[i])) for i in np.argsort(-de)[:min(top_k, n)]]
        fused = {}
        for scores in (bm, de):
            for r, i in enumerate(np.argsort(-scores)[:min(pool, n)]):
                fused[i] = fused.get(i, 0) + 1 / (60 + r)
        cand = sorted(fused, key=fused.get, reverse=True)[:pool]
        if mode == "hybrid":
            return [Hit(self.chunks[i], fused[i]) for i in cand[:top_k]]
        logits = self.reranker.predict([(query, self.chunks[i].text) for i in cand])
        probs = 1 / (1 + np.exp(-np.asarray(logits)))
        order = np.argsort(-probs)[:top_k]
        return [Hit(self.chunks[cand[j]], float(probs[j])) for j in order]
