import re
import hashlib
import numpy as np
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from utils.logger import log_info, log_error


class RagService:
    """
    Render-optimized RAG service.
    - Model : all-MiniLM-L6-v2  (~80 MB, CPU-only)
    - Hybrid : BM25 (keyword) + cosine similarity (semantic)
    - Cache  : KB embeddings are cached by content hash so repeated
               calls don't re-encode the whole knowledge base
    """

    CHUNK_SIZE    = 500   # words per chunk
    CHUNK_OVERLAP = 80    # word overlap between chunks
    TOP_K_FINAL   = 3     # results returned
    TOP_K_PRE     = 20    # candidates before final cut
    DENSE_WEIGHT  = 0.6
    BM25_WEIGHT   = 0.4
    MIN_SCORE     = 0.20

    def __init__(self):
        log_info("Loading all-MiniLM-L6-v2 (CPU)...")
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",          # explicit — no GPU probe on Render
        )
        self.embedder.max_seq_length = 128  # cap sequence length → faster inference

        # Cache: kb_hash → (chunk_texts, chunk_metas, doc_embeddings)
        self._cache: Dict[str, Tuple[List[str], List[Dict], np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    def _clean_text(self, text: str, max_len: int = 8000) -> str:
        if not text:
            return ""
        t = re.sub(r"(?i)<br\s*/?>", "\n", str(text))
        t = re.sub(r"(?i)</(p|div|h[1-6]|li)>", "\n", t)
        t = re.sub(r"<[^>]+>", " ", t)
        t = t.replace("&nbsp;", " ").replace("&amp;", "&")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
        return t[:max_len]

    def _article_text(self, article: Dict[str, Any], max_len: int = 8000) -> str:
        parts, seen = [], set()

        title = str(article.get("title", "")).strip()
        if title:
            parts += [title, title]   # repeat title for emphasis
            seen.add(title)

        for key in ["description", "description_text", "details", "content", "category", "folder"]:
            val = str(article.get(key) or "").strip()
            if val and val not in seen:
                parts.append(val)
                seen.add(val)

        for key in ["keywords", "tags"]:
            val = article.get(key)
            kw = " ".join(str(v) for v in val) if isinstance(val, list) else str(val or "").strip()
            if kw and kw not in seen:
                parts.append(kw)
                seen.add(kw)

        return self._clean_text(" ".join(parts), max_len=max_len)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks, start = [], 0
        step = self.CHUNK_SIZE - self.CHUNK_OVERLAP
        while start < len(words):
            end = min(start + self.CHUNK_SIZE, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end == len(words):
                break
            start += step
        return chunks

    def _build_chunks(
        self, knowledge_base: List[Dict]
    ) -> Tuple[List[str], List[Dict]]:
        texts, metas = [], []
        for art in knowledge_base:
            for chunk in self._chunk_text(self._article_text(art, max_len=100_000)):
                texts.append(chunk)
                metas.append({
                    "title":    art.get("title", ""),
                    "category": art.get("category", ""),
                    "keywords": art.get("keywords", []),
                    "tags":     art.get("tags", []),
                    "content":  chunk,
                    "_source":  art,
                })
        return texts, metas

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _kb_hash(self, knowledge_base: List[Dict]) -> str:
        """Stable hash of the KB so we only re-encode when content changes."""
        raw = str([(a.get("title", ""), a.get("description_text", "")) for a in knowledge_base])
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_chunks_and_embeddings(
        self, knowledge_base: List[Dict]
    ) -> Tuple[List[str], List[Dict], np.ndarray]:
        key = self._kb_hash(knowledge_base)
        if key not in self._cache:
            log_info("RAG cache miss — encoding knowledge base...")
            chunk_texts, chunk_metas = self._build_chunks(knowledge_base)
            doc_embs = self.embedder.encode(
                chunk_texts,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False,
            )
            self._cache = {key: (chunk_texts, chunk_metas, doc_embs)}  # keep only latest
        return self._cache[key]

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _bm25_scores(self, query: str, corpus: List[str]) -> np.ndarray:
        bm25 = BM25Okapi([self._tokenize(d) for d in corpus])
        scores = bm25.get_scores(self._tokenize(query))
        mx = scores.max()
        return scores / mx if mx > 0 else scores

    # ------------------------------------------------------------------
    # Contact intent
    # ------------------------------------------------------------------

    def _is_contact_intent(self, text: str) -> bool:
        triggers = [
            "contact support", "customer care", "customer service",
            "helpline", "phone number", "contact us", "reach myntra",
            "get in touch", "chat support",
        ]
        return any(t in text.lower() for t in triggers)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def filter_relevant_articles(
        self, query_text: str, knowledge_base: List[Dict]
    ) -> List[Dict]:
        if not knowledge_base:
            return []
        if not query_text.strip():
            return knowledge_base[: self.TOP_K_FINAL]

        # 1. Chunks + cached embeddings
        chunk_texts, chunk_metas, doc_embs = self._get_chunks_and_embeddings(knowledge_base)
        if not chunk_texts:
            return knowledge_base[: self.TOP_K_FINAL]

        # 2. Dense scores
        q_emb = self.embedder.encode(
            query_text, convert_to_numpy=True, show_progress_bar=False
        )
        norm_q = np.linalg.norm(q_emb)
        norm_d = np.linalg.norm(doc_embs, axis=1)
        dense = (
            np.dot(doc_embs, q_emb) / (norm_d * norm_q + 1e-10)
            if norm_q > 0
            else np.zeros(len(chunk_texts))
        )

        # 3. BM25 scores
        bm25 = self._bm25_scores(query_text, chunk_texts)

        # 4. Hybrid fusion
        hybrid = self.DENSE_WEIGHT * dense + self.BM25_WEIGHT * bm25

        # 5. Contact intent boost (capped)
        if self._is_contact_intent(query_text):
            contact_kws = {"contact", "helpline", "phone", "chat", "support"}
            for i, txt in enumerate(chunk_texts):
                if contact_kws & set(txt.lower().split()):
                    hybrid[i] = min(hybrid[i] + 0.10, 1.0)

        # 6. Select top results
        top_idx = np.argsort(hybrid)[::-1]
        results = [
            {**chunk_metas[i], "score": float(hybrid[i])}
            for i in top_idx
            if hybrid[i] >= self.MIN_SCORE
        ][: self.TOP_K_FINAL]

        # Fallback if nothing clears threshold
        if not results:
            results = [
                {**chunk_metas[i], "score": float(hybrid[i])}
                for i in top_idx[: self.TOP_K_FINAL]
            ]

        return results


# Singleton
rag_service = RagService()
