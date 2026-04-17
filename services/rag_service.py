import re
import numpy as np
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from utils.logger import log_info, log_error


class RagService:
    """
    Improved RAG service using:
    - Sliding-window chunking with overlap (no lost context)
    - Hybrid retrieval: BM25 (keyword) + dense embeddings (semantic), score-fused
    - Cross-encoder reranking on top candidates for precision
    """

    CHUNK_SIZE = 500          # tokens (approx chars / 4)
    CHUNK_OVERLAP = 80        # overlap in words to avoid context cuts
    TOP_K_RETRIEVE = 20       # candidates passed to reranker
    TOP_K_FINAL = 3          # final results returned
    DENSE_WEIGHT = 0.6        # weight for semantic similarity
    BM25_WEIGHT = 0.4         # weight for keyword match
    MIN_SCORE = 0.20          # minimum hybrid score to include

    def __init__(self):
        log_info("Loading bi-encoder (all-MiniLM-L6-v2) for dense retrieval...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        log_info("Loading cross-encoder (ms-marco-MiniLM-L-6-v2) for reranking...")
        # This model is specifically trained for passage relevance scoring
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    def _clean_text(self, text: str, max_len: int = 8000) -> str:
        """Strip HTML, collapse whitespace, preserve paragraph breaks."""
        if not text:
            return ""
        cleaned = re.sub(r"(?i)<br\s*/?>", "\n", str(text))
        cleaned = re.sub(r"(?i)</(p|div|h[1-6]|li)>", "\n", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = cleaned.replace("&nbsp;", " ").replace("&amp;", "&")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned[:max_len]

    def _article_text_for_retrieval(self, article: Dict[str, Any], max_len: int = 8000) -> str:
        """Concatenate all meaningful fields, deduplicating repeated content."""
        parts: List[str] = []
        seen = set()

        # Title gets repeated at the front for emphasis
        title = str(article.get("title", "")).strip()
        if title:
            parts.append(title)
            parts.append(title)   # intentional repeat — boosts title relevance
            seen.add(title)

        for key in ["description", "description_text", "details", "content", "category", "folder"]:
            val = article.get(key)
            if val:
                val_str = str(val).strip()
                if val_str and val_str not in seen:
                    parts.append(val_str)
                    seen.add(val_str)

        for key in ["keywords", "tags"]:
            val = article.get(key)
            if isinstance(val, list) and val:
                val_str = " ".join(str(v) for v in val)
            elif val:
                val_str = str(val).strip()
            else:
                continue
            if val_str and val_str not in seen:
                parts.append(val_str)
                seen.add(val_str)

        return self._clean_text(" ".join(parts), max_len=max_len)

    # ------------------------------------------------------------------
    # Sliding-window chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping word-level windows.
        CHUNK_SIZE and CHUNK_OVERLAP are in words (not chars) for consistency.
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.CHUNK_SIZE, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end == len(words):
                break
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP  # slide with overlap

        return chunks

    def _build_chunks(self, knowledge_base: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Returns:
            chunk_texts  : flat list of chunk strings
            chunk_metas  : parallel list of metadata dicts (title, category, source article)
        """
        chunk_texts: List[str] = []
        chunk_metas: List[Dict] = []

        for article in knowledge_base:
            full_text = self._article_text_for_retrieval(article, max_len=100_000)
            chunks = self._chunk_text(full_text)

            for chunk in chunks:
                chunk_texts.append(chunk)
                chunk_metas.append({
                    "title": article.get("title", ""),
                    "category": article.get("category", ""),
                    "keywords": article.get("keywords", []),
                    "tags": article.get("tags", []),
                    "content": chunk,
                    # Keep a reference to reconstruct the full article if needed
                    "_source": article,
                })

        return chunk_texts, chunk_metas

    # ------------------------------------------------------------------
    # BM25 sparse retrieval
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase word tokenizer for BM25."""
        return re.findall(r"\b\w+\b", text.lower())

    def _bm25_scores(self, query: str, corpus: List[str]) -> np.ndarray:
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(self._tokenize(query))
        # Normalize to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        return scores

    # ------------------------------------------------------------------
    # Contact-support intent (unchanged — kept as safety net)
    # ------------------------------------------------------------------

    def _is_contact_support_intent(self, text: str) -> bool:
        if not text:
            return False
        triggers = [
            "contact support", "customer care", "customer service",
            "helpline", "phone number", "contact us", "reach myntra",
            "get in touch", "chat support",
        ]
        return any(t in text.lower() for t in triggers)

    # ------------------------------------------------------------------
    # Main retrieval pipeline
    # ------------------------------------------------------------------

    def filter_relevant_articles(
        self, query_text: str, knowledge_base: List[Dict]
    ) -> List[Dict]:
        """
        Full retrieval pipeline:
          1. Chunk all articles with overlap
          2. Hybrid score = weighted BM25 + cosine similarity
          3. Take top-K candidates
          4. Rerank with cross-encoder
          5. Return top-5 chunks (with metadata)
        """
        if not knowledge_base:
            return []

        if not query_text.strip():
            return knowledge_base[: self.TOP_K_FINAL]

        # ── Step 1: Build chunks ──────────────────────────────────────
        chunk_texts, chunk_metas = self._build_chunks(knowledge_base)

        if not chunk_texts:
            return knowledge_base[: self.TOP_K_FINAL]

        # ── Step 2: Dense (semantic) scores ──────────────────────────
        query_emb = self.embedder.encode(query_text, convert_to_numpy=True)
        doc_embs = self.embedder.encode(chunk_texts, convert_to_numpy=True, batch_size=64)

        norm_q = np.linalg.norm(query_emb)
        norm_d = np.linalg.norm(doc_embs, axis=1)
        if norm_q == 0:
            dense_scores = np.zeros(len(chunk_texts))
        else:
            dense_scores = np.dot(doc_embs, query_emb) / (norm_d * norm_q + 1e-10)

        # ── Step 3: BM25 (keyword) scores ────────────────────────────
        bm25_scores = self._bm25_scores(query_text, chunk_texts)

        # ── Step 4: Hybrid fusion ────────────────────────────────────
        hybrid_scores = (
            self.DENSE_WEIGHT * dense_scores + self.BM25_WEIGHT * bm25_scores
        )

        # Contact-support safety boost (kept but capped at +0.10)
        if self._is_contact_support_intent(query_text):
            contact_kws = {"contact", "customer care", "helpline", "phone", "chat", "help center"}
            for i, text in enumerate(chunk_texts):
                if contact_kws & set(text.lower().split()):
                    hybrid_scores[i] = min(hybrid_scores[i] + 0.10, 1.0)

        # ── Step 5: Candidate selection ───────────────────────────────
        top_indices = np.argsort(hybrid_scores)[::-1][: self.TOP_K_RETRIEVE]
        candidates = [
            (chunk_texts[i], chunk_metas[i], float(hybrid_scores[i]))
            for i in top_indices
            if hybrid_scores[i] >= self.MIN_SCORE
        ]

        if not candidates:
            # Graceful fallback: return best chunks ignoring threshold
            candidates = [
                (chunk_texts[i], chunk_metas[i], float(hybrid_scores[i]))
                for i in top_indices[: self.TOP_K_FINAL]
            ]

        # ── Step 6: Cross-encoder reranking ──────────────────────────
        pairs = [(query_text, cand[0]) for cand in candidates]
        rerank_scores = self.reranker.predict(pairs)  # returns raw logits

        reranked = sorted(
            zip(rerank_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        # ── Step 7: Build final result list ──────────────────────────
        results = []
        for rerank_score, (chunk_text, meta, hybrid_score) in reranked[: self.TOP_K_FINAL]:
            results.append({
                **meta,
                # Expose scores for debugging; strip before sending to LLM if desired
                "score": float(rerank_score),
                "hybrid_score": hybrid_score,
            })

        return results


# Singleton
rag_service = RagService()