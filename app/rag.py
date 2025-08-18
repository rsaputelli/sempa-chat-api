# app/rag.py
import os
import pickle
import re
from typing import List, Tuple, Any, Optional

import numpy as np

# Try FAISS
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    _HAS_FAISS = False

# Try OpenAI client for query embeddings (only needed at query time)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()


class SimpleRAG:
    """
    Minimal RAG helper that loads a prebuilt FAISS index + chunk store from disk,
    then searches either:
      - via FAISS with OpenAI embeddings (if API key + library available), or
      - via a simple keyword fallback.

    Returned hits are tuples of (score, idx, text).
    """

    def __init__(self, emb_dir: Optional[str] = None):
        self.index: Any = None
        self.texts: List[str] = []
        self.dim: Optional[int] = None
        self._chunks_raw: List[Any] = []

        # Lazy OpenAI client
        self._oai_client: Optional[Any] = None
        self._embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

        if emb_dir:
            self.load(emb_dir)

    # ---------- Loading ----------
    def load(self, emb_dir: str) -> None:
        """
        Load FAISS index and chunks from a directory. Handles common filenames.
        """
        emb_dir = os.path.abspath(emb_dir)

        # Accept a few filename patterns for convenience
        idx_candidates = [
            os.path.join(emb_dir, "faiss_index.pkl"),
            os.path.join(emb_dir, "sempa_faiss_index.pkl"),
        ]
        chunk_candidates = [
            os.path.join(emb_dir, "chunks.pkl"),
            os.path.join(emb_dir, "sempa_chunks.pkl"),
        ]

        index_path = self._first_existing(idx_candidates)
        chunks_path = self._first_existing(chunk_candidates)

        if not index_path or not os.path.exists(index_path):
            # We allow running without an index (fallback search will still work)
            self.index = None
            self.dim = None
        else:
            with open(index_path, "rb") as f:
                self.index = pickle.load(f)
            # Detect index dimension if FAISS is present
            self.dim = getattr(self.index, "d", None)
            if self.dim is None and _HAS_FAISS:
                # Some FAISS objects expose nprobe etc, but .d should usually exist;
                # we keep None if not detectable.
                try:
                    self.dim = self.index.d  # type: ignore[attr-defined]
                except Exception:
                    self.dim = None

        if not chunks_path or not os.path.exists(chunks_path):
            # Still usable (you'll just get empty results)
            self._chunks_raw = []
            self.texts = []
            return

        # NOTE: Recent pickles created by tools/rebuild_embeddings.py store plain dicts.
        # Older pickles (pydantic v1 models) may break if unpickled in a different env;
        # since we've rebuilt, we assume simple dicts now.
        with open(chunks_path, "rb") as f:
            self._chunks_raw = pickle.load(f)

        self.texts = self._extract_texts(self._chunks_raw)

    @staticmethod
    def _first_existing(paths: List[str]) -> Optional[str]:
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _extract_texts(raw: List[Any]) -> List[str]:
        texts: List[str] = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                texts.append(_normalize_text(item["text"]))
            else:
                # Fallback: best effort stringification
                try:
                    texts.append(_normalize_text(str(item)))
                except Exception:
                    texts.append("")
        return texts

    # ---------- OpenAI Embeddings (lazy) ----------
    def _get_oai_client(self):
        if self._oai_client is not None:
            return self._oai_client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return None
        try:
            self._oai_client = OpenAI(api_key=api_key)
            return self._oai_client
        except Exception:
            return None

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        Embed a single query using OpenAI embeddings.
        Returns float32 numpy vector or None if unavailable.
        """
        client = self._get_oai_client()
        if client is None:
            return None

        try:
            resp = client.embeddings.create(
                model=self._embed_model,
                input=[query],
            )
            vec = resp.data[0].embedding
            return np.asarray(vec, dtype="float32")
        except Exception:
            return None

    # ---------- Search ----------
    def search(self, query: str, k: int = 5) -> List[Tuple[float, int, str]]:
        """
        Return top-k hits as (score, idx, text).
        If FAISS + embeddings available, do vector search (higher score = closer).
        Otherwise, fallback to keyword scoring.
        """
        if not self.texts:
            return []

        # Prefer FAISS+embedding if we can
        if _HAS_FAISS and self.index is not None:
            qvec = self._embed_query(query)
            if qvec is not None:
                return self._faiss_search(qvec, k)

        # Fallback: keyword scoring
        return self._keyword_search(query, k)

    def _faiss_search(self, qvec: np.ndarray, k: int) -> List[Tuple[float, int, str]]:
        # Ensure correct shape
        q = qvec.reshape(1, -1).astype("float32")

        # If index is IVF/Flat etc, search is the same API
        try:
            distances, indices = self.index.search(q, min(k, len(self.texts)))
        except Exception:
            # If something goes wrong, fallback gracefully
            return self._keyword_search(" ".join(map(str, qvec[:12])), k)

        hits: List[Tuple[float, int, str]] = []
        # FAISS returns smaller distance = closer for L2; many RAGs invert the sign to make 'score'
        # We'll convert to a simple descending score by negating distances.
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            txt = self.texts[idx]
            score = -float(dist)
            hits.append((score, int(idx), txt))

        # Sort by score desc (already roughly), just to be safe
        hits.sort(key=lambda x: x[0], reverse=True)
        return hits[:k]

    def _keyword_search(self, query: str, k: int) -> List[Tuple[float, int, str]]:
        """
        Extremely simple keyword overlap fallback. Not smart, but robust.
        """
        q_terms = set(self._tokenize(query))
        if not q_terms:
            q_terms = set(query.lower().split())

        scores: List[Tuple[float, int]] = []
        for i, text in enumerate(self.texts):
            t_terms = set(self._tokenize(text))
            if not t_terms:
                scores.append((0.0, i))
                continue
            overlap = len(q_terms & t_terms)
            norm = max(1, len(q_terms))
            score = overlap / norm
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[: min(k, len(scores))]
        return [(float(s), int(i), self.texts[i]) for s, i in top]

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

