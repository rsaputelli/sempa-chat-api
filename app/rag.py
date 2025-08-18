# app/rag.py
import os, json
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss

class SimpleRAG:
    def __init__(self, emb_dir: str):
        self.emb_dir = emb_dir
        index_path = os.path.join(emb_dir, "index.faiss")
        vecs_path  = os.path.join(emb_dir, "vectors.npy")
        data_path  = os.path.join(emb_dir, "data.jsonl")

        self.index = None
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.dim = 1536  # default for text-embedding-3-small

        if not (os.path.exists(index_path) and os.path.exists(vecs_path) and os.path.exists(data_path)):
            return  # leave empty index

        self.index = faiss.read_index(index_path)
        self.vecs  = np.load(vecs_path).astype("float32")
        self.dim   = int(self.vecs.shape[1])

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.texts.append(obj.get("text", ""))

                m = obj.get("meta") or {}
                if not isinstance(m, dict):
                    m = {"source": str(m)}
                # normalize common fields
                if "url" not in m and "source" in m:
                    m["url"] = m["source"]
                if "title" not in m:
                    m["title"] = "SEMPA"
                self.metas.append(m)

    def _embed(self, q: str) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        r = client.embeddings.create(model=model, input=[q])
        x = np.array(r.data[0].embedding, dtype="float32")
        faiss.normalize_L2(x.reshape(1, -1))
        return x

    def search(self, q: str, k: int = 5) -> List[Tuple[float, Dict[str, Any], str]]:
        if self.index is None:
            return []
        x = self._embed(q)
        scores, idxs = self.index.search(x.reshape(1, -1), k)
        out: List[Tuple[float, Dict[str, Any], str]] = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            meta = self.metas[i] if i < len(self.metas) else {}
            text = self.texts[i] if i < len(self.texts) else ""
            out.append((float(s), meta, text))
        return out
