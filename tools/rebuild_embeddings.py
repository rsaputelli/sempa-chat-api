# tools/rebuild_embeddings.py
import os, pickle, math
from typing import List
import numpy as np
import faiss  # faiss-cpu
from openai import OpenAI

# NEW: load .env so OPENAI_API_KEY is available when running this standalone script
from dotenv import load_dotenv

load_dotenv()

EMB_DIR = r"clients/sempa/embeddings"
TEXTS_TXT = os.path.join(EMB_DIR, "texts.txt")
CHUNKS_PKL = os.path.join(EMB_DIR, "chunks.pkl")
FAISS_PKL  = os.path.join(EMB_DIR, "faiss_index.pkl")

MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
DIM = 1536  # text-embedding-3-small dimension

def load_texts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    # Filter empty lines
    return [t for t in texts if t]

def get_embeddings(client: OpenAI, texts: List[str], batch_size: int = 256) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    arr = np.array(vecs, dtype="float32")
    # L2 normalize for inner product similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return arr

def main():
    assert os.path.exists(TEXTS_TXT), f"Missing {TEXTS_TXT}"
    client = OpenAI()  # needs OPENAI_API_KEY in env

    texts = load_texts(TEXTS_TXT)
    if not texts:
        raise RuntimeError("No texts found in texts.txt")

    print(f"Loaded {len(texts)} texts from texts.txt")
    embs = get_embeddings(client, texts)
    if embs.shape[1] != DIM:
        raise RuntimeError(f"Expected dim {DIM}, got {embs.shape[1]}")

    index = faiss.IndexFlatIP(DIM)
    index.add(embs)

    # Save chunks as a plain list[str] (no Pydantic)
    with open(CHUNKS_PKL, "wb") as f:
        pickle.dump(texts, f)

    # Save FAISS index as serialized bytes so it's easy to unpickle
    faiss_bytes = faiss.serialize_index(index)
    with open(FAISS_PKL, "wb") as f:
        pickle.dump(faiss_bytes, f)

    print(f"Wrote:\n - {CHUNKS_PKL}\n - {FAISS_PKL}")

if __name__ == "__main__":
    main()
