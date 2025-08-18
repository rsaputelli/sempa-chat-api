import argparse, os, json, re
from typing import Optional, List
import httpx
from bs4 import BeautifulSoup
import numpy as np
from openai import OpenAI
import faiss

def fetch_text(url: str) -> str:
    r = httpx.get(url, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)

def chunk_text(s: str, size: int = 1200, overlap: int = 200) -> List[str]:
    chunks = []
    i = 0
    n = len(s)
    while i < n:
        j = min(n, i + size)
        chunks.append(s[i:j])
        i = j - overlap
        if i < 0: i = 0
    return chunks

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    client = OpenAI()
    vecs: List[np.ndarray] = []
    batch = 256
    for k in range(0, len(texts), batch):
        part = texts[k:k+batch]
        resp = client.embeddings.create(model=model, input=part)
        vecs.extend([np.asarray(d.embedding, dtype="float32") for d in resp.data])
    arr = np.vstack(vecs)
    # cosine sim with IndexFlatIP requires normalized vectors
    faiss.normalize_L2(arr)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir, e.g., clients/sempa/embeddings_v2")
    ap.add_argument("--urls-file", default="scripts/urls.txt")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load URLs (workflow writes scripts/urls.txt). If missing, use a small default set.
    urls = []
    if os.path.exists(args.urls_file):
        with open(args.urls_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    if not urls:
        urls = [
            "https://www.sempa.org/categories-dues/",
            "https://sempa.site-ym.com/general/register_member_type.asp",
            "https://www.sempa.org/contact/",
        ]

    texts, metas = [], []
    for u in urls:
        try:
            txt = fetch_text(u)
            for ch in chunk_text(txt):
                texts.append(ch)
                metas.append({"url": u, "source": u})
        except Exception as e:
            print(f"[warn] {u}: {e}")

    if not texts:
        raise SystemExit("No content fetched; nothing to index.")

    emb = embed_texts(texts)
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, os.path.join(args.out, "index.faiss"))
    with open(os.path.join(args.out, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    with open(os.path.join(args.out, "metas.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)
    with open(os.path.join(args.out, "dim.txt"), "w", encoding="utf-8") as f:
        f.write(str(dim))

    print(f"[ok] Wrote {len(texts)} chunks to {args.out} (dim={dim})")

if __name__ == "__main__":
    main()
