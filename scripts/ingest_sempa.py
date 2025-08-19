import argparse, os, json, re, time
from typing import Optional, List
import httpx
from bs4 import BeautifulSoup
import numpy as np
from openai import OpenAI
import faiss

UA = "sempa-ingest/1.1 (+https://github.com/rsaputelli/sempa-chat-api)"

def fetch_text(url: str, timeout=25.0) -> str:
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        r = client.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(s: str, size: int = 1200, overlap: int = 200) -> List[str]:
    chunks = []
    i, n = 0, len(s)
    while i < n:
        j = min(n, i + size)
        chunks.append(s[i:j])
        i = max(0, j - overlap)
    return chunks

def embed_batches(client: OpenAI, inputs: List[str], model: str, batch: int = 64,
                  retries: int = 5, backoff: float = 1.7) -> np.ndarray:
    vecs: List[np.ndarray] = []
    total = len(inputs)
    for start in range(0, total, batch):
        part = inputs[start:start+batch]
        for attempt in range(retries):
            try:
                resp = client.embeddings.create(model=model, input=part)
                vecs.extend([np.asarray(d.embedding, dtype="float32") for d in resp.data])
                print(f"[embed] {min(start+len(part), total)}/{total}")
                break
            except Exception as e:
                wait = min(30.0, (backoff ** attempt))
                print(f"[warn] embed batch {start//batch} failed: {e!s}; retry in {wait:.1f}s")
                time.sleep(wait)
        else:
            raise RuntimeError("Failed to embed after retries")
    arr = np.vstack(vecs)
    faiss.normalize_L2(arr)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir, e.g., clients/sempa/embeddings_v2")
    ap.add_argument("--urls-file", default="scripts/urls.txt")
    ap.add_argument("--model", default=os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    ap.add_argument("--batch", type=int, default=int(os.getenv("EMBED_BATCH", "64")))
    ap.add_argument("--max-urls", type=int, default=0, help="Optional cap on number of URLs")
    ap.add_argument("--max-chunks", type=int, default=0, help="Optional cap on total chunks")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    urls: List[str] = []
    if os.path.exists(args.urls_file):
        with open(args.urls_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    if args.max_urls and len(urls) > args.max_urls:
        urls = urls[:args.max_urls]

    if not urls:
        raise SystemExit("No URLs to crawl (scripts/urls.txt missing or empty).")

    texts, metas = [], []
    for idx, u in enumerate(urls, 1):
        try:
            txt = fetch_text(u)
            chs = chunk_text(txt)
            if args.max_chunks and len(texts) + len(chs) > args.max_chunks:
                need = max(0, args.max_chunks - len(texts))
                chs = chs[:need]
            for ch in chs:
                texts.append(ch)
                metas.append({"url": u, "source": u})
            print(f"[ok] {idx}/{len(urls)} {u} -> {len(chs)} chunks (total={len(texts)})")
            if args.max_chunks and len(texts) >= args.max_chunks:
                print("[info] reached max-chunks cap; stopping crawl")
                break
        except Exception as e:
            print(f"[warn] {idx}/{len(urls)} {u}: {e!s}")

    if not texts:
        raise SystemExit("No content fetched; nothing to index.")

    client = OpenAI()
    emb = embed_batches(client, texts, model=args.model, batch=args.batch)
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    import json
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
