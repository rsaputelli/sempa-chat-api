import pickle, os

IN_PATH  = r"clients\sempa\embeddings\sempa_chunks.pkl"
OUT_PATH = r"clients\sempa\embeddings\texts.txt"

with open(IN_PATH, "rb") as f:
    obj = pickle.load(f)

texts = []
if isinstance(obj, list):
    if obj and hasattr(obj[0], "page_content"):  # LangChain Document[]
        texts = [getattr(d, "page_content", "") or "" for d in obj]
    elif obj and isinstance(obj[0], dict):
        texts = [d.get("text") or d.get("content") or "" for d in obj]
    else:
        texts = [str(x) for x in obj]
else:
    try:
        texts = [str(x) for x in list(obj)]
    except Exception:
        texts = []

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for t in texts:
        t = (t or "").replace("\r\n","\n").replace("\r","\n").strip()
        if t:
            f.write(t + "\n")

print(f"Wrote {len(texts)} lines to {OUT_PATH}")
