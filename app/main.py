import os
from typing import List

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env early
load_dotenv()

from .tenants import ALLOWED_TENANTS, TENANT_CONFIG
from .rag import SimpleRAG

VERSION = "2025-08-18f"  # visible in /debug/env

# -------------------
# FastAPI app
# -------------------
app = FastAPI(title="SEMPA Chat API", version="0.2.0")

# -------------------
# CORS
# -------------------
DEFAULT_ALLOWED_ORIGINS = [
    "https://www.sempa.org",
    "https://sempa.org",
    "https://chat.sempa.org",
    "https://sempa-chat-api.onrender.com",
]

# Allow adding more origins via env: ALLOWED_ORIGINS="https://x.com,https://y.com"
EXTRA_ALLOWED = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

ALLOWED_ORIGINS = sorted(set(DEFAULT_ALLOWED_ORIGINS + EXTRA_ALLOWED))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Debug & utility routes
# -------------------
@app.get("/")
async def root():
    return {
        "service": "SEMPA Chat API",
        "ok": True,
        "endpoints": ["/healthz", "/debug/env", "/chat", "/debug/tenant/{client_id}"],
    }

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/debug/env")
async def debug_env():
    return {
        "version": VERSION,
        "has_key": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "allowed_origins": ALLOWED_ORIGINS,
        "cwd": os.getcwd(),
    }

# -------------------
# OpenAI client (lazy)
# -------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # package not available

_oai_client = None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

def get_oai():
    """Lazily init the OpenAI client if OPENAI_API_KEY is set and package is available."""
    global _oai_client
    if _oai_client is not None:
        return _oai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    _oai_client = OpenAI(api_key=api_key)
    return _oai_client

# -------------------
# Schemas
# -------------------
class ChatRequest(BaseModel):
    question: str
    client_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

# -------------------
# Helpers
# -------------------
def verify_origin(request: Request):
    origin = (request.headers.get("origin") or "").rstrip("/")
    referer = request.headers.get("referer") or ""
    if origin in ALLOWED_ORIGINS:
        return
    if referer and any(referer.startswith(a) for a in ALLOWED_ORIGINS):
        return
    allow_no_origin = os.getenv("ALLOW_NO_ORIGIN", "false").lower() == "true"
    if not allow_no_origin:
        raise HTTPException(status_code=403, detail="Origin not allowed")

def load_tenant(client_id: str) -> SimpleRAG:
    if client_id not in ALLOWED_TENANTS:
        raise HTTPException(status_code=403, detail="Unknown or disabled client_id")
    cfg = TENANT_CONFIG.get(client_id, {})
    emb_dir = cfg.get("embedding_dir")
    if not emb_dir or not os.path.exists(emb_dir):
        raise HTTPException(status_code=500, detail=f"Embeddings not found for tenant '{client_id}'")
    return SimpleRAG(emb_dir)

def load_prompt_for(client_id: str) -> str:
    cfg = TENANT_CONFIG.get(client_id, {})
    prompt_path = cfg.get("prompt_path")
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return (
        "You are a helpful assistant for SEMPA (Society of Emergency Medicine Physician Assistants).\n"
        "Answer clearly and concisely based ONLY on the provided context. If the answer is not in the context,\n"
        "say you’re not sure and suggest contacting SEMPA (https://www.sempa.org/contact/)."
    )

# -------------------
# Main routes
# -------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, _=Depends(verify_origin)):
    # Retrieve RAG results
    rag = load_tenant(req.client_id)
    hits = rag.search(req.question, k=5)  # list of (score, meta, text)
    contexts = [txt for _, _, txt in hits] if hits else []
    ctx_block = ("\n\n---\n\n").join(contexts)[:7000] if contexts else ""

    # Try GPT
    client = get_oai()
    if client is not None:
        system_prompt = load_prompt_for(req.client_id)
        user_msg = (
            "You are SEMPA’s assistant. Use the provided CONTEXT when it's relevant. "
            "If the answer isn’t clearly supported by the context or you’re unsure, say so briefly "
            "and recommend contacting SEMPA (https://www.sempa.org/contact/). "
            "Prefer SEMPA policies and site info over general knowledge.\n\n"
            f"QUESTION:\n{req.question}\n\n"
        )
        if ctx_block:
            user_msg += (
                f"CONTEXT:\n{ctx_block}\n\n"
                "(If you used the context, keep the answer concise and cite the specific pages/topics.)"
            )
        else:
            user_msg += "No context was retrieved."

        try:
            completion = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.2,
                max_tokens=500,
                top_p=1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",  "content": user_msg},
                ],
            )
            answer = (completion.choices[0].message.content or "").strip()
            if answer:
                sources = [c[:160] for c in contexts] if contexts else []
                return ChatResponse(answer=answer, sources=sources)
        except Exception:
            # fall back if OpenAI errors
            pass

    # Fallback
    if contexts:
        best = contexts[0]
        answer = "Here’s what I found in SEMPA materials:\n\n" + (best[:900] + ("..." if len(best) > 900 else ""))
        sources = [c[:160] for c in contexts]
        return ChatResponse(answer=answer, sources=sources)

    return ChatResponse(
        answer="I’m not sure from SEMPA documents. Please contact SEMPA at https://www.sempa.org/contact/ for assistance.",
        sources=[],
    )

@app.get("/debug/tenant/{client_id}")
async def debug_tenant(client_id: str):
    rag = load_tenant(client_id)
    return {
        "client_id": client_id,
        "has_index": rag.index is not None,
        "num_texts": len(rag.texts),
        "dim": rag.dim,
    }