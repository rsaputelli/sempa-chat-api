import os
from typing import List

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, Response
from fastapi import Query
from fastapi.responses import HTMLResponse
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env early
load_dotenv()

from .tenants import ALLOWED_TENANTS, TENANT_CONFIG
from .rag import SimpleRAG

VERSION = "2025-08-18g"  # visible in /debug/env

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

@app.api_route("/healthz", methods=["GET", "HEAD"], include_in_schema=False)
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
        "You are SEMPA's assistant. Use the provided CONTEXT when it strengthens your answer.\n"
        "If the question is general (for example, definitions or common healthcare concepts) or the CONTEXT is empty,\n"
        "answer briefly and accurately using well-accepted general knowledge. Do not invent SEMPA-specific policies,\n"
        "benefits, prices, dates, or legal guidance—if those are asked and not clearly in the CONTEXT, say you are not\n"
        "sure and direct users to SEMPA's official pages or contact form (https://www.sempa.org/contact/).\n"
        "Prefer concise, plain-language answers; when appropriate, add one line that relates the answer to the SEMPA audience."
    )

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
            "You are SEMPA's assistant. Use the provided CONTEXT when it is relevant. "
            "If the answer is not clearly supported by the context or you are unsure, say so briefly "
            "and recommend contacting SEMPA (https://www.sempa.org/contact/). "
            "Prefer SEMPA policies and site info over general knowledge.\n\n"
            f"QUESTION:\n{req.question}\n\n"
        )
        if ctx_block:
            user_msg += (
                f"CONTEXT:\n{ctx_block}\n\n"
                "(If you used the context, keep the answer concise and cite the specific pages or topics.)"
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
        answer = "Here is what I found in SEMPA materials:\n\n" + (best[:900] + ("..." if len(best) > 900 else ""))
        sources = [c[:160] for c in contexts]
        return ChatResponse(answer=answer, sources=sources)

    return ChatResponse(
        answer="I am not sure from SEMPA documents. Please contact SEMPA at https://www.sempa.org/contact/ for assistance.",
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


# --- Embedded widget endpoint ---
WIDGET_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SEMPA Chat</title>
  <style>
    :root { --pad:16px; --radius:12px; --muted:#f6f6f6; --border:#e5e5e5; --brand:#003366; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: var(--pad); color:#111; }
    .brand { background:var(--brand); color:#fff; border-radius:12px; padding:10px 14px; margin:0 0 12px 0; font-weight:600; }
    h3 { margin: 0 0 8px 0; font-size: 20px; color: var(--brand); }
    .card { border:1px solid var(--border); border-radius: var(--radius); padding: var(--pad); background:#fff; }
    textarea { width: 100%; resize: vertical; min-height: 90px; font: inherit; padding: 10px; border-radius: 10px; border:1px solid var(--border); box-sizing: border-box; }
    button { margin-top: 10px; padding: 10px 14px; border-radius: 10px; border:1px solid var(--border); background:var(--brand); color:#fff; cursor:pointer; }
    button[disabled] { opacity:.6; cursor: not-allowed; }
    .hint { color:#666; font-size: 12px; margin-top:6px; }
    pre { white-space: pre-wrap; word-wrap: break-word; background: var(--muted); padding: 12px; border-radius: 10px; border:1px solid var(--border); }
  </style>
</head>
<body>
  <div class="brand">SEMPA Chat</div>
  <div class="card">
    <h3>Ask SEMPA</h3>
    <textarea id="q" placeholder="Type your question... (Ctrl+Enter to send)"></textarea>
    <button id="ask">Ask</button>
    <div class="hint" id="hint">Answers are based on SEMPA documents. If we can't find it, we'll suggest contacting SEMPA.</div>
  </div>
  <div style="height:12px"></div>
  <div class="card">
    <pre id="out">Ask a question to get started.</pre>
  </div>

  <script>
    (function () {
      const qEl = document.getElementById('q');
      const btn = document.getElementById('ask');
      const out = document.getElementById('out');

      const params = new URLSearchParams(location.search);
      const clientId = params.get('client_id') || 'sempa';

      function postHeight() {
        try {
          const h = document.documentElement.scrollHeight;
          window.parent && window.parent.postMessage({ type: 'sempaWidgetSize', height: h }, '*');
        } catch(e) {}
      }
      const ro = new ResizeObserver(postHeight);
      ro.observe(document.body);

      async function ask() {
        const question = qEl.value.trim();
        if (!question) return;
        btn.disabled = true;
        out.textContent = 'Thinking...';
        try {
          const r = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, client_id: clientId })
          });
          if (!r.ok) throw new Error('HTTP ' + r.status);
          const data = await r.json();
          out.textContent = data.answer || '(no answer)';
        } catch (e) {
          out.textContent = 'Sorry—there was a problem reaching the chat service.';
          console.error(e);
        } finally {
          btn.disabled = false;
          postHeight();
        }
      }

      btn.addEventListener('click', ask);
      qEl.addEventListener('keydown', (ev) => {
        if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') ask();
      });

      postHeight();
    })();
  </script>
</body>
</html>"""

@app.get("/widget", response_class=HTMLResponse)
def widget(client_id: str = Query(default="sempa", description="Tenant id")):
    return HTMLResponse(
        content=WIDGET_HTML,
        headers={
            "Content-Security-Policy": "frame-ancestors 'self' https://www.sempa.org https://sempa.org;",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Content-Type-Options": "nosniff",
        },
    )


# --- Embeddable floating chat bubble (served at /embed.js) ---
JS_EMBED = r"""(function(){
  var thisScript = document.currentScript || (function(){var s=document.getElementsByTagName('script');return s[s.length-1];})();
  var base = new URL(thisScript.src).origin;
  var ds = thisScript.dataset || {};
  var clientId = ds.clientId || 'sempa';
  var color = ds.color || '#003366';   // SEMPA navy from sempa.org
  var side = (ds.position || 'right').toLowerCase(); // 'right' or 'left'
  var title = ds.title || 'SEMPA Chat';

  // Launcher button (speech bubble icon)
  var btn = document.createElement('button');
  btn.setAttribute('aria-label','Open ' + title);
  btn.style.cssText = [
    'position:fixed',
    'bottom:24px',
    (side==='left'?'left:24px':'right:24px'),
    'width:56px','height:56px',
    'border-radius:999px',
    'border:none',
    'box-shadow:0 8px 20px rgba(0,0,0,.15)',
    'background:'+color,
    'color:#fff',
    'cursor:pointer',
    'z-index:2147483647',
    'display:flex','align-items:center','justify-content:center'
  ].join(';');
  btn.innerHTML = '<svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor" aria-hidden="true"><path d="M20 2H4a2 2 0 0 0-2 2v14l4-3h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z"/></svg>';

  // Panel container
  var panel = document.createElement('div');
  panel.style.cssText = [
    'position:fixed',
    'bottom:96px',
    (side==='left'?'left:24px':'right:24px'),
    'width:380px','max-width:95vw',
    'height:560px','max-height:85vh',
    'background:#fff',
    'border:1px solid #e5e5e5',
    'border-radius:12px',
    'box-shadow:0 12px 30px rgba(0,0,0,.18)',
    'overflow:hidden',
    'display:none',
    'z-index:2147483000'
  ].join(';');

  // Header
  var header = document.createElement('div');
  header.style.cssText = 'height:44px;display:flex;align-items:center;justify-content:space-between;padding:0 12px;border-bottom:1px solid #eee;background:'+color+';color:#fff;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-size:14px';
  var titleEl = document.createElement('div');
  titleEl.textContent = title;
  var close = document.createElement('button');
  close.textContent = '✕';
  close.setAttribute('aria-label','Close');
  close.style.cssText = 'border:none;background:transparent;font-size:18px;cursor:pointer;color:#fff';
  header.appendChild(titleEl); header.appendChild(close);

  // Iframe (loads your /widget)
  var iframe = document.createElement('iframe');
  iframe.src = base + '/widget?client_id=' + encodeURIComponent(clientId);
  iframe.style.cssText = 'width:100%;height:calc(100% - 44px);border:0;display:block;background:#fff';
  iframe.setAttribute('allow','clipboard-write *');
  iframe.setAttribute('title', title);

  panel.appendChild(header);
  panel.appendChild(iframe);

  // Toggle logic
  function toggle(open){
    var isOpen = panel.style.display !== 'none';
    var willOpen = (open===undefined) ? !isOpen : !!open;
    panel.style.display = willOpen ? 'block' : 'none';
    btn.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
  }
  btn.addEventListener('click', function(){ toggle(); });
  close.addEventListener('click', function(){ toggle(false); });

  // Resize panel based on widget messages
  window.addEventListener('message', function(ev){
    var d = ev.data || {};
    if (d && d.type === 'sempaWidgetSize') {
      var h = Math.max(420, Math.min(800, +d.height || 560));
      panel.style.height = h + 'px';
    }
  });

  document.addEventListener('DOMContentLoaded', function(){
    document.body.appendChild(panel);
    document.body.appendChild(btn);
  });
})();"""

@app.get("/embed.js")
def embed_js(client_id: str = "sempa"):
    return Response(content=JS_EMBED, media_type="application/javascript")



