# app/main.py
import os
import re
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from .tenants import ALLOWED_TENANTS, TENANT_CONFIG
from .rag import SimpleRAG

VERSION = "2025-08-18-h2"

# -------------------
# FastAPI app
# -------------------
app = FastAPI(title="SEMPA Chat API", version="0.3.0")

# Serve /static only if the folder exists (so we don't crash if assets aren't there yet)
if os.path.isdir("app/static"):
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

# -------------------
# CORS
# -------------------
DEFAULT_ALLOWED_ORIGINS = [
    "https://www.sempa.org",
    "https://sempa.org",
    "https://chat.sempa.org",
    "https://sempa-chat-api.onrender.com",
]
EXTRA_ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGINS = sorted(set(DEFAULT_ALLOWED_ORIGINS + EXTRA_ALLOWED))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Debug & health
# -------------------
@app.api_route("/", methods=["GET"], include_in_schema=False)
async def root():
    return {
        "service": "SEMPA Chat API",
        "ok": True,
        "version": VERSION,
        "endpoints": ["/healthz", "/debug/env", "/chat", "/debug/tenant/{client_id}", "/widget", "/embed.js"],
    }

# explicit HEAD for Render checks
@app.head("/")
async def root_head():
    return Response(status_code=200)

@app.api_route("/healthz", methods=["GET", "HEAD"], include_in_schema=False)
async def healthz():
    return {"ok": True}

@app.get("/debug/env")
async def debug_env():
    return {
        "has_key": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "allowed_origins": ALLOWED_ORIGINS,
        "version": VERSION,
        "cwd": os.getcwd(),
    }

# -------------------
# OpenAI client (lazy)
# -------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

_oai_client = None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

def get_oai():
    global _oai_client
    if _oai_client is not None:
        return _oai_client
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    _oai_client = OpenAI(api_key=key)
    return _oai_client

# -------------------
# Models
# -------------------
class Source(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    score: Optional[float] = None

class ChatRequest(BaseModel):
    question: str
    client_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []

# -------------------
# FAQ + synonyms (from Streamlit version)
# -------------------
FAQS = {
    "join or renew": "Visit the Join or Renew page to get started with SEMPA membership: https://sempa.site-ym.com/general/register_member_type.asp",
    "membership categories": "SEMPA offers various membership categories. See: https://www.sempa.org/categories-dues/",
    "register for events": "Go to the Event Calendar or Education sections at sempa.org to register for events like SEMPA 360.",
    "member discounts": "Yes! Members save up to 40% on events, CME, and partner resources.",
    "access session recordings": "Log in to your SEMPA account and go to the 'My Education' section to find session recordings.",
    "contact sempa": "You can reach SEMPA at sempa@sempa.org or call 877-297-7954.",
}

SYNONYM_MAP = {
    "join or renew": ["join", "sign up", "enroll", "become a member", "renew", "renewal"],
    "membership categories": ["categories", "types", "levels", "dues", "cost"],
    "register for events": ["register", "sign up", "attend", "enroll", "conference"],
    "member discounts": ["discount", "save", "benefits", "perks"],
    "access session recordings": ["recordings", "sessions", "videos", "past sessions"],
    "contact sempa": ["contact", "email", "call", "support", "help"],
}

def find_faq_answer(q: str) -> Optional[str]:
    text = q.strip().lower()
    # direct key match
    for key, ans in FAQS.items():
        if key in text:
            return ans
    # synonym match (word-boundary regex)
    for key, syns in SYNONYM_MAP.items():
        for s in syns:
            if re.search(rf"(^|\b){re.escape(s)}(\b|$)", text):
                return FAQS.get(key)
    return None

# -------------------
# Routing heuristics
# -------------------
def normalize_for_search(q: str) -> str:
    """Expand 'PA' to Physician Assistant/Associate unless the text clearly refers to Pennsylvania."""
    text = q
    # Obvious Pennsylvania cues
    pa_state = re.search(r"\b(Pennsylvania|Philly|Philadelphia|Pittsburgh|Harrisburg|PA\s?\d{5})\b", text, re.I)
    if not pa_state:
        text = re.sub(r"\bPAs\b", "Physician Assistants (Physician Associates)", text)
        text = re.sub(r"\bPA\b", "Physician Assistant (Physician Associate)", text)
    return text

def is_definitional(q: str) -> bool:
    t = q.strip().lower()
    return bool(re.match(r"^(what is|who is|define|explain)\b", t))

def needs_clarification(q: str) -> bool:
    t = q.strip().lower()
    # Heuristic: SEMPA-ish but ambiguous
    return any(w in t for w in ["membership", "dues", "join", "renew", "benefit", "discount", "conference", "event", "policy"]) and not is_definitional(t)

def build_sources(hits) -> List[Source]:
    out: List[Source] = []
    for item in (hits or [])[:3]:
        try:
            # Unpack common shapes: (score, meta, text)
            score = None
            meta = None
            txt = None
            if isinstance(item, (list, tuple)):
                if len(item) >= 1: score = item[0]
                if len(item) >= 2: meta = item[1]
                if len(item) >= 3: txt = item[2]
            else:
                txt = str(item)

            if DEBUG_SOURCES:
                print("build_sources item:", type(item).__name__, "meta:", type(meta).__name__ if meta is not None else None)

            url = None
            title = None

            # Prefer explicit metadata when it's a dict
            if isinstance(meta, dict):
                url = meta.get("url") or meta.get("source")
                title = meta.get("title") or (url or None)
            elif isinstance(meta, str) and meta.startswith("http"):
                url = meta
                title = meta
            elif isinstance(meta, (list, tuple)):
                # scan any string elements for a URL
                for m in meta:
                    if isinstance(m, str) and m.startswith("http"):
                        url = m
                        title = m
                        break

            # If still no URL, try to pull one out of the text
            if not url and isinstance(txt, str):
                url = first_url_from_text(txt)
                if url and not title:
                    title = title_from_url(url)

            if url:
                try:
                    fscore = float(score) if score is not None else None
                except Exception:
                    fscore = None
                out.append(Source(title=title or "Source", url=url, score=fscore))
        except Exception as e:
            if DEBUG_SOURCES:
                print("build_sources error:", e)
            continue
    return out
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
    # Default SEMPA guardrails + the routing behaviors
    return (
        "You are a helpful assistant for SEMPA (Society of Emergency Medicine Physician Assistants).\n"
        "Use SEMPA documents if provided. If context is weak or empty and the question is general/common-knowledge,\n"
        "answer briefly and accurately using well-accepted medical knowledge (tie to emergency medicine when relevant).\n"
        "If the question is SEMPA-specific but ambiguous or not supported by context, ask one concise clarifying question.\n"
        "Prefer SEMPA policies over general sources when both exist."
    )

# -------------------
# Chat route with routing/heuristics + FAQs + sources
# -------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, _=Depends(verify_origin)):
    user_q = req.question.strip()
    if not user_q:
        return ChatResponse(answer="Please enter a question.", sources=[])

    # FAQ short-circuit
    faq = find_faq_answer(user_q)
    if faq:
        return ChatResponse(answer=faq, sources=[])

    # Normalize for search
    search_q = normalize_for_search(user_q)

    # Retrieve RAG results
    rag = load_tenant(req.client_id)
    hits = rag.search(search_q, k=5)
    contexts = [txt for _, _, txt in (hits or [])] if hits else []
    ctx_block = ("\n\n---\n\n").join(contexts)[:7000] if contexts else ""
    oai = get_oai()
    system_prompt = load_prompt_for(req.client_id)

    # With context -> LLM answer citing SEMPA
    if contexts and oai is not None:
        user_msg = (
            "Answer the QUESTION using the CONTEXT. If the answer is unclear in the context, ask one short clarifying question.\n\n"
            f"QUESTION:\n{user_q}\n\nCONTEXT:\n{ctx_block}\n"
        )
        try:
            comp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.2,
                max_tokens=500,
                top_p=1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            answer = (comp.choices[0].message.content or "").strip()
            return ChatResponse(answer=answer, sources=build_sources(hits))
        except Exception:
            pass  # fall through to extractive fallback

    # No/weak context -> route:
    if oai is not None and is_definitional(user_q):
        # General concise definition (tie to EM when relevant)
        general_prompt = (
            "Provide a concise, plain-language definition suitable for a patient-facing website. "
            "If the term is 'PA', interpret as Physician Assistant (Physician Associate) unless clearly the state of Pennsylvania. "
            "Keep it to 2-4 sentences."
        )
        try:
            comp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.2,
                max_tokens=250,
                top_p=1,
                messages=[
                    {"role": "system", "content": general_prompt},
                    {"role": "user", "content": user_q},
                ],
            )
            answer = (comp.choices[0].message.content or "").strip()
            return ChatResponse(answer=answer, sources=[])
        except Exception:
            pass

    if needs_clarification(user_q):
        return ChatResponse(
            answer="Quick clarification: do you mean membership dues, eligibility, or something else?",
            sources=[],
        )

    # Extractive fallback if we had some context but LLM failed
    if contexts:
        best = contexts[0]
        answer = "Here is what I found in SEMPA materials:\n\n" + (best[:900] + ("..." if len(best) > 900 else ""))
        return ChatResponse(answer=answer, sources=build_sources(hits))

    # Final fallback
    return ChatResponse(
        answer="I do not have SEMPA-specific details on that. Can you share a little more about what you are looking for?",
        sources=[],
    )

# -------------------
# Tenant debug
# -------------------
@app.get("/debug/tenant/{client_id}")
async def debug_tenant(client_id: str):
    rag = load_tenant(client_id)
    return {
        "client_id": client_id,
        "has_index": rag.index is not None,
        "num_texts": len(rag.texts),
        "dim": rag.dim,
    }

# -------------------
# Minimal widget (Enter=send, Shift+Enter=newline) with transcript + source links
# -------------------
WIDGET_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SEMPA Chat</title>
  <style>
    :root { --pad:16px; --radius:12px; --muted:#f6f6f6; --border:#e5e5e5; --ink:#111; --hint:#666; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: var(--pad); color: var(--ink); }
    .card { border:1px solid var(--border); border-radius: var(--radius); padding: var(--pad); background:#fff; }
    .brand { display:flex; align-items:center; gap:8px; font-weight:600; margin-bottom:8px; }
    .brand img { height:20px; }
    #thread { max-height: 60vh; overflow:auto; padding-right:4px; }
    .msg { white-space: pre-wrap; margin: 0 0 12px 0; }
    .from { font-size:12px; color:var(--hint); margin-bottom:4px; }
    .sources { margin-top:6px; font-size:12px; color:#555; }
    .sources a { text-decoration:none; border-bottom:1px dotted #999; }
    .sources a:hover { border-bottom-style:solid; }
    textarea { width: 100%; resize: vertical; min-height: 90px; font: inherit; padding: 10px; border-radius: 10px; border:1px solid var(--border); box-sizing: border-box; }
    button { margin-top: 10px; padding: 10px 14px; border-radius: 10px; border:1px solid var(--border); background:#003366; color:#fff; cursor:pointer; }
    button[disabled] { opacity:.6; cursor: not-allowed; }
    .hint { color:var(--hint); font-size: 12px; margin-top:6px; }
  </style>
</head>
<body>
  <div class="card">
    <div class="brand">
      <img src="/static/sempa-mark-24.png" alt="SEMPA" onerror="this.style.display='none'">
      <div>SEMPA Chat</div>
    </div>
    <div id="thread" aria-live="polite"></div>
    <textarea id="q" placeholder="Type your question… (Enter to send, Shift+Enter for newline)"></textarea>
    <button id="ask">Ask</button>
    <div class="hint">Answers use SEMPA docs when available; otherwise brief general info or a clarifying question.</div>
  </div>

  <script>
    (function () {
      const qEl = document.getElementById('q');
      const btn = document.getElementById('ask');
      const thread = document.getElementById('thread');

      const params = new URLSearchParams(location.search);
      const clientId = params.get('client_id') || 'sempa';
      const KEY = 'sempaChat:' + clientId;

      const load = () => JSON.parse(sessionStorage.getItem(KEY) || '[]');
      const save = (t) => sessionStorage.setItem(KEY, JSON.stringify(t));
      let transcript = load();

      function linkifySources(sources) {
        if (!Array.isArray(sources) || !sources.length) return '';
        const parts = sources.map((s) => {
          if (typeof s === 'string') return '<a href="' + s + '" target="_blank" rel="noopener">' + s + '</a>';
          if (s && s.url) return '<a href="' + s.url + '" target="_blank" rel="noopener">' + (s.title || s.url) + '</a>';
          return '';
        }).filter(Boolean);
        return parts.length ? '<div class="sources">' + parts.join(' • ') + '</div>' : '';
      }

      function render() {
        thread.innerHTML = '';
        for (const turn of transcript) {
          const wrap = document.createElement('div');
          wrap.className = 'msg';
          const who = document.createElement('div');
          who.className = 'from';
          who.textContent = turn.role === 'user' ? 'You' : 'SEMPA';
          const body = document.createElement('div');
          body.textContent = turn.text || '';
          wrap.appendChild(who);
          wrap.appendChild(body);
          if (turn.role === 'assistant' && turn.sources) {
            const div = document.createElement('div');
            div.className = 'sources';
            div.innerHTML = linkifySources(turn.sources);
            if (div.innerHTML) wrap.appendChild(div);
          }
          thread.appendChild(wrap);
        }
        thread.scrollTop = thread.scrollHeight;
      }

      function push(role, text, sources) {
        transcript.push({ role, text, sources: sources || [] });
        save(transcript);
        render();
      }

      function postHeight() {
        try {
          const h = document.documentElement.scrollHeight;
          window.parent && window.parent.postMessage({ type: 'sempaWidgetSize', height: h }, '*');
        } catch(e) {}
      }

      async function ask() {
        const question = qEl.value.trim();
        if (!question) return;
        btn.disabled = true;
        push('user', question);
        qEl.value = '';
        try {
          const r = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, client_id: clientId })
          });
          if (!r.ok) throw new Error('HTTP ' + r.status);
          const data = await r.json();
          push('assistant', data.answer || '(no answer)', data.sources || []);
        } catch (e) {
          push('assistant', 'Sorry—there was a problem reaching the chat service.');
          console.error(e);
        } finally {
          btn.disabled = false;
          postHeight();
        }
      }

      btn.addEventListener('click', ask);
      qEl.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' && !ev.shiftKey) {
          ev.preventDefault();
          ask();
        } else if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') {
          ev.preventDefault();
          ask();
        }
      });

      // initial height post for iframe
      postHeight();
      render();
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

# -------------------
# Floating bubble script (/embed.js)
# -------------------
EMBED_JS = """
(function(){
  var thisScript = document.currentScript || (function(){var s=document.getElementsByTagName('script');return s[s.length-1];})();
  var base = new URL(thisScript.src).origin;
  var ds = thisScript.dataset || {};
  var clientId = ds.clientId || 'sempa';
  var title = ds.title || 'SEMPA Chat';
  var color = ds.color || '#003366';   // SEMPA navy
  var side = (ds.position || 'right').toLowerCase(); // 'right' or 'left'
  var z = ds.z || 2147483647;

  // --- Button (speech bubble icon or logo) ---
  var btn = document.createElement('button');
  btn.setAttribute('aria-label', 'open ' + title);
  btn.style.cssText = [
    'position:fixed',
    (side==='left'?'left:24px;':'right:24px;'),
    'bottom:24px',
    'width:56px','height:56px',
    'border-radius:999px',
    'border:none',
    'box-shadow:0 8px 20px rgba(0,0,0,.15)',
    'background:'+color,
    'color:#fff',
    'cursor:pointer',
    'z-index:'+z
  ].join(';');
  btn.innerHTML = '<img src=\"/static/sempa-mark-26.png\" alt=\"\" width=\"26\" height=\"26\" style=\"display:block;margin:auto;\" onerror=\"this.replaceWith(document.createElement(\\'span\\')).textContent=\\'💬\\'\">';

  // --- Panel container ---
  var panel = document.createElement('div');
  panel.style.cssText = [
    'position:fixed',
    (side==='left'?'left:24px;':'right:24px;'),
    'bottom:96px',
    'width:380px','max-width:95vw',
    'height:560px','max-height:85vh',
    'background:#fff',
    'border:1px solid #e5e5e5',
    'border-radius:16px',
    'box-shadow:0 12px 28px rgba(0,0,0,.18)',
    'overflow:hidden',
    'display:none',
    'z-index:'+z
  ].join(';');

  // --- Header ---
  var header = document.createElement('div');
  header.style.cssText = 'height:44px;display:flex;align-items:center;justify-content:space-between;padding:0 12px;border-bottom:1px solid #eee;background:'+color+';color:#fff;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-size:14px';
  var titleEl = document.createElement('div'); titleEl.textContent = title;
  var close = document.createElement('button'); close.textContent = '×';
  close.setAttribute('aria-label','close');
  close.style.cssText = 'border:none;background:transparent;font-size:18px;cursor:pointer;color:#fff;';
  header.appendChild(titleEl); header.appendChild(close);

  // --- Iframe ---
  var iframe = document.createElement('iframe');
  iframe.src = base + '/widget?client_id=' + encodeURIComponent(clientId);
  iframe.style.cssText = 'width:100%;height:calc(100% - 44px);border:0;display:block;background:#fff;';

  panel.appendChild(header); panel.appendChild(iframe);
  document.body.appendChild(panel);
  document.body.appendChild(btn);

  function openPanel(){ panel.style.display = 'block'; }
  function closePanel(){ panel.style.display = 'none'; }
  btn.addEventListener('click', function(){
    if (panel.style.display === 'block') { closePanel(); } else { openPanel(); }
  });
  close.addEventListener('click', closePanel);

  // Resize messages from widget
  window.addEventListener('message', function(ev){
    var d = ev.data || {};
    if (d && d.type === 'sempaWidgetSize' && typeof d.height === 'number') {
      var max = Math.min(window.innerHeight * 0.85, 720);
      var h = Math.max(300, Math.min(d.height + 120, max));
      panel.style.height = h + 'px';
    }
  });
})();
"""

@app.get("/embed.js")
def embed_js(client_id: str = Query(default="sempa", description="Default tenant id")):
    # Note: client_id from query is not required here because we allow script data attributes.
    return Response(content=EMBED_JS, media_type="application/javascript")



