SEMPA Multi-tenant Chat Backend (Milestone 1)

Included:
- /chat POST endpoint with {question, client_id}
- Tenant routing (only 'sempa' enabled)
- Domain allow-list (semsa.org + www.sempa.org)
- /healthz endpoint
- Render deployment config (render.yaml)

Local run:
1) pip install -r requirements.txt
2) uvicorn app.main:app --reload
3) curl example:
   curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -H "Origin: https://www.sempa.org" -d "{\"question\": \"How do I renew?\", \"client_id\": \"sempa\"}"

Render deploy:
- Push to GitHub and create a Web Service in Render.
- Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT

Embeddings:
- Copy SEMPA FAISS assets into clients/sempa/embeddings/
  expected files: index.faiss, embeddings.npy (optional), texts.txt or docs.jsonl
