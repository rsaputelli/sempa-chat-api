# app/tenants.py — robust single-tenant config (SEMPA)

from typing import Dict, Any

TENANT_CONFIG: Dict[str, Dict[str, Any]] = {
    "sempa": {
        "embedding_dir": "clients/sempa/embeddings_v2",
        "prompt_path": "clients/sempa/prompt.txt",
    }
}

# Use a set to avoid list/append vs set/add issues
ALLOWED_TENANTS = {"sempa"}
