from typing import Dict, List

# Allowed tenants (client_ids). Start with only 'sempa' enabled
ALLOWED_TENANTS: List[str] = ["sempa"]

# Map each tenant to its content/config folder (relative to project root)
TENANT_CONFIG: Dict[str, dict] = {
    "sempa": {
        "embedding_dir": "clients/sempa/embeddings",  # put FAISS index here
        "prompt_path": "clients/sempa/prompt.txt",
        "display_name": "SEMPA Assistant",
    }
}
# --- patched override: ensure SEMPA points to embeddings_v2 and is allowed ---
try:
    TENANT_CONFIG
except NameError:
    TENANT_CONFIG = {}

TENANT_CONFIG["sempa"] = {
    "embedding_dir": "clients/sempa/embeddings_v2",
    "prompt_path": "clients/sempa/prompt.txt",
}

try:
    ALLOWED_TENANTS
except NameError:
    ALLOWED_TENANTS = set()

# if someone used a dict earlier, coerce to set of keys
if isinstance(ALLOWED_TENANTS, dict):
    ALLOWED_TENANTS = set(ALLOWED_TENANTS.keys())

ALLOWED_TENANTS.add("sempa")
