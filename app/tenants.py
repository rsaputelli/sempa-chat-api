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
