"""Names and ids shared by the scripts that write and read the vector index (Qdrant)."""
import hashlib
import uuid


def collection_name(method: str, model: str) -> str:
    """One collection per chunking method and embedding model: vectors of different models never mix."""
    return f"chunks__{method}__{model}"


def point_id(chunk_id: str) -> str:
    """Stable Qdrant id from the MongoDB chunk id: rerunning updates the same point, no duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def text_hash(text: str) -> str:
    """Fingerprint of the embedded text: a chunk is embedded again only if its text changes."""
    return hashlib.sha256(text.encode()).hexdigest()