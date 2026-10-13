"""Liveness checks for the two databases."""
from pymongo import MongoClient
from qdrant_client import QdrantClient

from draft_coach.config import MONGO_URI, QDRANT_URL


def mongo_version() -> str:
    """Return the MongoDB server version; raise if the server does not answer."""
    # 3 s timeout: the pymongo default (30 s) makes a down server look like a hang
    with MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000) as client:
        return client.server_info()["version"]


def qdrant_version() -> str:
    """Return the Qdrant server version; raise if the server does not answer."""
    client = QdrantClient(url=QDRANT_URL, timeout=3)
    return client.info().version