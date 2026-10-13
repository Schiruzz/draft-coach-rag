"""Connection settings, read from the environment (or from .env in local dev)."""
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env if present; real environment variables take precedence

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")