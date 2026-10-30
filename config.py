import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0"
)

REDIS_TTL = int(os.getenv("REDIS_TTL", "3600"))
BASE_DIR = Path(__file__).resolve().parent

QDRANT_PATH = BASE_DIR / "qdrant_data"
QDRANT_COLLECTION = "knowledge_base"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS"))