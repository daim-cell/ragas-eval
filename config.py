OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

DOCUMENTS_DIR = "./documents"
PERSIST_DIR = "./store"
COLLECTION_NAME = "rag_eval_collection"

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer ONLY using the information provided in the context below. "
    "Do not use any prior knowledge, training data, or outside information. "
    "If the answer cannot be found in the provided context, respond with exactly: "
    "'I don't know based on the provided documents.' "
    "Never guess, fabricate, or extrapolate beyond what the context explicitly states."
)
