OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"
# RAGAS judge. Kept separate from LLM_MODEL so it can be swapped independently.
# llama3.2:3b was tried and does NOT work: RAGAS prompts embed a JSON Schema and the 3B model echoes the
# schema (or the prompt) back instead of an instance of it, so most judge calls fail to parse (NaN).
# qwen2.5:7b follows the schema (~90% of calls parse) but is slow: ~90s/call on an 8 GB machine.
JUDGE_MODEL = "qwen2.5:7b"
JUDGE_TIMEOUT = 600.0  # seconds per judge call; context_recall prompts are the slowest and hit 300s before
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3
OLLAMA_TIMEOUT = 300.0  # seconds; 5 min ceiling — qwen2.5:7b on long contexts can be slow
# Must be set explicitly: LlamaIndex otherwise sends the model's max (32768 for qwen2.5:7b) as num_ctx,
# whose KV cache pushes a 7B model into swap on an 8 GB machine. TOP_K * CHUNK_SIZE + prompt fits in 4096.
LLM_CONTEXT_WINDOW = 4096

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
