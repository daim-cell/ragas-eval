"""
Ingests a random subset of the FiQA corpus directly from HuggingFace into ChromaDB.
Bypasses the ./documents folder — documents are streamed from the dataset instead.

Usage:
    python ingest_fiqa.py              # ingest 5000 passages (default)
    python ingest_fiqa.py --n 2000     # ingest a custom number
"""

import argparse
import sys

import chromadb
from datasets import load_dataset
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PERSIST_DIR,
)

DEFAULT_N = 5_000
SEED = 42


def main(n: int) -> None:
    # --- 1. Load corpus from HuggingFace and sample n passages ---
    print(f"Loading FiQA corpus from HuggingFace (sampling {n} passages, seed={SEED})...")
    raw = load_dataset("vibrantlabsai/fiqa", "corpus", split="corpus")
    raw = raw.shuffle(seed=SEED).select(range(n))
    print(f"  Corpus rows selected: {len(raw)}")

    # --- 2. Check for existing index and prompt before overwriting ---
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    existing = collection.count()
    if existing > 0:
        answer = input(
            f"\nIndex already exists ({existing} chunks in '{COLLECTION_NAME}'). "
            "Rebuild? (y/n): "
        ).strip().lower()
        if answer != "y":
            print("Skipping rebuild. Exiting.")
            sys.exit(0)
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Existing collection deleted.")
        # Re-create after deletion
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # --- 3. Configure LlamaIndex to use local Ollama models ---
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # --- 4. Wrap each passage as a LlamaIndex Document ---
    # metadata["source"] lets query_pipeline() report which corpus doc was retrieved
    print("Preparing documents...")
    documents = [
        Document(
            text=row["doc"],
            metadata={"file_name": f"fiqa_corpus_{i:05d}", "source": "fiqa"},
        )
        for i, row in enumerate(raw)
    ]
    print(f"  Documents prepared: {len(documents)}")

    # --- 5. Build ChromaDB-backed index and embed all chunks ---
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print(f"\nEmbedding and indexing {len(documents)} documents "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    print("This may take several minutes — progress is shown per batch below.\n")

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    final_count = collection.count()
    print(f"\nDone. {final_count} chunks stored in ChromaDB at '{PERSIST_DIR}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest FiQA corpus into ChromaDB.")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of corpus passages to ingest (default: {DEFAULT_N})")
    args = parser.parse_args()
    main(args.n)
