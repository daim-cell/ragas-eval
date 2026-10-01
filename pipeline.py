from typing import Any, Dict, List

import chromadb

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PERSIST_DIR,
    SYSTEM_PROMPT,
    TOP_K,
)


def _configure_settings() -> None:
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def build_index() -> VectorStoreIndex:
    # Point LlamaIndex at the local Ollama LLM and embedding model
    _configure_settings()

    # Load every file under DOCUMENTS_DIR; SimpleDirectoryReader handles .txt, .pdf, .md, .docx
    # Each document gets metadata["file_name"] set automatically by the reader
    documents = SimpleDirectoryReader(input_dir=DOCUMENTS_DIR, recursive=True).load_data()

    # Open (or create) a persistent ChromaDB collection; all vectors are stored in PERSIST_DIR
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # Wire the ChromaDB collection into LlamaIndex's storage layer
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index: chunks each document with SentenceSplitter, embeds every chunk,
    # and upserts the vectors into ChromaDB — PersistentClient writes to disk automatically
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )
    return index


def load_index() -> VectorStoreIndex:
    _configure_settings()

    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # If the collection is empty there is nothing to load — build from scratch instead
    if chroma_collection.count() == 0:
        print("No existing index found. Building from scratch...")
        return build_index()

    # from_vector_store re-creates the index handle in memory without re-embedding anything;
    # all retrieval operations are delegated to the already-populated ChromaDB collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def build_query_engine(index: VectorStoreIndex):
    # Inject SYSTEM_PROMPT as a system-role message so the model stays grounded in context
    qa_prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, "
                    "answer the query.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
            ),
        ]
    )
    return index.as_query_engine(similarity_top_k=TOP_K, text_qa_template=qa_prompt)


def query_pipeline(question: str) -> Dict[str, Any]:
    # Load (or lazily build) the index from the persistent ChromaDB store
    index = load_index()

    # Attach the query engine with custom grounding prompt and top-k retrieval
    query_engine = build_query_engine(index)

    # Execute the query — LlamaIndex retrieves TOP_K chunks then synthesises an answer
    response = query_engine.query(question)

    # Pull the plain answer string out of the response object
    answer = str(response)

    # Extract the raw chunk text for each retrieved node (used by RAGAS as "contexts")
    contexts: List[str] = [node.get_content() for node in response.source_nodes]

    # Extract the originating filename from each node's metadata (set by SimpleDirectoryReader)
    source_nodes: List[str] = [
        node.metadata.get("file_name", node.metadata.get("file_path", "unknown"))
        for node in response.source_nodes
    ]

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "source_nodes": source_nodes,
    }
