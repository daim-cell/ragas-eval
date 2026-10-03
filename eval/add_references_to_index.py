"""
Adds each golden-set reference answer to the existing ChromaDB index as its own document.

Why: the index holds a 5,000-document sample of the FiQA corpus, chosen independently of the golden
questions, so only ~11 of the 100 reference answers were retrievable. context_recall and
context_precision were therefore near zero for reasons unrelated to retrieval quality. Adding the
references (labelled fiqa_gold_XXXX) keeps the 5,000 existing documents as distractors while
guaranteeing every question has a findable answer.

Idempotent: references already in the index (matched by file_name) are skipped.
No full rebuild — new chunks are embedded and inserted into the existing collection.

Usage:
    python eval/add_references_to_index.py
"""

import os
import sys

# Add project root to path so pipeline/config are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, PERSIST_DIR
from pipeline import load_index

sys.path.insert(0, os.path.dirname(__file__))
from run_eval import GOLDEN_PATH, load_golden  # noqa: E402


def main() -> None:
    samples = load_golden(GOLDEN_PATH)

    collection = chromadb.PersistentClient(path=PERSIST_DIR).get_or_create_collection(COLLECTION_NAME)
    before = collection.count()

    docs = []
    skipped = 0
    for sample in samples:
        # sample id is fiqa_0007 -> fiqa_gold_0007
        name = sample["id"].replace("fiqa_", "fiqa_gold_")
        if collection.get(where={"file_name": name}, limit=1)["ids"]:
            continue
        # ~11% of references already exist in the corpus sample under a fiqa_corpus_* name; indexing
        # them again would put duplicate chunks in the top-k and depress context_precision.
        if collection.get(where_document={"$contains": sample["reference"][:50]}, limit=1)["ids"]:
            skipped += 1
            continue
        docs.append(Document(text=sample["reference"], metadata={"file_name": name, "source": "fiqa_gold"}))

    if not docs:
        print(f"All {len(samples)} references already indexed ({before} chunks). Nothing to do.")
        return

    print(f"Adding {len(docs)} reference documents to the index ({before} chunks currently); "
          f"{skipped} already present in the corpus.")
    index = load_index()  # also configures the Ollama embed model on Settings

    # Same splitter settings as pipeline.build_index so chunking is identical to the rest of the corpus
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    index.insert_nodes(nodes)

    print(f"Done. Index now has {collection.count()} chunks (+{collection.count() - before}).")


if __name__ == "__main__":
    main()
