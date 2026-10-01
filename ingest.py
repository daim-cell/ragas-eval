import os
import sys

import chromadb

from config import COLLECTION_NAME, DOCUMENTS_DIR, PERSIST_DIR
from pipeline import build_index


def main() -> None:
    if not os.path.isdir(DOCUMENTS_DIR):
        print(f"Error: DOCUMENTS_DIR '{DOCUMENTS_DIR}' does not exist.")
        print("Create the directory and add documents before running ingest.")
        sys.exit(1)

    doc_files = [
        f for f in os.listdir(DOCUMENTS_DIR)
        if os.path.isfile(os.path.join(DOCUMENTS_DIR, f))
    ]
    if not doc_files:
        print(f"No files found in '{DOCUMENTS_DIR}'. Add documents first.")
        sys.exit(1)

    print(f"Found {len(doc_files)} file(s) in '{DOCUMENTS_DIR}':")
    for f in sorted(doc_files):
        print(f"  - {f}")

    # Check whether a ChromaDB collection with data already exists
    try:
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        existing_count = collection.count()
    except Exception:
        existing_count = 0

    if existing_count > 0:
        answer = input(
            f"\nIndex already exists ({existing_count} chunks in '{COLLECTION_NAME}'). "
            "Rebuild? (y/n): "
        ).strip().lower()
        if answer != "y":
            print("Skipping rebuild. Exiting.")
            sys.exit(0)
        # Drop the old collection so build_index starts clean
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Existing collection deleted.")

    print("\nBuilding index...")
    build_index()
    print(f"\nDone. Index persisted to '{PERSIST_DIR}'.")


if __name__ == "__main__":
    main()
