from pipeline import query_pipeline


def main() -> None:
    print("RAG Pipeline ready. Type your question or 'exit' to quit.")
    print("-" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye.")
            break

        result = query_pipeline(question)

        print(f"\nAnswer: {result['answer']}")

        unique_sources = sorted(set(result["source_nodes"]))
        if unique_sources:
            print("\nSources:")
            for src in unique_sources:
                print(f"  - {src}")


if __name__ == "__main__":
    main()
