# Local RAG with LlamaIndex, Ollama, and ChromaDB

A local retrieval-augmented generation (RAG) pipeline built with
[LlamaIndex](https://www.llamaindex.ai/), [Ollama](https://ollama.com/), and
[ChromaDB](https://www.trychroma.com/). It can index either your own documents
or a reproducible sample of the FiQA financial question-answering corpus, then
answer questions using only the retrieved context.

The default setup runs both the language model and embedding model through
Ollama, so document contents and queries do not need to be sent to a hosted
model API.

## How it works

```text
Documents or FiQA corpus
        │
        ▼
SentenceSplitter → Ollama embeddings → persistent ChromaDB collection
                                              │
Question → top-k retrieval → Ollama LLM → grounded answer + sources
```

By default, documents are split into 512-token chunks with a 50-token overlap.
The five most similar chunks are passed to `qwen2.5:7b`, along with a prompt
that instructs the model to answer only from the supplied context.

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/download), installed and running
- Enough disk space and memory for the selected Ollama models and vector index

Pull the default models before starting:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

If Ollama is not already running as a background service, start it with:

```bash
ollama serve
```

## Installation

Clone the repository, create a virtual environment, and install the Python
dependencies:

```bash
git clone <repository-url>
cd llamaindex-rag-eval

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Quick start with your own documents

1. Create the input directory and add documents to it:

   ```bash
   mkdir -p documents
   cp /path/to/your/files/* documents/
   ```

   `SimpleDirectoryReader` supports common formats such as plain text,
   Markdown, PDF, and DOCX. The ingestion command expects at least one file at
   the top level of `documents/`; the reader itself processes the directory
   recursively.

2. Build the persistent vector index:

   ```bash
   python ingest.py
   ```

3. Start the interactive question-answering loop:

   ```bash
   python main.py
   ```

   Enter a question at the prompt, or type `exit` to quit. Each response
   includes the filenames of the retrieved source documents.

If an index already exists, `ingest.py` asks before deleting and rebuilding
the current collection. The index is stored locally in `./store`.

## Using the FiQA corpus

To experiment with financial question answering without supplying local
documents, ingest a deterministic random sample of the
[`vibrantlabsai/fiqa`](https://huggingface.co/datasets/vibrantlabsai/fiqa)
corpus from Hugging Face:

```bash
# Ingest the default 5,000 passages
python ingest_fiqa.py

# Or choose a different sample size
python ingest_fiqa.py --n 2000
```

The sample is shuffled with seed `42` for reproducibility. After ingestion,
run `python main.py` and query the index normally.

Local-document ingestion and FiQA ingestion use the same ChromaDB collection.
They are alternative data sources: if a collection already contains data, the
ingestion script prompts before replacing it.

## Programmatic usage

The pipeline can also be called from Python:

```python
from pipeline import query_pipeline

result = query_pipeline("When should an investor sell a stock?")

print(result["answer"])
print(result["source_nodes"])
```

`query_pipeline()` returns a dictionary with:

| Key | Description |
| --- | --- |
| `question` | The original question |
| `answer` | The generated answer |
| `contexts` | Text from each retrieved chunk |
| `source_nodes` | Source filename or FiQA document identifier for each chunk |

The `contexts` field is suitable for passing to RAG evaluation tooling such as
RAGAS.

## Evaluation dataset

The repository includes `eval/golden_dataset.jsonl`, a reproducible set of 100
FiQA test examples. Each JSON Lines record has this shape:

```json
{"id": "fiqa_0000", "user_input": "When to sell a stock?", "reference": "..."}
```

Regenerate the file with:

```bash
python eval/prepare_golden_dataset.py
```

This downloads the FiQA test split, shuffles it with seed `42`, and selects the
first 100 examples. The project includes the dataset preparation and RAGAS
dependency, but it does not currently include a script that runs evaluation
metrics.

## Configuration

Runtime settings live in `config.py`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `qwen2.5:7b` | Answer-generation model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `512` | Maximum chunk size |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `DOCUMENTS_DIR` | `./documents` | Local document input directory |
| `PERSIST_DIR` | `./store` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `rag_eval_collection` | ChromaDB collection name |

After changing the embedding model, chunking settings, or source documents,
rebuild the index so stored vectors match the new configuration.

## Project structure

```text
.
├── config.py                       # Models, paths, retrieval, and prompt settings
├── ingest.py                       # Index local files from ./documents
├── ingest_fiqa.py                  # Download and index a FiQA corpus sample
├── main.py                         # Interactive command-line interface
├── pipeline.py                     # Index construction, loading, and querying
├── requirements.txt
└── eval/
    ├── golden_dataset.jsonl         # 100 reference QA examples
    └── prepare_golden_dataset.py    # Rebuild the reference dataset
```

## Troubleshooting

**Cannot connect to Ollama**

Confirm that Ollama is running and that `OLLAMA_BASE_URL` matches its address.

**Model not found**

Pull the models named in `config.py` with `ollama pull <model-name>`, or change
the configuration to models already installed locally.

**No files found in `./documents`**

Create the directory and place at least one document directly inside it before
running `python ingest.py`. Alternatively, use `ingest_fiqa.py`.

**Queries are slow on first use**

Initial model loading and document embedding can take time. Later queries reuse
the persisted ChromaDB index and do not re-embed documents.

**Answers are missing expected information**

The assistant is deliberately restricted to retrieved context. Check that the
relevant content was indexed, then consider tuning `TOP_K`, `CHUNK_SIZE`, or
`CHUNK_OVERLAP` and rebuilding the index.
