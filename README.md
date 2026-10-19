# Local RAG Evaluation with LlamaIndex, Ollama, ChromaDB, and RAGAS

A fully local retrieval-augmented generation (RAG) pipeline for indexing
documents, answering grounded questions, and measuring retrieval and answer
quality. The project uses:

- [LlamaIndex](https://www.llamaindex.ai/) for ingestion, retrieval, and answer synthesis
- [Ollama](https://ollama.com/) for local generation and embeddings
- [ChromaDB](https://www.trychroma.com/) for persistent vector storage
- [RAGAS](https://docs.ragas.io/) for evaluation
- [FiQA](https://huggingface.co/datasets/vibrantlabsai/fiqa) as the included financial QA benchmark

Both the RAG model and RAGAS judge run through Ollama, so document contents,
questions, and generated answers do not need to be sent to a hosted model API.

## How it works

```text
Local documents or FiQA corpus
             │
             ▼
SentenceSplitter → Ollama embeddings → persistent ChromaDB collection
                                                 │
Question → top-k retrieval → Ollama LLM → answer + contexts + source IDs
                                                 │
Golden reference ────────────────────────────────┴→ RAGAS metrics
```

The default pipeline splits documents into 512-token chunks with 50-token
overlap, retrieves the three most similar chunks, and asks `qwen2.5:7b` to
answer from that context. A strict grounding prompt is used for normal queries;
evaluation runs can also select a more helpful relaxed prompt for comparison.

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/download), installed and running
- Enough disk space and memory for the models, vector index, and evaluation workload

Pull the default generation/judge and embedding models:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

If Ollama is not already running as a background service, start it with:

```bash
ollama serve
```

## Installation

```bash
git clone <repository-url>
cd llamaindex-rag-eval

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
All commands below assume the virtual environment is active and are run from
the repository root.

## Quick start with your own documents

1. Create the input directory and add documents:

   ```bash
   mkdir -p documents
   cp /path/to/your/files/* documents/
   ```

   `SimpleDirectoryReader` supports common formats including plain text,
   Markdown, PDF, and DOCX. The ingestion command expects at least one file at
   the top level of `documents/`; the reader then processes it recursively.

2. Build the persistent vector index:

   ```bash
   python ingest.py
   ```

3. Start the interactive question-answering loop:

   ```bash
   python main.py
   ```

Enter a question at the prompt, or type `exit` to quit. Each response includes
the identifiers of the retrieved source documents. If an index already exists,
`ingest.py` asks before deleting and rebuilding it.

The vector index is stored locally in `./store`. If no populated index exists
when a query starts, the pipeline attempts to build one from `./documents`.

## Using the FiQA corpus

For a reproducible financial QA experiment, index a seeded random sample of the
FiQA corpus directly from Hugging Face:

```bash
# Index the default 5,000 passages
python ingest_fiqa.py

# Or choose a different sample size
python ingest_fiqa.py --n 2000
```

The corpus is shuffled with seed `42`. After ingestion, run `python main.py`
and query it normally.

Local-document and FiQA ingestion write to the same ChromaDB collection. They
are alternative data sources: both scripts ask before replacing an existing
populated collection.

## Running an evaluation

The repository includes a 100-question golden FiQA dataset and a complete
RAGAS evaluation runner. A typical experiment is:

```bash
# 1. Build the FiQA retrieval corpus
python ingest_fiqa.py

# 2. Optional: add missing golden reference answers to the corpus
python eval/add_references_to_index.py

# 3. Run a small smoke test
python eval/run_eval.py --limit 5

# 4. Run the full 100-question baseline
python eval/run_eval.py
```

`add_references_to_index.py` is idempotent. It keeps the sampled corpus as
distractors, skips references already present, and inserts each missing
reference as `fiqa_gold_XXXX`. This creates a controlled retrieval experiment
where every question has an answer-bearing document available. Omit this step
when you want to measure the end-to-end coverage of the sampled corpus itself.

The evaluator reports four metrics:

| Metric | What it measures |
| --- | --- |
| `faithfulness` | Whether the answer is supported by retrieved context |
| `answer_relevancy` | Whether the answer addresses the question |
| `context_precision` | Whether retrieved chunks are useful and well ranked |
| `context_recall` | Whether retrieved chunks cover the reference answer |

### Evaluation options

Evaluate only selected metrics from previously cached pipeline answers:

```bash
python eval/run_eval.py --metrics faithfulness answer_relevancy
```

Compare retrieval depths and prompt strategies:

```bash
python eval/run_eval.py --top-k 10
python eval/run_eval.py --prompt relaxed
python eval/run_eval.py --top-k 10 --prompt relaxed
```

Use `--tag <name>` to give a custom experiment its own cache and results file:

```bash
python eval/run_eval.py --top-k 5 --prompt relaxed --tag candidate_a
```

Run `python eval/run_eval.py --help` for the full command reference.

### Evaluation outputs and caching

- The baseline writes detailed per-sample scores, aggregate scores, successful
  judge-call counts, and configuration metadata to `eval/results.json`.
- Modified, partial, or tagged runs write distinct filenames such as
  `eval/results_topk10_relaxed.json` or `eval/results_limit5.json`.
- Full-run model answers are cached in `eval/pipeline_cache.json`; experiment
  variants get their own tagged cache. Delete the relevant cache to regenerate
  answers after changing the pipeline.
- `--limit` runs are not cached, so a smoke test cannot contaminate a later
  full run.

The evaluator runs one local judge request at a time to avoid overwhelming
Ollama on memory-constrained machines. A full run can therefore take a long
time. Failed or timed-out judge calls appear as unscored samples; check the
reported count for each aggregate rather than reading the mean alone.

The project uses a calibrated answer-relevancy prompt designed for the local
7B judge. Its answer-relevancy scores are not directly comparable with scores
produced by RAGAS's stock prompt.

### Regenerating the golden dataset

`eval/golden_dataset.jsonl` contains 100 records with this shape:

```json
{"id": "fiqa_0000", "user_input": "When to sell a stock?", "reference": "..."}
```

Regenerate the same seeded sample with:

```bash
python eval/prepare_golden_dataset.py
```

### Experiment report

`eval/experiment_report.ipynb` provides a reproducible analysis of completed
runs, including scorecards, question-level comparisons, confidence intervals,
latency, and the effects of the calibrated judge prompt, relaxed generation
prompt, and larger retrieval depth.

The notebook reads curated result JSON files from `eval/results/` and runtime
logs from `eval/logs/`. Open it in a Jupyter-compatible environment after
placing the runs you want to compare in those directories; its `FILES` mapping
lists the expected filenames and can be adjusted for new experiments.

## Programmatic usage

For a single query:

```python
from pipeline import query_pipeline

result = query_pipeline("When should an investor sell a stock?")

print(result["answer"])
print(result["source_nodes"])
```

`query_pipeline()` returns:

| Key | Description |
| --- | --- |
| `question` | Original question |
| `answer` | Generated answer |
| `contexts` | Text from each retrieved chunk |
| `source_nodes` | Source filename or FiQA document identifier for each chunk |

For a batch workload, load the index and build the query engine once:

```python
from pipeline import build_query_engine, load_index, run_query

index = load_index()
engine = build_query_engine(index, top_k=5, prompt="relaxed")

results = [run_query(engine, question) for question in questions]
```

Supported prompt names are `strict` and `relaxed`.

## Configuration

Runtime settings live in `config.py`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `qwen2.5:7b` | Answer-generation model |
| `JUDGE_MODEL` | `qwen2.5:7b` | RAGAS judge model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding and similarity model |
| `OLLAMA_TIMEOUT` | `300` seconds | Generation request timeout |
| `JUDGE_TIMEOUT` | `600` seconds | Per-judge-call timeout |
| `LLM_CONTEXT_WINDOW` | `4096` | Generation model context window |
| `CHUNK_SIZE` | `512` | Maximum chunk size |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K` | `3` | Default number of retrieved chunks |
| `DOCUMENTS_DIR` | `./documents` | Local document input directory |
| `PERSIST_DIR` | `./store` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `rag_eval_collection` | ChromaDB collection name |
| `PROMPT_VARIANTS` | `strict`, `relaxed` | Available answer-generation prompts |

Rebuild the index after changing the embedding model, chunk settings, or
source documents so the stored vectors match the configuration. Prompt,
generation-model, and retrieval-depth changes do not require re-embedding, but
evaluation caches should be regenerated when they affect model answers.

## Project structure

```text
.
├── config.py                        # Models, paths, prompts, and retrieval settings
├── ingest.py                        # Index local files from ./documents
├── ingest_fiqa.py                   # Download and index a FiQA corpus sample
├── main.py                          # Interactive command-line interface
├── pipeline.py                      # Index construction, loading, and querying
├── requirements.txt
└── eval/
    ├── add_references_to_index.py   # Add missing golden answers to the index
    ├── experiment_report.ipynb       # Analyze and visualize experiment results
    ├── golden_dataset.jsonl          # 100 reference QA examples
    ├── prepare_golden_dataset.py     # Rebuild the reference dataset
    └── run_eval.py                   # Generate answers and run RAGAS metrics
```

Generated Chroma data, answer caches, and evaluation result files are local
runtime artifacts and should not be committed.

## Troubleshooting

**Cannot connect to Ollama**

Confirm that Ollama is running and that `OLLAMA_BASE_URL` matches its address.

**Model not found**

Pull the models named in `config.py` with `ollama pull <model-name>`, or update
the configuration to use models already installed locally.

**No files found in `./documents`**

Place at least one document directly inside that directory before running
`python ingest.py`, or use `python ingest_fiqa.py` instead.

**Queries or evaluation are slow**

Initial model loading and embedding can take time, and each RAGAS metric may
require judge-model inference. Start with `--limit 5` or a single metric before
launching a full evaluation.

**Evaluation keeps using old answers**

Delete `eval/pipeline_cache.json` or the relevant tagged cache. The cache is
deliberate so judge failures can be retried without repeating generation.

**A metric has fewer scored samples than the run total**

One or more judge calls timed out or returned invalid output. The aggregate
excludes those samples; inspect the count in the result file and retry the
metric or use a judge model that follows the expected JSON schema reliably.

**Answers are missing expected information**

The strict prompt deliberately refuses when retrieved context does not contain
the answer. Confirm that the content was indexed, inspect the returned sources,
or compare a larger `--top-k` value and the `relaxed` prompt.
