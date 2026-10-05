"""
Runs the RAG pipeline over the 100-sample golden dataset and evaluates with RAGAS.

Metrics scored:
  - faithfulness       : is the answer grounded in retrieved contexts?
  - answer_relevancy   : is the answer relevant to the question?
  - context_precision  : are retrieved chunks actually useful?
  - context_recall     : do retrieved chunks cover the reference answer?

Usage:
    python eval/run_eval.py              # all questions
    python eval/run_eval.py --limit 10   # first 10 questions only
    python eval/run_eval.py --metrics answer_relevancy   # re-score one metric from the cached answers
    python eval/run_eval.py --top-k 10   # experiment: retrieve 10 chunks (own cache + results, tag topk10)
    python eval/run_eval.py --top-k 10 --prompt relaxed   # tag topk10_relaxed

Output:
    eval/results.json                    — per-sample scores + aggregate summary + the config used
    eval/results_<tag>_<metrics>_limit<N>.json — same, for a --tag/--top-k, --metrics and/or --limit run
                                           (leaves results.json untouched)

Note: answer_relevancy uses CalibratedResponseRelevancePrompt (below), not the stock RAGAS prompt.
"""

import argparse
import json
import os
import sys
import time

# Add project root to path so pipeline/config are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
# evaluate() only accepts legacy Metric instances; ragas.metrics.collections holds the
# new BaseMetric classes, which it rejects with "All metrics must be initialised metric objects"
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama as LangchainOllama
from langchain_community.embeddings import OllamaEmbeddings

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    JUDGE_MODEL,
    JUDGE_TIMEOUT,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PROMPT_VARIANTS,
    TOP_K,
)

# Private module: RAGAS 0.4.x doesn't re-export its prompt classes from ragas.metrics.
from ragas.metrics._answer_relevance import (
    ResponseRelevanceInput,
    ResponseRelevanceOutput,
    ResponseRelevancePrompt,
)


class CalibratedResponseRelevancePrompt(ResponseRelevancePrompt):
    """RAGAS's stock answer_relevancy prompt, recalibrated for a 7B judge.

    The stock prompt defines noncommittal as "evasive, vague, or ambiguous" and shows only two examples
    (a clean one-liner and a bare "I don't know"). The judge never sees the question, so a 7B model
    reads any answer containing a caveat ("Yes, but it isn't free") as vague and flags it
    noncommittal=1, which makes RAGAS multiply the score by 0. In the first full run 24 of 100 clearly
    committed answers scored exactly 0 for this reason. The examples below are generic on purpose —
    none come from the golden dataset — so the calibration isn't fitted to the answers being scored.

    Scores from this prompt are NOT comparable with ones from the stock prompt.
    """

    instruction = (
        "Generate a question for the given answer, and identify whether the answer is noncommittal.\n"
        "Set noncommittal to 1 ONLY if the answer declines to answer: it says it does not know, cannot "
        "answer, or that the needed information is not available or not provided, or it gives no "
        "substantive content.\n"
        "Set noncommittal to 0 if the answer gives substantive content, even when it includes caveats, "
        "risks, conditions or opinions (for example \"Yes, but it costs money\" or \"It depends on X, "
        "but generally you should Y\"). An answer that starts with \"Based on the context\" and then "
        "gives concrete information is committal (0)."
    )
    examples = [
        *ResponseRelevancePrompt.examples,
        (
            ResponseRelevanceInput(
                response="Yes, you can usually get a refund within 30 days, but the shipping fee is not refunded."
            ),
            ResponseRelevanceOutput(question="Can you get a refund on something you bought?", noncommittal=0),
        ),
        (
            ResponseRelevanceInput(
                response=(
                    "The context does not provide information about the capital of Australia. "
                    "It only mentions that Sydney is a large city."
                )
            ),
            ResponseRelevanceOutput(question="What is the capital of Australia?", noncommittal=1),
        ),
        (
            ResponseRelevanceInput(
                response=(
                    "Based on the context, preparing for a marathon takes months of training, a gradual "
                    "increase in weekly mileage and regular rest days to avoid injury."
                )
            ),
            ResponseRelevanceOutput(question="How should someone prepare to run a marathon?", noncommittal=0),
        ),
    ]


answer_relevancy.question_generation = CalibratedResponseRelevancePrompt()

ALL_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "pipeline_cache.json")


def load_golden(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline_over_golden(samples: list[dict], top_k: int = TOP_K, prompt: str = "strict") -> list[dict]:
    from pipeline import build_query_engine, load_index, run_query

    # Build index and query engine ONCE — reused for all 100 questions.
    # Previously query_pipeline() did this inside the loop, causing a fresh
    # ChromaDB connection + Ollama client init on every single question.
    print(f"  Initialising index and query engine (top_k={top_k}, prompt={prompt})...")
    index = load_index()
    query_engine = build_query_engine(index, top_k=top_k, prompt=prompt)
    print("  Ready.\n")

    results = []
    total = len(samples)
    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{total}] {sample['user_input'][:72]}...")
        t0 = time.time()
        result = run_query(query_engine, sample["user_input"])
        elapsed = time.time() - t0
        results.append({
            "id": sample["id"],
            "user_input": sample["user_input"],
            "reference": sample["reference"],
            "response": result["answer"],
            # RAGAS expects retrieved_contexts as a list of strings
            "retrieved_contexts": result["contexts"],
            "source_nodes": result["source_nodes"],
            "latency_s": round(elapsed, 2),
        })
        print(f"         answer: {result['answer'][:80]}...")
        print(f"         latency: {elapsed:.1f}s | chunks retrieved: {len(result['contexts'])}")
    return results


def build_ragas_dataset(results: list[dict]) -> Dataset:
    # RAGAS 0.4.x single-turn schema:
    #   user_input, response, retrieved_contexts, reference
    return Dataset.from_dict({
        "user_input":          [r["user_input"] for r in results],
        "response":            [r["response"] for r in results],
        "retrieved_contexts":  [r["retrieved_contexts"] for r in results],
        "reference":           [r["reference"] for r in results],
    })


def build_judge():
    # Every RAGAS metric prompt expects a JSON reply. Unconstrained, qwen sometimes answers in prose
    # ("Based on the provided documents, I don't know.") -> OutputParserException -> NaN score.
    # format="json" makes Ollama constrain sampling to valid JSON.
    # num_ctx: judge prompts reach ~3.6k tokens, which overflows Ollama's default 4096 once the
    #          reply is added; 8192 costs ~0.5 GB of KV cache (vs 1.8 GB at the model's 32768 max).
    # num_predict: caps runaway generations that would otherwise sit until the timeout. 2048 because
    #              context_recall lists every reference sentence with a reason; a cut-off reply is
    #              invalid JSON.
    judge_llm = LangchainLLMWrapper(
        LangchainOllama(
            model=JUDGE_MODEL,
            base_url=OLLAMA_BASE_URL,
            timeout=JUDGE_TIMEOUT,
            format="json",
            num_ctx=8192,
            num_predict=2048,
            temperature=0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    )
    return judge_llm, judge_embeddings


def main(
    limit: int | None = None,
    metric_names: list[str] | None = None,
    top_k: int | None = None,
    tag: str | None = None,
    prompt: str = "strict",
) -> None:
    metric_names = metric_names or list(ALL_METRICS)
    # Any non-baseline setting without a tag would read/overwrite the baseline cache, so it gets one
    # derived from the settings, e.g. "topk10", "relaxed" or "topk10_relaxed".
    if tag is None:
        parts = ([f"topk{top_k}"] if top_k is not None else []) + ([prompt] if prompt != "strict" else [])
        tag = "_".join(parts) or None
    effective_top_k = top_k if top_k is not None else TOP_K
    cache_path = CACHE_PATH.replace(".json", f"_{tag}.json") if tag else CACHE_PATH
    if not os.path.exists(GOLDEN_PATH):
        print(f"Golden dataset not found at {GOLDEN_PATH}")
        print("Run: python eval/prepare_golden_dataset.py")
        sys.exit(1)

    samples = load_golden(GOLDEN_PATH)
    if limit is not None:
        samples = samples[:limit]
    print(f"Loaded {len(samples)} samples from golden dataset.\n")

    # --- Step 1: Run the RAG pipeline over all 100 questions ---
    print("=" * 60)
    print("Step 1: Running RAG pipeline over golden questions...")
    print("=" * 60)
    # The pipeline step takes ~30 min; cache it so a failure in scoring doesn't throw it away.
    # Delete the cache file (eval/pipeline_cache[_<tag>].json) to force a fresh run.
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            results = json.load(f)[: len(samples)]
        print(f"Loaded {len(results)} cached pipeline results from {cache_path}")
    else:
        results = run_pipeline_over_golden(samples, top_k=effective_top_k, prompt=prompt)
        # Never cache a --limit run: a later full run would load only the first N answers
        if limit is None:
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)
    print(f"\nPipeline complete. {len(results)} answers generated.\n")

    # --- Step 2: Build RAGAS dataset ---
    ragas_dataset = build_ragas_dataset(results)

    # --- Step 3: Configure RAGAS to use local Ollama instead of OpenAI ---
    # RAGAS needs an LLM for faithfulness/answer_relevancy scoring
    # and an embedding model for answer_relevancy similarity
    print("=" * 60)
    print(f"Step 2: Running RAGAS evaluation ({', '.join(metric_names)}) with {JUDGE_MODEL} as judge...")
    print("=" * 60)

    judge_llm, judge_embeddings = build_judge()
    metrics = [ALL_METRICS[name] for name in metric_names]

    # RAGAS defaults (16 workers, 180s timeout) overwhelm a single local Ollama model on 8 GB:
    # requests queue up behind each other and time out. Run one judge call at a time.
    scores = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(timeout=int(JUDGE_TIMEOUT), max_workers=1),
    )

    # --- Step 4: Save results ---
    # EvaluationResult has no dict interface in ragas 0.4.x; everything goes through the DataFrame.
    # mean() skips NaN, so a metric job that failed on one sample doesn't zero the aggregate.
    scores_df = scores.to_pandas()
    aggregate = {m: round(float(scores_df[m].mean()), 4) for m in metric_names if m in scores_df.columns}
    # How many samples each aggregate is really based on. A NaN is a failed judge call (bad JSON or
    # timeout), not a zero — an average over the survivors is biased and must be read with this count.
    scored = {m: int(scores_df[m].notna().sum()) for m in aggregate}

    # Attach per-sample scores
    for i, row in enumerate(results):
        for metric_name in aggregate:
            row[metric_name] = round(float(scores_df.iloc[i][metric_name]), 4)

    output = {
        "aggregate": aggregate,
        "scored": {"n_samples": len(results), **scored},
        # The settings that produced these numbers, so a results file is self-describing
        "config": {
            "top_k": effective_top_k,
            "prompt": prompt,
            "system_prompt": PROMPT_VARIANTS[prompt]["system"],
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "llm_model": LLM_MODEL,
            "embed_model": EMBED_MODEL,
            "judge_model": JUDGE_MODEL,
            "tag": tag,
        },
        "samples": results,
    }
    # A tagged / partial run (--tag, --limit, --metrics) gets its own file so it can't clobber results.json
    suffix = f"_{tag}" if tag else ""
    if set(metric_names) != set(ALL_METRICS):
        suffix += "_" + "+".join(metric_names)
    if limit is not None:
        suffix += f"_limit{limit}"
    results_path = RESULTS_PATH.replace(".json", f"{suffix}.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Step 5: Print summary ---
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Results")
    print("=" * 60)
    for metric, score in aggregate.items():
        print(f"  {metric:<25} {score:.4f}   ({scored[metric]}/{len(results)} samples scored)")
    print("=" * 60)
    failed = {m: len(results) - n for m, n in scored.items() if n < len(results)}
    if failed:
        print("WARNING: judge calls failed (invalid JSON / timeout), so these averages exclude samples:")
        for m, n in failed.items():
            print(f"  {m}: {n} sample(s) unscored")
        print("Treat the averages as unreliable; use a judge that follows the RAGAS output schema.")
    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline + RAGAS eval over the golden dataset.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="only evaluate the first N golden questions (default: all)")
    parser.add_argument("--metrics", nargs="+", choices=list(ALL_METRICS), default=None, metavar="METRIC",
                        help=f"only score these metrics (default: all). Choices: {', '.join(ALL_METRICS)}")
    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help=f"retrieve K chunks per question instead of config.TOP_K ({TOP_K}); "
                             "results/cache are tagged topk<K> unless --tag is given")
    parser.add_argument("--prompt", choices=list(PROMPT_VARIANTS), default="strict",
                        help="answer-generation prompt: 'strict' (baseline, refuses unless the context "
                             "states the answer) or 'relaxed' (answers from partial context)")
    parser.add_argument("--tag", default=None,
                        help="label for this configuration; gives it its own cache and results files "
                             "(default: derived from --top-k / --prompt)")
    args = parser.parse_args()
    main(limit=args.limit, metric_names=args.metrics, top_k=args.top_k, tag=args.tag, prompt=args.prompt)
