"""
Downloads the FiQA test split from vibrantlabsai/fiqa, randomly selects 100 samples,
and writes them to eval/golden_dataset.jsonl as the golden evaluation set.

Each line: {"id": "fiqa_0000", "user_input": "<question>", "reference": "<answer>"}
"""

import json
import os

from datasets import load_dataset


OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.jsonl")


def main() -> None:
    print("Loading vibrantlabsai/fiqa (test split)...")
    dataset = load_dataset("vibrantlabsai/fiqa", "main", split="test")

    print(f"Total test samples: {len(dataset)}")

    # Reproducible 100-sample subset — seed=42 ensures the same rows every run
    dataset = dataset.shuffle(seed=42).select(range(100))

    print(f"Selected {len(dataset)} samples. Writing to {OUTPUT_PATH} ...")

    with open(OUTPUT_PATH, "w") as f:
        for i, row in enumerate(dataset):
            sample = {
                "id": f"fiqa_{i:04d}",
                "user_input": row["question"],
                # ground_truths is a list; first entry is used as the reference answer
                "reference": row["ground_truths"][0],
            }
            f.write(json.dumps(sample) + "\n")

    print(f"Done. Golden dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
