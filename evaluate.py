# evaluate.py
"""Quick evaluator to run zero-shot classification on a dataset subset."""
from transformers import pipeline
from datasets import load_dataset
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ag_news", help="Hugging Face dataset name (default ag_news)")
    parser.add_argument("--split", default="test", help="Which split")
    parser.add_argument("--num", type=int, default=200, help="Number of examples to evaluate (small subset)")
    parser.add_argument("--model", default="facebook/bart-large-mnli")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    classifier = pipeline("zero-shot-classification", model=args.model)

    if args.dataset == "ag_news":
        labels = ["World", "Sports", "Business", "Science/Tech"]
        text_field = "text"
    else:
        labels = ["positive", "negative", "neutral"]
        text_field = "text"

    total, correct = 0, 0
    for item in tqdm(ds.select(range(min(args.num, len(ds))))):
        text = item[text_field]
        true_label_idx = item.get("label")
        if isinstance(true_label_idx, int) and true_label_idx < len(labels):
            true_label = labels[true_label_idx]
        else:
            true_label = str(true_label_idx)

        res = classifier(text, candidate_labels=labels)
        pred = res["labels"][0]
        if pred.lower() == true_label.lower():
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy (zero-shot) on {total} examples: {acc:.4f}")

if __name__ == "__main__":
    main()
