# app.py
"""Simple CLI Zero-Shot Text Classifier using Hugging Face Transformers"""
from transformers import pipeline
import argparse, json

def main():
    parser = argparse.ArgumentParser(description="Zero-shot text classifier (CLI)")
    parser.add_argument("--text", type=str, help="Text to classify. If omitted, reads from stdin.")
    parser.add_argument("--labels", type=str, required=True,
                        help="Comma-separated candidate labels, e.g. 'technology,sports,politics'")
    parser.add_argument("--model", type=str, default="facebook/bart-large-mnli",
                        help="Zero-shot model name (default: facebook/bart-large-mnli)")
    args = parser.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if not args.text:
        print("Enter text (end with Ctrl+D on Linux/macOS or Ctrl+Z on Windows):")
        import sys
        args.text = sys.stdin.read().strip()

    classifier = pipeline("zero-shot-classification", model=args.model)
    result = classifier(args.text, candidate_labels=labels)

    print("\nInput text:")
    print(args.text)
    print("\nPredicted label:", result["labels"][0])
    print("All labels & scores:")
    for lab, score in zip(result["labels"], result["scores"]):
        print(f"  {lab:20s} {score:.4f}")

    out = {"text": args.text, "labels": result["labels"], "scores": result["scores"]}
    print("\nRaw output JSON:")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
