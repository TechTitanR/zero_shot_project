# save_result.py
from transformers import pipeline
import json

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text = "Apple released a new iPhone with longer battery life."
labels = ["technology", "sports", "politics", "food"]

res = classifier(text, candidate_labels=labels)
out = {"text": text, "labels": res["labels"], "scores": res["scores"]}
with open("last_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

# also write CSV-friendly line
import csv
with open("results.csv", "a", newline='', encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow([text, res["labels"][0]] + [round(s,4) for s in res["scores"]])

print("Saved last_result.json and appended to results.csv")
