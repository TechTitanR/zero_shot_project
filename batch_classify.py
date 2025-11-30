# batch_classify.py
from transformers import pipeline
import csv

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["technology","sports","politics","food"]

with open('assets/examples.txt','r',encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]

with open('batch_results.csv','w',newline='',encoding='utf-8') as out:
    w = csv.writer(out)
    w.writerow(["text","predicted_label","score_top","score_2","score_3","score_4"])
    for t in lines:
        res = classifier(t, candidate_labels=labels)
        row = [t, res['labels'][0]] + [round(s,4) for s in res['scores']]
        w.writerow(row)
        print(row)
