"""Claim-verification F1 on SciFact using the NLI verifier (the same one used in agents.verify).
Download from https://github.com/allenai/scifact (data.tar.gz): need claims_dev.jsonl and corpus.jsonl.
Run: python -m eval.scifact_eval data/scifact/claims_dev.jsonl data/scifact/corpus.jsonl [--n 300]
Labels: SUPPORT / CONTRADICT / NEI. Premise = abstract sentences of the claim's cited docs (oracle retrieval),
so this isolates verifier quality. For end-to-end, swap the premise with index.search() results."""
import json, sys
import numpy as np
from sklearn.metrics import classification_report, f1_score
from agents import nli

THR = 0.5

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)

def gold_label(c):
    ev = c.get("evidence", {})
    if not ev: return "NEI"
    first = next(iter(ev.values()))[0]
    return "SUPPORT" if first["label"] == "SUPPORT" else "CONTRADICT"

def predict(claim, sentences):
    p = softmax(nli().predict([(s, claim) for s in sentences]))
    ent, con = p[:, 1].max(), p[:, 0].max()
    if con > ent and con > THR: return "CONTRADICT"
    if ent > THR: return "SUPPORT"
    return "NEI"

def main(claims_path, corpus_path, n=300):
    corpus = {}
    for l in open(corpus_path):
        d = json.loads(l); corpus[d["doc_id"]] = d["abstract"]
    claims = [json.loads(l) for l in open(claims_path)]
    claims = [c for c in claims if c.get("cited_doc_ids")][:n]
    y, yhat = [], []
    for c in claims:
        sents = [s for d in c["cited_doc_ids"] for s in corpus.get(int(d), [])]
        if not sents: continue
        y.append(gold_label(c)); yhat.append(predict(c["claim"], sents))
    print(classification_report(y, yhat, digits=3))
    print("Macro-F1:", round(f1_score(y, yhat, average="macro"), 3))

if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300
    main(sys.argv[1], sys.argv[2], n)
