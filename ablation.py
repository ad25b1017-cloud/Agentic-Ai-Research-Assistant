"""Retrieval baselines. Make eval/qa.jsonl lines like:
{"question": "...", "paper": "file.pdf", "page": 4}   (hand-label 20-30 questions; include hard ones!)
Run: python -m eval.ablation papers_folder eval/qa.jsonl
"""
import json, sys, glob, os
import numpy as np
from parser import parse_pdf
from index import HybridIndex

def main(folder, qa_path, k=5):
    idx = HybridIndex()
    for p in glob.glob(os.path.join(folder, "*.pdf")):
        idx.add(parse_pdf(p, os.path.basename(p), start_id=len(idx.chunks)))
    qa = [json.loads(l) for l in open(qa_path)]
    for mode in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        rec, mrr = [], []
        for ex in qa:
            hits = idx.search(ex["question"], top_k=k, mode=mode)
            rel = [h.chunk.paper == ex["paper"] and abs(h.chunk.page - ex["page"]) <= 1 for h in hits]
            rec.append(any(rel)); mrr.append(next((1 / (i + 1) for i, r in enumerate(rel) if r), 0))
        print(f"{mode:15s} Recall@{k}={np.mean(rec):.2f}  MRR={np.mean(mrr):.2f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
