"""End-to-end eval: abstention accuracy, citation precision, unsupported-claim rate.
eval/qa_questions.jsonl lines: {"question": "...", "answerable": true}  (include ~30% unanswerable ones)
Run: python -m eval.qa_eval papers_folder eval/qa_questions.jsonl"""
import glob, json, os, sys
import numpy as np
from parser import parse_pdf
from index import HybridIndex
import agents

def main(folder, path):
    idx = HybridIndex()
    for p in glob.glob(os.path.join(folder, "*.pdf")):
        idx.add(parse_pdf(p, os.path.basename(p), start_id=len(idx.chunks)))
    qa = [json.loads(l) for l in open(path)]
    abst_ok, sup, cited, unsup = [], 0, 0, 0
    for ex in qa:
        r = agents.answer_question(idx, ex["question"], [])
        abst_ok.append(r["abstained"] == (not ex["answerable"]))
        for v in r["verification"]:
            cited += 1; sup += v["status"] == "SUPPORTED"
            unsup += v["status"] in ("UNSUPPORTED", "CONTRADICTED", "NO CITATION")
    print(f"Abstention accuracy : {np.mean(abst_ok):.2f}")
    print(f"Citation precision  : {sup / max(cited, 1):.2f}  (claims whose cited passage entails them)")
    print(f"Unsupported claims  : {unsup} / {cited}")
    print("Plain-LLM baseline has 0 citations by construction -> citation precision/recall undefined.")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
