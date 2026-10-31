"""Planner -> iterative retriever -> synthesizer -> verifier, plus compare / contradictions / gaps."""
import re
import numpy as np
from sentence_transformers import CrossEncoder
from config import NLI_MODEL, ABSTAIN_PROB
from llm import generate, generate_json

_nli = None
def nli():
    global _nli
    if _nli is None: _nli = CrossEncoder(NLI_MODEL)
    return _nli

SYS = ("You are a careful scientific research assistant. Use ONLY the numbered evidence given. "
       "Every factual sentence must end with citation tags like [E1] or [E2][E5]. "
       "If the evidence does not answer the question, say 'INSUFFICIENT EVIDENCE'. Never invent citations or numbers.")

def fmt(hits, offset=1):
    return "\n\n".join(f"[E{i}] ({h.chunk.paper}, {h.chunk.section}, p.{h.chunk.page}) {h.chunk.text}"
                       for i, h in enumerate(hits, offset))

# ---------- agentic retrieval ----------
def standalone(question, history):
    if not history: return question
    ctx = "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in history[-3:])
    return generate(f"Conversation:\n{ctx}\n\nRewrite the follow-up as a standalone research question.\n"
                    f"Follow-up: {question}\nStandalone question:").strip()

def plan(question):
    out = generate_json(f"Decompose this research question into 2-4 focused search queries.\n"
                        f"Question: {question}\nJSON list of strings.")
    return out if isinstance(out, list) and out else [question]

def retrieve(index, question, rounds=2, top_k=5):
    queries, seen, hits = plan(question), set(), []
    for r in range(rounds):
        for q in queries:
            for h in index.search(q, top_k=top_k):
                if h.chunk.id not in seen:
                    seen.add(h.chunk.id); hits.append(h)
        if r == rounds - 1: break
        check = generate_json(f"Question: {question}\nEvidence:\n{fmt(hits[:10])}\n\n"
                              'Is this enough to answer? JSON: {"enough": bool, "follow_up_query": string}')
        if not check or check.get("enough") or not check.get("follow_up_query"): break
        queries = [check["follow_up_query"]]
    return sorted(hits, key=lambda h: -h.score)[:10], queries

# ---------- verification ----------
def verify(answer, hits):
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", answer)
    rows = []
    for s in sents:
        ids = [int(x) for x in re.findall(r"\[E(\d+)\]", s)]
        claim = re.sub(r"\[E\d+\]", "", s).strip()
        if len(claim) < 25: continue
        valid = [i for i in ids if 1 <= i <= len(hits)]
        if not valid:
            rows.append({"claim": claim, "status": "NO CITATION", "entail": 0.0, "cites": []}); continue
        logits = nli().predict([(hits[i - 1].chunk.text[:1500], claim) for i in valid])
        e = np.exp(logits - logits.max(axis=1, keepdims=True)); p = e / e.sum(axis=1, keepdims=True)
        ent, con = p[:, 1].max(), p[:, 0].max()   # label order: contradiction, entailment, neutral
        status = "SUPPORTED" if ent > 0.5 else ("CONTRADICTED" if con > 0.5 else "UNSUPPORTED")
        rows.append({"claim": claim, "status": status, "entail": float(ent), "cites": valid})
    return rows

def answer_question(index, question, history):
    q = standalone(question, history)
    hits, queries = retrieve(index, q)
    top = hits[0].score if hits else 0.0
    if not hits or top < ABSTAIN_PROB:
        return {"question": q, "answer": "Insufficient evidence in the uploaded papers to answer this.",
                "hits": hits, "verification": [], "confidence": float(top), "abstained": True, "queries": queries}
    ans = generate(f"Evidence:\n{fmt(hits)}\n\nQuestion: {q}\nAnswer concisely with citations.", SYS)
    ver = verify(ans, hits)
    frac = np.mean([v["status"] == "SUPPORTED" for v in ver]) if ver else 0.0
    conf = float(0.5 * top + 0.5 * frac)
    return {"question": q, "answer": ans, "hits": hits, "verification": ver, "confidence": conf,
            "abstained": "INSUFFICIENT EVIDENCE" in ans.upper(), "queries": queries}

# ---------- report sections ----------
def _grouped_evidence(index, question, per_paper=4):
    parts, flat = [], []
    for p in index.papers:
        hs = index.search(f"{question} method dataset results", top_k=per_paper, paper=p)
        parts.append(f"=== {p} ===\n" + fmt(hs, offset=len(flat) + 1)); flat += hs
    return "\n\n".join(parts), flat

def compare_papers(index, question):
    ev, _ = _grouped_evidence(index, question)
    return generate_json(f"{ev}\n\nTopic: {question}\nBuild a comparison. JSON list, one object per paper with keys: "
                         '"paper","method","dataset","key_result","limitation". Use "not reported" if absent.') or []

def find_contradictions(index, question):
    ev, _ = _grouped_evidence(index, question)
    return generate_json(f"{ev}\n\nTopic: {question}\nFind claims where papers AGREE (consensus) or DISAGREE (conflict). "
                         'JSON list of objects: {"claim","type":"consensus|conflict","papers_for":[],"papers_against":[],'
                         '"evidence_ids":["E1"],"explanation"}. Only use what the evidence states.') or []

def research_gaps(index, question):
    hs = index.search(question + " limitations future work", top_k=8, section="limitations") \
         or index.search(question + " limitations future work", top_k=8)
    return generate(f"{fmt(hs)}\n\nTopic: {question}\nList 3-5 research gaps grounded in the stated limitations. "
                    "Cite evidence tags and say which are your inference vs. stated by the authors.", SYS)
