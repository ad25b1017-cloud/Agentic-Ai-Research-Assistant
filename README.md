# Agentic Research Assistant
PDF -> section-aware chunks -> BM25 + dense (RRF) -> cross-encoder rerank -> planner / iterative retrieval
-> cited answer -> NLI claim verification -> abstention -> report (comparison, contradictions, gaps, graph).

## Setup
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    export GEMINI_API_KEY=your_key      # Windows: set GEMINI_API_KEY=your_key
    # or fully local: export LLM_PROVIDER=ollama  (ollama pull llama3.1:8b)
    streamlit run app.py

First run downloads the embedding / reranker / NLI models (~1 GB total). Put demo PDFs in data/papers/.

## Evaluate (baselines)
    python -m eval.ablation data/papers eval/qa.jsonl            # BM25 vs dense vs hybrid vs +rerank (Recall@k, MRR)
    python -m eval.qa_eval data/papers eval/qa_questions.jsonl   # abstention acc, citation precision
    python -m eval.scifact_eval data/scifact/claims_dev.jsonl data/scifact/corpus.jsonl --n 300   # claim-verification F1

## Files
app.py UI · parser.py PDF parsing · index.py hybrid retrieval · agents.py planner/synthesizer/verifier/report
llm.py cached LLM · graph.py knowledge graph · langgraph_pipeline.py optional LangGraph · eval/ metrics · docs/ architecture + failures

## Config
config.py: model names, chunk size, ABSTAIN_PROB (tune it!). Docs: docs/ARCHITECTURE.md, docs/failure_cases.md
