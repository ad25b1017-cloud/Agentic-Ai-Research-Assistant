import os
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"   # labels: contradiction, entailment, neutral
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "ollama"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # change if your key supports another
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
CHUNK_CHARS = 900
ABSTAIN_PROB = 0.15   # if best reranker probability is below this -> abstain (tune on your data)
