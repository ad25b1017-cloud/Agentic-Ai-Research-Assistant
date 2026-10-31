# Architecture
```mermaid
flowchart LR
  A[PDF upload] --> B[PyMuPDF parser + section chunker]
  B --> C[BM25 index]
  B --> D[Dense index: bge-small]
  C --> E[RRF hybrid retriever]
  D --> E
  E --> F[Cross-encoder reranker]
  Q[Question + chat memory] --> P[Planner: decompose / rewrite]
  P --> E
  F --> G[Iterative retrieval check]
  G --> H[Evidence synthesizer: cited answer]
  F -->|low score| X[Abstain]
  H --> V[Verifier: NLI claim vs cited passage]
  V --> R[Report: compare, contradictions, gaps, graph]
  R --> U[Streamlit UI]
```
## Data & preprocessing
Papers: your uploaded PDFs (list in README). References sections are dropped; captions glued to table bodies;
hyphenation repaired; chunks ~900 chars within one section, with paper/section/page metadata.
Claim verification evaluated on SciFact (allenai/scifact). No test data used for tuning thresholds (tune on dev).
