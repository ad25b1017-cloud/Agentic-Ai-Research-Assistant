# Limitations and failure cases (fill as you test!)
| # | Input | What went wrong | Cause | Mitigation / future work |
|---|-------|-----------------|-------|--------------------------|
| 1 | two-column PDF | blocks interleaved | PyMuPDF block order | use GROBID |
| 2 | | | | |
Known limits: NLI model is small (deberta-v3-small); numeric claims are weakly verified; scanned PDFs need OCR;
LLM JSON output can fail to parse (tables then empty); abstain threshold tuned on few examples.
