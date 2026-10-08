---
title: DataLens
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: QC + semantic search for LLM fine-tuning datasets
---

# DataLens — live demo

Data-quality platform for LLM fine-tuning datasets: import from the Hugging Face Hub, 8 automatic
QC checks per sample (MinHash near-duplicates, PII, refusals, length outliers, formatting),
natural-language and semantic (pgvector) search, and clean JSONL export.

This Space boots with a 3,000-row slice of `databricks/databricks-dolly-15k`; QC and embeddings run
in the background for the first few minutes. Storage is ephemeral — anything you import is gone
after a restart.

Source code: https://github.com/sangamyo/datalens
