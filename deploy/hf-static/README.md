---
title: DataLens
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
short_description: Data-quality checks + semantic search for LLM fine-tuning data
---

# DataLens — static demo

DataLens is a data-quality platform for LLM fine-tuning datasets: import from the Hugging Face Hub, 8 automatic
QC checks per sample (exact and MinHash near-duplicates, PII, refusals, language, length outliers, formatting),
natural-language and semantic (embedding) search, and clean train/val JSONL export.

This Space is a **read-only snapshot**: 3,000 rows of `databricks/databricks-dolly-15k` with every QC result
and `BAAI/bge-small-en-v1.5` embedding precomputed by the real backend. It runs entirely in your browser —
filters and natural-language search use a TypeScript port of the backend's logic, and semantic search
embeds your query with the same model via transformers.js (downloaded once, ~34 MB). Importing, re-running QC
and exporting need the full stack: run it locally with Docker Compose.

Source code: https://github.com/sangamyo/datalens
