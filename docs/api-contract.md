# DataLens API contract

DataLens imports instruction / chat datasets used to fine-tune LLMs (from Hugging Face), runs automatic data-quality checks on every sample, lets you browse and search samples in natural language, and exports a clean train/val split as JSONL.

Single source of truth for backend and frontend. Shapes live in `api/app/schemas.py`; tables in `api/app/models.py`. The frontend calls everything under `/api/*` (the Vite dev server strips `/api` and proxies to `http://localhost:8000`).

## Conventions
- JSON everywhere except zip downloads. Errors: `{"detail": "..."}` with 400/404/409/422/503.
- Long work runs in the arq worker; the API returns `202` and the client polls.
- Sample text is stored in Postgres (it is small). Object storage holds the raw downloaded parquet (`datasets/{id}/raw/{n}.parquet`) and export zips (`exports/{id}.zip`).
- `tokens_est` = rough token estimate `ceil(total_chars / 4)` over prompt + context + response.

## Datasets  (owner: `routers/datasets.py`, `jobs/import_job.py`)
| Method | Path | Body / query | Response |
|---|---|---|---|
| POST | `/datasets/import` | `ImportRequest {hf_repo_id, config?, split="train", max_samples=2000, fields?}` | `202 DatasetOut` (status `pending`), enqueues `import_dataset(dataset_id, max_samples)`. `409` if the same repo+config+split is already imported. |
| GET | `/datasets` | — | `DatasetOut[]`, newest first |
| GET | `/datasets/{id}` | — | `DatasetDetail` (adds `qc_counts`, `categories [{name, count}]`, `token_histogram [{start, end, count}]` with ~12 buckets) |
| DELETE | `/datasets/{id}` | — | `204`, removes rows and objects under `datasets/{id}/` and its export zips |

Import job: status `pending → importing → ready | failed`. Reads the Hugging Face auto-converted parquet (`refs/convert/parquet` revision: `{config}/{split}/*.parquet`), falling back to parquet/jsonl files in the main revision. Field mapping is auto-detected unless `fields` is given:
- prompt: `instruction | prompt | question | query | input`
- context: `context | input` (only when a separate instruction column exists)
- response: `response | output | answer | completion | chosen`
- category: `category | task | type | label | source`
- chat format: a `messages` / `conversations` column (list of `{role, content}` or `{from, value}`) → first user turn = prompt, first assistant turn = response.
Stores the mapping used in `Dataset.fields`. Imports the first `max_samples` rows, then enqueues `run_dataset_qc(dataset_id)`.

## Samples  (owner: `routers/samples.py`)
| Method | Path | Query | Response |
|---|---|---|---|
| GET | `/datasets/{id}/samples` | `qc_status` (repeatable), `category`, `lang`, `min_tokens`, `max_tokens`, `text_contains`, `failed_check`, `limit=50`, `offset=0` | `SampleList {items: SampleOut[], total}` ordered by `sample_index` |
| GET | `/samples/{id}` | — | `SampleDetail` (full `prompt`, `context`, `response`, `qc_results`, `prev_id`, `next_id` within the dataset) |

## Quality checks  (owner: `routers/qc.py`, `jobs/qc_job.py`, `app/qc/`)
| Method | Path | Response |
|---|---|---|
| POST | `/datasets/{id}/qc` | `202 Enqueued {enqueued: 1}`; sets samples to `pending`, enqueues `run_dataset_qc(dataset_id)` |
| POST | `/samples/{id}/qc` | `202 Enqueued {enqueued: 1}`; enqueues `run_sample_qc(sample_id)` (uses dataset-wide stats) |
| GET | `/samples/{id}/qc` | `QCResultOut[]` |
| GET | `/datasets/{id}/qc/summary` | `QCSummary {counts, by_check: {check_name: {warn, fail}}}` |

Checks (`check_name`): `empty_or_short`, `length_outlier`, `exact_duplicate`, `near_duplicate`, `pii`, `non_english`, `refusal_boilerplate`, `formatting`. Each result has `passed`, `severity` (`pass|warn|fail`), a one-line `message`, and `details`. `details.spans` (for `pii`) is a list of `{field, start, end, kind}` so the UI can highlight; `details.duplicate_of` / `details.same_prompt_as` (for `exact_duplicate`) are lists of sample ids and `details.similar_to` (for `near_duplicate`) is a list of `{id, jaccard}`. `Sample.qc_status` = worst severity; `qc_score` = share of checks passed.

## Search  (owner: `routers/search.py`, `app/nlsearch.py`)
| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/search` | `SearchRequest {query, dataset_id?, limit=50}` | `SearchResponse {filter: SampleFilter, parser: "llm" \| "rules", items: SampleOut[], total}`; `404` if `dataset_id` is unknown |

The query becomes a validated `SampleFilter` (never raw SQL): LLM when `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` is set, otherwise (or on any LLM error) the rule-based parser.

## Exports  (owner: `routers/exports.py`, `jobs/export_job.py`)
| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/exports` | `ExportCreate {dataset_id, filter, val_ratio=0.1, format="jsonl"}` | `202 ExportOut`, enqueues `build_export(export_id)` |
| GET | `/exports?dataset_id=` | — | `ExportOut[]`, newest first |
| GET | `/exports/{id}` | — | `ExportOut` |
| GET | `/exports/{id}/download` | — | `application/zip`; `409` if not ready |

Formats: `jsonl` → one `{"prompt", "context", "response", "category"}` object per line; `chat` → one `{"messages": [{"role":"user",...},{"role":"assistant",...}]}` per line (context appended to the user turn). Zip: `manifest.json` (source repo/config/split/revision, filter, format, counts, created_at), `train.jsonl`, `val.jsonl`. Deterministic split seeded by export id.
