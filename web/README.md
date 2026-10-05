# DataLens web

React 19 + TypeScript + Vite frontend for **DataLens** — data-quality checks for LLM fine-tuning datasets
(import instruction/chat datasets from Hugging Face, inspect automatic QC results, search samples in natural
language and export a clean train/val split).

```bash
npm install
npm run dev -- --host 127.0.0.1   # http://127.0.0.1:5173, proxies /api -> http://localhost:8000 (override with API_URL)
npm run build && npm run lint
```

## Routes
- `/` — **Datasets**: import form (example quick-picks, config/split/max samples, optional field mapping),
  dataset table with live import progress (polls every 2 s) and QC pass-rate bar, in-page delete confirm.
- `/datasets/:id` — **Dataset detail**: header with Hugging Face link, revision and detected field mapping;
  QC summary cards; per-check bars, category bars and token-length histogram (all click-to-filter);
  natural-language search (`POST /search`) with removable filter chips and parser badge; manual filters;
  paginated samples table; export panel (JSONL / chat, validation ratio, polling + download).
  All filters live in the URL query string.
- `/samples/:id` — **Sample viewer**: prompt / context / response with preserved whitespace, ``` code fences and
  inline PII highlights; QC results with links to duplicate / near-duplicate samples; metadata;
  prev/next with `[` / `]`; breadcrumb back to the dataset with its filters; per-sample QC re-run.

## Layout
- `src/api/` — typed client (`client.ts`) and types mirroring `api/app/schemas.py` (`types.ts`)
- `src/pages/` — one component per route
- `src/components/` — shared UI (pills, empty/error states, spinner, confirm button) plus
  `dataset/` and `sample/` page sections
- `src/lib/` — filter ⇄ URL helpers, formatting, code-fence/PII text segmentation
- `src/hooks/useResource.ts` — fetch + optional polling hook

Charts are hand-rolled SVG/CSS (no chart library). Light and dark themes follow `prefers-color-scheme`.
