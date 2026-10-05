# EpisodeHub

Import open robot-learning datasets (Hugging Face LeRobot format), find bad episodes with automated quality checks, and search episodes in natural language.

**Status:** Week 2: foundation. The stack runs and health checks pass; the features below are not built yet.

## Quickstart

```bash
cp .env.example .env        # then change the passwords
docker compose up --build
curl http://localhost:8000/health        # {"status":"ok"}
curl http://localhost:8000/health/ready  # {"status":"ready"} once Postgres is up
```

- API docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001

Run tests without Docker:

```bash
cd api
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## Architecture

FastAPI + Postgres/pgvector + Redis/arq worker + MinIO + React. See [docs/design.md](docs/design.md).

## Roadmap / TODO for me

- [ ] **Week 2: Foundation**
  - [ ] Run `docker compose up --build` on my machine and confirm all five services are healthy
  - [x] `GET /health` returns `{"status":"ok"}`, `GET /health/ready` checks the DB
  - [x] Four tables defined in `api/app/models.py`; CI runs pytest
- [ ] **Week 3: Import**
  - [ ] Add Alembic; first migration creates the `vector` extension and all four tables
  - [ ] `POST /datasets/import {"hf_repo_id":"lerobot/pusht"}` stores pusht metadata (fps, robot_type, all episodes with length/duration) in Postgres and its video files in MinIO
  - [ ] `GET /datasets` lists it with `status: "ready"` and the correct `num_episodes`
  - [ ] Importing the same repo twice returns 409
  - [ ] Tests for both endpoints (DB tests can use a Postgres service container in CI)
- [ ] **Week 4: QC workers**
  - [ ] Six checks implemented as pure functions with one unit test each, using synthetic data (gap, dropped, frozen, joint-limit, velocity spike, length outlier)
  - [ ] `POST /datasets/{id}/qc` enqueues one `run_qc` job per episode; the worker writes `qc_results` rows
  - [ ] Every pusht episode ends with `qc_status` in pass/warn/fail; `GET /datasets/{id}` returns counts per status
- [ ] **Week 5: React UI**
  - [ ] Datasets page with an import form
  - [ ] Episodes table with QC badges, filter by status, pagination
  - [ ] Episode detail page plays the video (presigned MinIO URL) and lists failed checks
- [ ] **Week 6: NL search + export**
  - [ ] `POST /search {"query":"failed episodes shorter than 5 seconds"}` → LLM returns JSON that validates against a Pydantic `EpisodeFilter`; invalid JSON returns 422, never raw SQL
  - [ ] Filter → SQLAlchemy query is unit-tested without calling the LLM
  - [ ] Episode embeddings stored in `embeddings`; free-text part ranked by pgvector cosine distance
  - [ ] `GET /datasets/{id}/export?format=csv` downloads the filtered episode list
- [ ] **Week 7: Polish**
  - [ ] Test coverage on core logic, CI green with Postgres service
  - [ ] Deployed somewhere public with a live URL
  - [ ] README with screenshots, design decisions, and a 2-minute demo video

## Data

Public open datasets only (e.g. [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht)). No proprietary data.

## License

MIT
