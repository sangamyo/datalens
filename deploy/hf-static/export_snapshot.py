"""Export a running DataLens instance into static files for the read-only demo build.

Everything except the vectors comes from the real API endpoints, so the shapes match exactly what the
UI gets from FastAPI. The vectors are read with psql (the API never returns them).

    python3 export_snapshot.py --api http://localhost:17860/api --out out/data \
        --psql "colima ssh -- docker exec -i dl-snap psql -h 127.0.0.1 -U datalens datalens"

Layout of --out (read by web/src/api/static.ts):

    datasets.json                      GET /datasets
    datasets/{id}/detail.json          GET /datasets/{id}
    datasets/{id}/qc-summary.json      GET /datasets/{id}/qc/summary
    datasets/{id}/embeddings.json      GET /datasets/{id}/embeddings
    datasets/{id}/index.json           {chunk_size, columns, rows}: one compact row per sample (SampleOut
                                       fields + the checks that flagged it), ordered by sample_index
    datasets/{id}/samples/{n}.json     GET /samples/{id} (SampleDetail) for rows n*chunk_size ..
    datasets/{id}/vectors.bin          float32 little-endian, dim floats per sample, in index order
    datasets/{id}/vectors.json         {model, dim, count, ids}: sample id of each vector
"""

from __future__ import annotations

import argparse
import json
import shlex
import struct
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK_SIZE = 250
PAGE = 500
# compact index rows: SampleOut fields in this order, then the flagged check names
INDEX_COLUMNS = [
    "id", "sample_index", "category", "prompt_preview", "tokens_est", "lang", "qc_status", "qc_score", "flagged",
]


def get(api: str, path: str):
    with urllib.request.urlopen(f"{api}{path}", timeout=60) as r:
        return json.load(r)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def all_samples(api: str, dataset_id: int) -> list[dict]:
    items: list[dict] = []
    while True:
        page = get(api, f"/datasets/{dataset_id}/samples?limit={PAGE}&offset={len(items)}")
        items += page["items"]
        if not page["items"] or len(items) >= page["total"]:
            return items


def read_vectors(psql: str, dataset_id: int, model: str) -> dict[int, list[float]]:
    sql = (
        "COPY (SELECT e.sample_id, e.embedding::text FROM embeddings e JOIN samples s ON s.id = e.sample_id "
        f"WHERE s.dataset_id = {int(dataset_id)} AND e.model = '{model}' ORDER BY s.sample_index) TO STDOUT"
    )
    out = subprocess.run([*shlex.split(psql), "-X", "-q", "-c", sql], check=True, capture_output=True, text=True).stdout
    vectors: dict[int, list[float]] = {}
    for line in out.splitlines():
        sid, vec = line.split("\t")
        vectors[int(sid)] = [float(x) for x in vec.strip("[]").split(",")]
    return vectors


def export_dataset(api: str, psql: str, out: Path, ds: dict, workers: int) -> None:
    did = ds["id"]
    base = out / "datasets" / str(did)
    write_json(base / "detail.json", get(api, f"/datasets/{did}"))
    write_json(base / "qc-summary.json", get(api, f"/datasets/{did}/qc/summary"))
    status = get(api, f"/datasets/{did}/embeddings")
    write_json(base / "embeddings.json", status)

    samples = all_samples(api, did)
    with ThreadPoolExecutor(workers) as pool:
        details = list(pool.map(lambda s: get(api, f"/samples/{s['id']}"), samples))
    assert [s["id"] for s in samples] == [d["id"] for d in details]
    rows = []
    for d in details:
        flagged = sorted({r["check_name"] for r in d["qc_results"] if r["severity"] in ("warn", "fail")})
        rows.append([d[c] for c in INDEX_COLUMNS[:-1]] + [flagged])
    write_json(base / "index.json", {"chunk_size": CHUNK_SIZE, "columns": INDEX_COLUMNS, "rows": rows})
    for n in range(0, len(details), CHUNK_SIZE):
        write_json(base / "samples" / f"{n // CHUNK_SIZE}.json", details[n : n + CHUNK_SIZE])

    vectors = read_vectors(psql, did, status["model"])
    ids = [d["id"] for d in details if d["id"] in vectors]
    with open(base / "vectors.bin", "wb") as f:
        for sid in ids:
            v = vectors[sid]
            assert len(v) == status["dim"], (sid, len(v))
            f.write(struct.pack(f"<{len(v)}f", *v))
    write_json(base / "vectors.json", {"model": status["model"], "dim": status["dim"], "count": len(ids), "ids": ids})
    print(f"dataset {did}: {len(details)} samples, {len(ids)} vectors", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:17860/api", help="DataLens API base URL")
    ap.add_argument("--out", type=Path, required=True, help="output directory (e.g. out/data)")
    ap.add_argument(
        "--psql",
        default="colima ssh -- docker exec -i dl-snap psql -h 127.0.0.1 -U datalens datalens",
        help="command that runs psql against the DataLens database",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    datasets = [d for d in get(args.api, "/datasets") if d["status"] == "ready"]
    write_json(args.out / "datasets.json", datasets)
    for ds in datasets:
        export_dataset(args.api, args.psql, args.out, ds, args.workers)


if __name__ == "__main__":
    main()
