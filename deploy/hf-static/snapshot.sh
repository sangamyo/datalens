#!/bin/sh
# Rebuild the demo snapshot in deploy/hf-static/data from the all-in-one image (deploy/hf-space): boot it, let it
# import 3,000 Dolly rows and finish QC + embeddings, export with export_snapshot.py, remove the container.
#   deploy/hf-space/build.sh /tmp/space && docker build -t datalens-space /tmp/space   # once
#   DOCKER="colima ssh -- docker" deploy/hf-static/snapshot.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DOCKER="${DOCKER:-docker}"
NAME="${NAME:-dl-snap}"
IMAGE="${IMAGE:-datalens-space}"
DATA="${DATA:-$HERE/data}"

$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
$DOCKER run -d --name "$NAME" --user 1000 "$IMAGE" >/dev/null
trap '$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

api() { $DOCKER exec "$NAME" curl -fsS "http://127.0.0.1:7860/api$1" 2>/dev/null; }
echo "waiting for import, QC and embeddings (a few minutes)..."
while :; do
  emb="$(api /datasets/1/embeddings || true)"
  qc="$(api /datasets/1/qc/summary || true)"
  if python3 -c 'import json,sys; e=json.loads(sys.argv[1]); q=json.loads(sys.argv[2]); sys.exit(not (e["total"] > 0 and e["embedded"] == e["total"] and q["counts"]["pending"] == 0))' "$emb" "$qc" 2>/dev/null; then
    break
  fi
  echo "  embeddings: ${emb:-starting}"
  sleep 15
done

# export inside the container (stdlib python + local psql), then copy the files out as a tar stream
$DOCKER exec -i "$NAME" sh -c 'rm -rf /tmp/snap && python - --api http://127.0.0.1:7860/api --out /tmp/snap --psql "psql -h 127.0.0.1 -U datalens datalens"' \
  < "$HERE/export_snapshot.py"
rm -rf "$DATA" && mkdir -p "$DATA"
$DOCKER exec "$NAME" tar -C /tmp/snap -cf - . | tar -xf - -C "$DATA"
du -sh "$DATA"
