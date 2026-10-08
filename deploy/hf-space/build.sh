#!/bin/sh
# Lay out a Space repo in $1: Dockerfile + api/ + built web UI. Usage: deploy/hf-space/build.sh <out-dir>
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$1"; [ -n "$OUT" ] || { echo "usage: $0 <out-dir>"; exit 1; }
(cd "$ROOT/web" && npm run build >/dev/null)
mkdir -p "$OUT"
rsync -a --delete --exclude .git --exclude __pycache__ --exclude .pytest_cache --exclude tests --exclude .env "$ROOT/api/" "$OUT/api/"
rsync -a --delete "$ROOT/web/dist/" "$OUT/web-dist/"
for f in Dockerfile serve.py seed.py supervisord.conf entrypoint.sh README.md; do cp "$ROOT/deploy/hf-space/$f" "$OUT/"; done
echo "Space repo laid out in $OUT"
