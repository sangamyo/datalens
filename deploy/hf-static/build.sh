#!/bin/sh
# Build the static demo site into deploy/hf-static/out: the React UI with VITE_STATIC_DEMO=1 (hash routing,
# in-browser API over the snapshot), the snapshot in data/, and the Space README. Serve it with any static
# file server, e.g. `python3 -m http.server -d deploy/hf-static/out`.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/out"
[ -f "$HERE/data/datasets.json" ] || { echo "no snapshot in $HERE/data: run $HERE/snapshot.sh first"; exit 1; }
(cd "$ROOT/web" && VITE_STATIC_DEMO=1 npm run build -- --outDir "$OUT" --emptyOutDir >/dev/null)
rsync -a --delete "$HERE/data/" "$OUT/data/"
cp "$HERE/README.md" "$OUT/README.md"
cp "$HERE/space.gitattributes" "$OUT/.gitattributes"
echo "static site in $OUT ($(du -sh "$OUT" | cut -f1), $(find "$OUT" -type f | wc -l | tr -d ' ') files)"
