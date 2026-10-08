#!/bin/sh
# Data is ephemeral on a free Space: every boot starts from an empty database and re-seeds.
set -e
DATA=/home/user/data
mkdir -p "$DATA/seaweed" "$DATA/redis" "$DATA/run"
if [ ! -s "$DATA/pg/PG_VERSION" ]; then
  initdb -D "$DATA/pg" -U datalens --auth=trust >/dev/null
fi
export DATABASE_URL="postgresql+psycopg://datalens@127.0.0.1:5432/datalens"
export REDIS_URL="redis://127.0.0.1:6379/0"
export S3_ENDPOINT_URL="http://127.0.0.1:8333" S3_ACCESS_KEY=any S3_SECRET_KEY=any S3_BUCKET=datalens
exec supervisord -c /app/supervisord.conf
