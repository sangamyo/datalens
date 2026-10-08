"""Background worker. Start with: arq app.worker.WorkerSettings"""

import logging

from arq.connections import RedisSettings

from app.config import get_settings
from app.jobs.embed_job import embed_dataset
from app.jobs.export_job import build_export
from app.jobs.import_job import import_dataset
from app.jobs.qc_job import run_dataset_qc, run_sample_qc


# Show app log lines (import/QC/embedding timings) alongside arq's own job logs.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class WorkerSettings:
    functions = [import_dataset, run_dataset_qc, run_sample_qc, embed_dataset, build_export]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 4
    job_timeout = 1800  # large imports / dataset-wide QC can take minutes
