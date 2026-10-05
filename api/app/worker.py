"""Background worker. Start with: arq app.worker.WorkerSettings"""

from typing import Any

from arq.connections import RedisSettings

from app.config import get_settings


async def run_qc(ctx: dict[str, Any], episode_id: int) -> None:
    """TODO (Week 4): run all quality checks for one episode.

    1. Load the Episode (and its Dataset for fps) from Postgres.
    2. Load its frame data (parquet: timestamps, observation.state) from S3 storage / HF.
    3. Run each check as a pure function returning (check_name, passed, severity, details):
       timestamp_gap, dropped_frames, frozen_frames, joint_limit, velocity_spike, length_outlier.
    4. Delete old QCResult rows for this episode, insert the new ones.
    5. Set Episode.qc_status to the worst severity ("pass" / "warn" / "fail").

    Enqueue from the API with: `await redis.enqueue_job("run_qc", episode_id)`.
    """
    raise NotImplementedError("run_qc is a Week 4 TODO")


class WorkerSettings:
    functions = [run_qc]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
