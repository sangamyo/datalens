from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/datasets", tags=["datasets"])


class ImportRequest(BaseModel):
    hf_repo_id: str  # e.g. "lerobot/pusht"


@router.post("/import", status_code=201)
def import_dataset(req: ImportRequest) -> None:
    """TODO (Week 3): import a LeRobot dataset from Hugging Face.

    1. Validate `req.hf_repo_id` looks like "owner/name"; 409 if a Dataset with it already exists.
    2. Insert a `Dataset` row with status="importing".
    3. Download `meta/info.json` and `meta/episodes.jsonl` with `huggingface_hub`
       (hf_hub_download / snapshot_download with allow_patterns).
    4. Fill Dataset.fps, robot_type, num_episodes from info.json.
    5. For each episode: insert an `Episode` row (episode_index, length_frames,
       duration_s = length / fps, task) and upload its video file(s) to MinIO
       under "<hf_repo_id>/videos/...", storing the object key in `video_key`.
    6. Set status="ready" (or "failed" on error) and return the dataset as JSON.

    Later improvement: do steps 3-5 in an arq job and return 202 + dataset id immediately.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet (Week 3)")


@router.get("")
def list_datasets() -> None:
    """TODO (Week 3): return all datasets, newest first.

    Response: list of {id, hf_repo_id, name, fps, robot_type, num_episodes, status, created_at}.
    Define a Pydantic response model (DatasetOut) with `model_config = ConfigDict(from_attributes=True)`
    and use `db: Session = Depends(get_db)` to query `select(Dataset).order_by(Dataset.created_at.desc())`.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet (Week 3)")
