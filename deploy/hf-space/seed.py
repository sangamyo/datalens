"""Import a demo dataset on boot (the free Space has no persistent disk)."""

import time

import httpx

API = "http://127.0.0.1:7860/api"
DEMO = {"hf_repo_id": "databricks/databricks-dolly-15k", "split": "train", "max_samples": 3000}

for _ in range(300):
    try:
        if httpx.get(f"{API}/health/ready", timeout=5).status_code == 200:
            break
    except httpx.HTTPError:
        pass
    time.sleep(2)

if not httpx.get(f"{API}/datasets", timeout=10).json():
    r = httpx.post(f"{API}/datasets/import", json=DEMO, timeout=60)
    print(f"seed: import requested -> {r.status_code} {r.text[:200]}", flush=True)
else:
    print("seed: datasets already present, skipping", flush=True)
