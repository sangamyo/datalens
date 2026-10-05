"""Thin wrapper around the S3-compatible object store (SeaweedFS in docker compose)."""

from collections.abc import Iterator
from functools import lru_cache

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import get_settings


@lru_cache
def s3() -> BaseClient:
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def bucket() -> str:
    return get_settings().s3_bucket


def ensure_bucket() -> None:
    try:
        s3().head_bucket(Bucket=bucket())
    except ClientError:
        s3().create_bucket(Bucket=bucket())


def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    s3().put_object(Bucket=bucket(), Key=key, Body=data, ContentType=content_type)


def put_file(key: str, path: str, content_type: str = "application/octet-stream") -> None:
    s3().upload_file(path, bucket(), key, ExtraArgs={"ContentType": content_type})


def get_bytes(key: str) -> bytes:
    return s3().get_object(Bucket=bucket(), Key=key)["Body"].read()


def head(key: str) -> dict:
    """Object metadata (ContentLength, ContentType). Raises ClientError if missing."""
    return s3().head_object(Bucket=bucket(), Key=key)


def stream(key: str, byte_range: str | None = None, chunk_size: int = 1 << 16) -> tuple[dict, Iterator[bytes]]:
    """Return (response metadata, chunk iterator). `byte_range` is an HTTP Range value like 'bytes=0-1023'."""
    kwargs = {"Bucket": bucket(), "Key": key}
    if byte_range:
        kwargs["Range"] = byte_range
    obj = s3().get_object(**kwargs)
    return obj, obj["Body"].iter_chunks(chunk_size)


def delete_prefix(prefix: str) -> int:
    """Delete every object under `prefix`. Returns the number deleted."""
    deleted = 0
    paginator = s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            s3().delete_objects(Bucket=bucket(), Delete={"Objects": keys})
            deleted += len(keys)
    return deleted

