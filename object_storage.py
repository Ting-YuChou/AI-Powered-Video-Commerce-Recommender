"""
Object storage abstraction for local and S3-compatible upload persistence.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple
from urllib.parse import urlparse

from config import ObjectStorageConfig


class ObjectStorage:
    """Persist uploads either locally or to a remote S3-compatible bucket."""

    def __init__(self, config: ObjectStorageConfig):
        self.config = config
        self._client = None

    @property
    def is_remote(self) -> bool:
        return self.config.backend == "s3"

    async def initialize(self) -> None:
        if not self.is_remote:
            return
        import boto3
        from botocore.client import Config as BotoConfig

        self._client = boto3.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            region_name=self.config.region,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            config=BotoConfig(s3={"addressing_style": "path" if self.config.force_path_style else "virtual"}),
        )
        if self.config.create_bucket_on_startup:
            await asyncio.to_thread(self._ensure_bucket_exists)

    async def persist_staged_file(
        self,
        local_path: str,
        *,
        object_name: str,
        content_type: Optional[str] = None,
    ) -> str:
        if not self.is_remote:
            return local_path
        await asyncio.to_thread(
            self._client.upload_file,
            local_path,
            self.config.bucket,
            object_name,
            ExtraArgs={"ContentType": content_type} if content_type else None,
        )
        return f"s3://{self.config.bucket}/{object_name}"

    async def materialize_for_processing(
        self,
        storage_path: str,
        *,
        suggested_suffix: str = "",
    ) -> Tuple[str, bool]:
        if not self.is_remote or not self.is_remote_uri(storage_path):
            return storage_path, False

        os.makedirs(self.config.download_dir, exist_ok=True)
        suffix = suggested_suffix or Path(urlparse(storage_path).path).suffix
        fd, temp_path = tempfile.mkstemp(dir=self.config.download_dir, suffix=suffix)
        os.close(fd)
        bucket, key = self._parse_s3_uri(storage_path)
        await asyncio.to_thread(self._client.download_file, bucket, key, temp_path)
        return temp_path, True

    async def delete_uploaded_object(self, storage_path: str) -> None:
        if self.is_remote and self.is_remote_uri(storage_path):
            bucket, key = self._parse_s3_uri(storage_path)
            await asyncio.to_thread(self._client.delete_object, Bucket=bucket, Key=key)
            return
        if storage_path and os.path.exists(storage_path):
            os.remove(storage_path)

    @staticmethod
    def is_remote_uri(storage_path: str) -> bool:
        return storage_path.startswith("s3://")

    def build_object_name(self, *, content_id: str, suffix: str) -> str:
        prefix = self.config.prefix.strip("/")
        filename = f"{content_id}{suffix}"
        return f"{prefix}/{filename}" if prefix else filename

    def _ensure_bucket_exists(self) -> None:
        assert self._client is not None
        bucket = self.config.bucket
        try:
            self._client.head_bucket(Bucket=bucket)
        except Exception:
            create_kwargs = {"Bucket": bucket}
            if self.config.region and self.config.region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.config.region
                }
            self._client.create_bucket(**create_kwargs)

    @staticmethod
    def _parse_s3_uri(storage_path: str) -> Tuple[str, str]:
        parsed = urlparse(storage_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI: {storage_path}")
        return bucket, key
