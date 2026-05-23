"""
Object storage abstraction for local and S3-compatible upload persistence.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional, Tuple
from urllib.parse import urlparse

from video_commerce.common.config import ObjectStorageConfig


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
            config=BotoConfig(
                s3={"addressing_style": "path" if self.config.force_path_style else "virtual"},
                connect_timeout=self.config.connect_timeout_seconds,
                read_timeout=self.config.read_timeout_seconds,
                retries={"max_attempts": self.config.max_attempts, "mode": "standard"},
            ),
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
        if not Path(local_path).exists():
            raise FileNotFoundError(f"Staged file does not exist: {local_path}")
        if not self.is_remote:
            return local_path
        upload_kwargs = {}
        extra_args = self._build_upload_extra_args(content_type)
        if extra_args:
            upload_kwargs["ExtraArgs"] = extra_args
        await asyncio.to_thread(
            self._client.upload_file,
            local_path,
            self.config.bucket,
            object_name,
            **upload_kwargs,
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
        try:
            await asyncio.to_thread(self._client.download_file, bucket, key, temp_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
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

    def build_artifact_object_name(
        self,
        *,
        model_name: str,
        model_version: str,
        filename: str,
    ) -> str:
        prefix = self.config.prefix.strip("/")
        safe_model_name = self._sanitize_path_segment(model_name)
        safe_model_version = self._sanitize_path_segment(model_version or "latest")
        safe_filename = Path(filename).name
        object_name = f"artifacts/{safe_model_name}/{safe_model_version}/{safe_filename}"
        return f"{prefix}/{object_name}" if prefix else object_name

    async def sync_to_local_path(
        self,
        storage_path: str,
        local_path: str,
        *,
        expected_sha256: Optional[str] = None,
    ) -> str:
        target_path = Path(local_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        staged_path = await self.stage_to_local_temp(
            storage_path,
            local_path,
            expected_sha256=expected_sha256,
        )
        if Path(staged_path).resolve() == target_path.resolve():
            return str(target_path)

        try:
            os.replace(staged_path, target_path)
        finally:
            if os.path.exists(staged_path):
                os.remove(staged_path)
        return str(target_path)

    async def stage_to_local_temp(
        self,
        storage_path: str,
        local_path: str,
        *,
        expected_sha256: Optional[str] = None,
    ) -> str:
        """Download/copy an artifact into target directory without activating it."""
        target_path = Path(local_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(target_path.parent),
            prefix=f"{target_path.stem}.",
            suffix=target_path.suffix,
        )
        os.close(tmp_fd)
        try:
            if self.is_remote and self.is_remote_uri(storage_path):
                bucket, key = self._parse_s3_uri(storage_path)
                await asyncio.to_thread(self._client.download_file, bucket, key, tmp_path)
            else:
                source_path = Path(storage_path)
                if source_path.resolve() == target_path.resolve():
                    if expected_sha256:
                        await asyncio.to_thread(
                            self.verify_sha256,
                            str(target_path),
                            expected_sha256,
                        )
                    os.remove(tmp_path)
                    return str(target_path)
                await asyncio.to_thread(shutil.copy2, str(source_path), tmp_path)

            if expected_sha256:
                await asyncio.to_thread(self.verify_sha256, tmp_path, expected_sha256)
            return tmp_path
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @staticmethod
    def calculate_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def verify_sha256(cls, path: str, expected_sha256: str) -> None:
        actual_sha256 = cls.calculate_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Artifact checksum mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
            )

    async def sync_local_file_atomically(
        self,
        source_path: str,
        target_path: str,
        *,
        expected_sha256: Optional[str] = None,
    ) -> str:
        return await self.sync_to_local_path(
            source_path,
            target_path,
            expected_sha256=expected_sha256,
        )

    def _ensure_bucket_exists(self) -> None:
        assert self._client is not None
        bucket = self.config.bucket
        try:
            self._client.head_bucket(Bucket=bucket)
        except Exception as exc:
            error_code = self._get_s3_error_code(exc)
            if error_code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            create_kwargs = {"Bucket": bucket}
            if self.config.region and self.config.region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.config.region
                }
            self._client.create_bucket(**create_kwargs)

    def _build_upload_extra_args(self, content_type: Optional[str]) -> dict:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if self.config.checksum_algorithm:
            extra_args["ChecksumAlgorithm"] = self.config.checksum_algorithm
        if self.config.server_side_encryption:
            extra_args["ServerSideEncryption"] = self.config.server_side_encryption
        return extra_args

    @staticmethod
    def _get_s3_error_code(exc: Exception) -> str:
        response = getattr(exc, "response", {}) or {}
        error = response.get("Error", {}) if isinstance(response, dict) else {}
        return str(error.get("Code", ""))

    @staticmethod
    def _parse_s3_uri(storage_path: str) -> Tuple[str, str]:
        parsed = urlparse(storage_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI: {storage_path}")
        return bucket, key

    @staticmethod
    def _sanitize_path_segment(value: str) -> str:
        sanitized = value.replace("\\", "-").replace("/", "-").strip()
        return sanitized or "artifact"
