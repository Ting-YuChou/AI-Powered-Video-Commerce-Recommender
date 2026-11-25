"""
Object storage abstraction for local and S3-compatible upload persistence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import AsyncIterator, Optional, Tuple
from urllib.parse import urlparse

from video_commerce.common.config import ObjectStorageConfig

logger = logging.getLogger(__name__)

# S3 uploads are written to ``<prefix>/raw/`` so a bucket lifecycle rule can
# target raw videos without touching ``artifacts/`` or ``content-features/``
# under the same prefix. deploy/object-storage/lifecycle.json depends on it.
RAW_UPLOAD_SUBPREFIX = "raw"
# Direct uploads are POSTed by the client to ``<prefix>/incoming/`` and copied
# to ``raw/`` on /complete. A lifecycle rule expires ``incoming/`` after a day.
INCOMING_UPLOAD_SUBPREFIX = "incoming"


@dataclass(frozen=True)
class StoredObject:
    uri: str
    size: int
    last_modified: float


@dataclass(frozen=True)
class ObjectHead:
    size: int
    etag: str
    content_type: str


class UploadObjectChanged(Exception):
    """The copy source changed or was removed after it was checked."""


class ObjectStorage:
    """Persist uploads either locally or to a remote S3-compatible bucket."""

    def __init__(self, config: ObjectStorageConfig):
        self.config = config
        self._client = None
        self._presign_client = None
        self._presign_client_lock = threading.Lock()

    @property
    def is_remote(self) -> bool:
        return self.config.backend == "s3"

    async def initialize(self) -> None:
        if not self.is_remote:
            return
        self._client = self._build_client(self.config.endpoint_url)
        if self.config.create_bucket_on_startup:
            await asyncio.to_thread(self._ensure_bucket_exists)

    def _get_presign_client(self):
        """Client that signs presigned POSTs for the browser-facing endpoint.

        Built on first use, so services that never presign (and the gateway
        while direct uploads are off) do not depend on
        OBJECT_STORAGE_PUBLIC_ENDPOINT_URL being valid.
        """
        with self._presign_client_lock:
            if self._presign_client is None:
                self._presign_client = self._build_client(
                    self.config.public_endpoint_url or self.config.endpoint_url,
                    signature_version="s3v4",
                )
            return self._presign_client

    def _build_client(self, endpoint_url: Optional[str], **config_kwargs):
        import boto3
        from botocore.client import Config as BotoConfig

        return boto3.client(
            "s3",
            # Helm and .env send unset endpoints as "", which boto3 rejects;
            # None selects the AWS endpoint for the region.
            endpoint_url=(endpoint_url or "").strip() or None,
            region_name=self.config.region,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            config=BotoConfig(
                s3={
                    "addressing_style": "path"
                    if self.config.force_path_style
                    else "virtual"
                },
                connect_timeout=self.config.connect_timeout_seconds,
                read_timeout=self.config.read_timeout_seconds,
                retries={"max_attempts": self.config.max_attempts, "mode": "standard"},
                **config_kwargs,
            ),
        )

    async def health_check(self, *, local_path: Optional[str] = None) -> dict:
        """Verify the storage operations required by uploads and reconciliation."""
        if not self.is_remote:
            if not local_path:
                return {"status": "healthy", "backend": "local"}
            path = Path(local_path)
            healthy = path.is_dir() and os.access(path, os.W_OK)
            return {
                "status": "healthy" if healthy else "unhealthy",
                "backend": "local",
                "error": None if healthy else f"Local storage is not writable: {path}",
            }
        if self._client is None:
            return {
                "status": "unhealthy",
                "backend": "s3",
                "error": "S3 client is not initialized",
            }
        try:
            await asyncio.to_thread(
                self._client.list_objects_v2,
                Bucket=self.config.bucket,
                Prefix=f"{self.config.prefix.rstrip('/')}/",
                MaxKeys=1,
            )
            return {"status": "healthy", "backend": "s3", "error": None}
        except Exception as exc:
            return {"status": "unhealthy", "backend": "s3", "error": str(exc)}

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

    async def persist_staged_file_create_only(
        self,
        local_path: str,
        *,
        object_name: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Create an immutable S3 object and fail if the key already exists."""
        if not Path(local_path).exists():
            raise FileNotFoundError(f"Staged file does not exist: {local_path}")
        if not self.is_remote:
            raise RuntimeError("create-only object persistence requires S3 storage")
        extra_args = self._build_upload_extra_args(content_type)

        def _put() -> None:
            event_name = "before-call.s3.PutObject"
            event_id = f"create-only-{id(self)}-{object_name}"

            def _add_create_only_header(*, params, **_kwargs) -> None:
                params.setdefault("headers", {})["If-None-Match"] = "*"

            # Older botocore service models do not expose IfNoneMatch as a
            # PutObject argument. Injecting the signed HTTP header preserves
            # the server-side atomic create-only condition on S3 and MinIO.
            self._client.meta.events.register_first(
                event_name, _add_create_only_header, unique_id=event_id
            )
            with open(local_path, "rb") as handle:
                try:
                    self._client.put_object(
                        Bucket=self.config.bucket,
                        Key=object_name,
                        Body=handle,
                        **extra_args,
                    )
                finally:
                    self._client.meta.events.unregister(event_name, unique_id=event_id)

        await asyncio.to_thread(_put)
        return f"s3://{self.config.bucket}/{object_name}"

    async def persist_immutable_bytes(
        self,
        payload: bytes,
        *,
        object_name: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Persist content-addressed bytes without allowing conflicting rewrites."""
        if (
            not object_name
            or Path(object_name).is_absolute()
            or ".." in Path(object_name).parts
        ):
            raise ValueError("immutable object name must be a safe relative path")
        if not self.is_remote:
            target = Path(self.config.download_dir) / object_name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                if target.read_bytes() != payload:
                    raise FileExistsError(
                        f"immutable object already exists with different bytes: {target}"
                    )
                return str(target)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                target.unlink(missing_ok=True)
                raise
            return str(target)

        descriptor, staged_path = tempfile.mkstemp(
            dir=self.config.download_dir
            if os.path.isdir(self.config.download_dir)
            else None,
            suffix=Path(object_name).suffix,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            try:
                return await self.persist_staged_file_create_only(
                    staged_path,
                    object_name=object_name,
                    content_type=content_type,
                )
            except Exception as exc:
                response = getattr(exc, "response", {}) or {}
                metadata = response.get("ResponseMetadata", {}) or {}
                status_code = metadata.get("HTTPStatusCode")
                error_code = str((response.get("Error", {}) or {}).get("Code") or "")
                if status_code == 412 or error_code in {
                    "PreconditionFailed",
                    "ConditionalRequestConflict",
                }:
                    return f"s3://{self.config.bucket}/{object_name}"
                raise
        finally:
            if os.path.exists(staged_path):
                os.remove(staged_path)

    def build_content_feature_object_name(self, *, content_id: str, sha256: str) -> str:
        prefix = self.config.prefix.strip("/")
        safe_content_id = self._sanitize_path_segment(content_id)
        safe_sha256 = self._sanitize_path_segment(sha256)
        path = f"content-features/{safe_content_id}/{safe_sha256}.json"
        return f"{prefix}/{path}" if prefix else path

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

    async def storage_path_exists(self, storage_path: str) -> bool:
        if self.is_remote and self.is_remote_uri(storage_path):
            bucket, key = self._parse_s3_uri(storage_path)
            try:
                await asyncio.to_thread(
                    self._client.head_object, Bucket=bucket, Key=key
                )
                return True
            except Exception as exc:
                if self._is_not_found(exc):
                    return False
                raise
        return bool(storage_path and os.path.exists(storage_path))

    def object_uri(self, object_name: str) -> str:
        return f"s3://{self.config.bucket}/{object_name}"

    def presign_direct_upload(
        self,
        storage_uri: str,
        *,
        content_type: str,
        size_bytes: int,
        expires_in: int,
    ) -> dict:
        """Presigned POST that writes exactly one file of this size and type.

        Returns ``{"url": ..., "fields": {...}}``; the client sends every field
        followed by the file as ``multipart/form-data``. The policy pins the
        key, the Content-Type, the exact length and SSE, so the client cannot
        write anywhere else or send a different size. Blocking: signing may
        fetch or refresh credentials, so call it from a worker thread.
        """
        if not self.is_remote:
            raise RuntimeError("direct uploads require S3 object storage")
        bucket, object_name = self._parse_s3_uri(storage_uri)
        fields = {"Content-Type": content_type}
        conditions: list = [
            ["content-length-range", int(size_bytes), int(size_bytes)],
            {"Content-Type": content_type},
        ]
        if self.config.server_side_encryption:
            fields["x-amz-server-side-encryption"] = self.config.server_side_encryption
            conditions.append(
                {"x-amz-server-side-encryption": self.config.server_side_encryption}
            )
        post = self._get_presign_client().generate_presigned_post(
            Bucket=bucket,
            Key=object_name,
            Fields=fields,
            Conditions=conditions,
            ExpiresIn=int(expires_in),
        )
        return {"url": post["url"], "fields": dict(post["fields"])}

    async def head_upload_object(self, storage_uri: str) -> Optional[ObjectHead]:
        """Size, ETag and Content-Type of an S3 object, or None when it is missing."""
        bucket, key = self._parse_s3_uri(storage_uri)
        try:
            response = await asyncio.to_thread(
                self._client.head_object, Bucket=bucket, Key=key
            )
        except Exception as exc:
            if self._is_not_found(exc):
                return None
            raise
        return ObjectHead(
            size=int(response.get("ContentLength") or 0),
            etag=str(response.get("ETag") or ""),
            content_type=str(response.get("ContentType") or ""),
        )

    async def copy_upload_object(
        self, source_uri: str, *, object_name: str, if_match: str
    ) -> str:
        """Server-side copy of ``source_uri`` to ``object_name`` in the bucket.

        Copies only while the source still has ETag ``if_match``; raises
        ``UploadObjectChanged`` if it was replaced or removed since then.
        """
        if not if_match:
            raise ValueError("if_match must be the ETag of the checked source")
        source_bucket, source_key = self._parse_s3_uri(source_uri)
        kwargs = {
            "Bucket": self.config.bucket,
            "Key": object_name,
            "CopySource": {"Bucket": source_bucket, "Key": source_key},
            "CopySourceIfMatch": if_match,
            "MetadataDirective": "COPY",
        }
        if self.config.server_side_encryption:
            kwargs["ServerSideEncryption"] = self.config.server_side_encryption
        if self.config.checksum_algorithm:
            kwargs["ChecksumAlgorithm"] = self.config.checksum_algorithm
        try:
            await asyncio.to_thread(self._client.copy_object, **kwargs)
        except Exception as exc:
            if self._is_not_found(exc) or self._is_precondition_failed(exc):
                raise UploadObjectChanged(source_uri) from exc
            raise
        return self.object_uri(object_name)

    async def list_storage_objects(self, prefix_uri: str) -> list[StoredObject]:
        if self.is_remote and self.is_remote_uri(prefix_uri):
            bucket, prefix = self._parse_s3_uri(prefix_uri.rstrip("/") + "/_")
            prefix = prefix.rsplit("/", 1)[0] + "/"

            def _list() -> list[StoredObject]:
                result = []
                paginator = self._client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for item in page.get("Contents", []):
                        key = str(item.get("Key") or "")
                        modified = item.get("LastModified")
                        if key:
                            result.append(
                                StoredObject(
                                    uri=f"s3://{bucket}/{key}",
                                    size=int(item.get("Size") or 0),
                                    last_modified=float(
                                        modified.timestamp() if modified else 0.0
                                    ),
                                )
                            )
                return sorted(result, key=lambda item: item.uri)

            return await asyncio.to_thread(_list)
        root = Path(prefix_uri)
        if not root.exists():
            return []
        result = []
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                result.append(
                    StoredObject(
                        uri=str(path),
                        size=int(stat.st_size),
                        last_modified=float(stat.st_mtime),
                    )
                )
        return sorted(result, key=lambda item: item.uri)

    async def iter_upload_object_pages(
        self, prefix_uri: str, *, page_size: int = 1000
    ) -> AsyncIterator[list[StoredObject]]:
        """Yield pages of upload objects stored directly under ``prefix_uri``.

        Uploads are flat within their prefix (``<prefix>/raw/<content_id><suffix>``,
        or ``<prefix>/<content_id><suffix>`` for older uploads); nested keys such
        as ``<prefix>/artifacts/`` and ``<prefix>/content-features/`` hold model
        and feature artifacts that are not referenced by ``content_jobs`` and
        must never be treated as orphan uploads. Only one page is held in memory.
        """
        page_size = max(1, int(page_size))
        if self.is_remote and self.is_remote_uri(prefix_uri):
            bucket, prefix = self._parse_s3_uri(prefix_uri.rstrip("/") + "/_")
            prefix = prefix.rsplit("/", 1)[0] + "/"
            token: Optional[str] = None
            while True:
                kwargs = {
                    "Bucket": bucket,
                    "Prefix": prefix,
                    "Delimiter": "/",
                    "MaxKeys": page_size,
                }
                if token:
                    kwargs["ContinuationToken"] = token
                response = await asyncio.to_thread(
                    self._client.list_objects_v2, **kwargs
                )
                page = []
                for item in response.get("Contents", []) or []:
                    key = str(item.get("Key") or "")
                    modified = item.get("LastModified")
                    if key and key != prefix:
                        page.append(
                            StoredObject(
                                uri=f"s3://{bucket}/{key}",
                                size=int(item.get("Size") or 0),
                                last_modified=float(
                                    modified.timestamp() if modified else 0.0
                                ),
                            )
                        )
                if page:
                    yield page
                token = response.get("NextContinuationToken")
                if not response.get("IsTruncated") or not token:
                    return
        root = Path(prefix_uri)
        if not root.is_dir():
            return
        iterator = os.scandir(root)
        try:
            while True:
                page = await asyncio.to_thread(
                    self._next_local_object_page, iterator, page_size
                )
                if not page:
                    return
                yield page
        finally:
            iterator.close()

    @staticmethod
    def _next_local_object_page(iterator, page_size: int) -> list[StoredObject]:
        page: list[StoredObject] = []
        for entry in iterator:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            page.append(
                StoredObject(
                    uri=str(Path(entry.path)),
                    size=int(stat.st_size),
                    last_modified=float(stat.st_mtime),
                )
            )
            if len(page) >= page_size:
                break
        return page

    async def list_storage_uris(
        self, prefix_uri: str, *, suffix: str = ""
    ) -> list[str]:
        """List immutable objects under a local or configured S3 prefix."""
        if self.is_remote and self.is_remote_uri(prefix_uri):
            bucket, prefix = self._parse_s3_uri(prefix_uri.rstrip("/") + "/_")
            prefix = prefix.rsplit("/", 1)[0] + "/"
            if bucket != self.config.bucket:
                raise ValueError("storage prefix bucket does not match configuration")

            def _list() -> list[str]:
                paginator = self._client.get_paginator("list_objects_v2")
                uris = []
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for item in page.get("Contents", []):
                        key = str(item.get("Key") or "")
                        if key and (not suffix or key.endswith(suffix)):
                            uris.append(f"s3://{bucket}/{key}")
                return sorted(uris)

            return await asyncio.to_thread(_list)
        root = Path(prefix_uri)
        if not root.exists():
            return []
        return sorted(
            str(path)
            for path in root.rglob(f"*{suffix}" if suffix else "*")
            if path.is_file()
        )

    @staticmethod
    def is_remote_uri(storage_path: str) -> bool:
        return storage_path.startswith("s3://")

    def build_object_name(self, *, content_id: str, suffix: str) -> str:
        """S3 key for a new upload: ``<prefix>/raw/<content_id><suffix>``."""
        return f"{self._raw_upload_key_prefix()}/{content_id}{suffix}"

    def upload_reconcile_prefixes(self, *, local_upload_dir: str) -> list[str]:
        """Prefixes whose direct children are upload objects, for orphan cleanup.

        S3: ``<prefix>/raw`` and, for uploads written before that layout,
        ``<prefix>`` itself. With an empty prefix both would be shared with
        anything else in the bucket (the root, and a root-level ``raw/``), so
        nothing is listed, as before the ``raw/`` layout. Local storage keeps
        the flat upload directory.
        """
        if not self.is_remote:
            return [local_upload_dir] if local_upload_dir else []
        prefix = self.config.prefix.strip("/")
        if not prefix:
            logger.warning(
                "object_storage_orphan_reconcile_disabled: OBJECT_STORAGE_PREFIX "
                "is empty, so upload objects share the bucket root with other "
                "objects; set a prefix to enable orphan cleanup"
            )
            return []
        bucket = self.config.bucket
        return [
            f"s3://{bucket}/{prefix}/{RAW_UPLOAD_SUBPREFIX}",
            f"s3://{bucket}/{prefix}",
        ]

    def build_incoming_object_name(self, *, content_id: str, suffix: str) -> str:
        """S3 key a direct upload is POSTed to: ``<prefix>/incoming/<content_id><suffix>``."""
        return f"{self._upload_key_prefix(INCOMING_UPLOAD_SUBPREFIX)}/{content_id}{suffix}"

    def _raw_upload_key_prefix(self) -> str:
        return self._upload_key_prefix(RAW_UPLOAD_SUBPREFIX)

    def _upload_key_prefix(self, subprefix: str) -> str:
        prefix = self.config.prefix.strip("/")
        return f"{prefix}/{subprefix}" if prefix else subprefix

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
        object_name = (
            f"artifacts/{safe_model_name}/{safe_model_version}/{safe_filename}"
        )
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
                await asyncio.to_thread(
                    self._client.download_file, bucket, key, tmp_path
                )
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
    def _get_s3_status(exc: Exception) -> Optional[int]:
        response = getattr(exc, "response", {}) or {}
        if not isinstance(response, dict):
            return None
        return (response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode")

    @classmethod
    def _is_not_found(cls, exc: Exception) -> bool:
        return cls._get_s3_status(exc) == 404 or cls._get_s3_error_code(exc) in {
            "404",
            "NoSuchKey",
            "NotFound",
        }

    @classmethod
    def _is_precondition_failed(cls, exc: Exception) -> bool:
        return cls._get_s3_status(exc) == 412 or cls._get_s3_error_code(exc) in {
            "412",
            "PreconditionFailed",
        }

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
