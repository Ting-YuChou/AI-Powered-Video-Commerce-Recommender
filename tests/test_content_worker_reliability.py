from types import SimpleNamespace

import pytest

from video_commerce.common.config import ObjectStorageConfig
from video_commerce.common.models import ContentFeatures
from video_commerce.data_plane.kafka_client import NonRetryableProcessingError
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.content_processor import ContentModalityError, InvalidVideoError
from video_commerce.services.content_worker import video_processor


class RecordingObservability:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("record_") and not name.startswith("update_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args))

        return record

    def named(self, name):
        return [args for call, args in self.calls if call == name]


class FakeFeatureStore:
    def __init__(self):
        self.statuses = []
        self.features = None

    async def update_content_status(self, _content_id, status):
        self.statuses.append(status)

    async def store_content_features(self, _content_id, features):
        self.features = features


class Processor:
    def __init__(self, *, error=None, embedding=(0.6, 0.8), modality_status=None):
        self.error = error
        self.embedding = list(embedding)
        self.modality_status = modality_status or {"clip": "completed"}
        self.paths = []

    async def process_video(self, path, content_id):
        self.paths.append(path)
        if self.error is not None:
            raise self.error
        return ContentFeatures(
            content_id=content_id,
            visual_embedding=self.embedding,
            modality_status=dict(self.modality_status),
        )


class KafkaManager:
    def __init__(self):
        self.events = []

    async def send_feature_update(self, **kwargs):
        self.events.append(kwargs)


def _worker(monkeypatch, *, processor, system_store=None, object_storage=None):
    config = SimpleNamespace(
        kafka_config=SimpleNamespace(),
        data_config=SimpleNamespace(cleanup_temp_files=True),
        model_config=SimpleNamespace(content_pipeline_version="pipeline-v1"),
        service_topology_config=SimpleNamespace(content_processing_lease_seconds=60),
    )
    worker = video_processor.VideoProcessorWorker(config)
    worker.feature_store = FakeFeatureStore()
    worker.content_processor = processor
    worker.system_store = system_store
    worker.object_storage = object_storage
    worker.observability = RecordingObservability()
    kafka = KafkaManager()
    monkeypatch.setattr(video_processor, "get_kafka_manager", lambda **_kwargs: kafka)
    return worker, kafka


def _task(path, **overrides):
    task = {
        "event_id": "event-1",
        "content_id": "content-1",
        "pipeline_version": "pipeline-v1",
        "file_path": str(path),
        "filename": "clip.mp4",
    }
    task.update(overrides)
    return task


def _local_storage(tmp_path):
    return ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "dl"))
    )


def test_content_worker_does_not_load_product_vector_index():
    assert not hasattr(video_processor, "VectorSearchEngine")
    worker = video_processor.VideoProcessorWorker(SimpleNamespace(kafka_config=None))
    assert not hasattr(worker, "vector_search")


@pytest.mark.asyncio
async def test_local_source_is_kept_after_success(monkeypatch, tmp_path):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    worker, _ = _worker(
        monkeypatch, processor=Processor(), object_storage=_local_storage(tmp_path)
    )

    await worker._handle_video_task("video-processing-tasks", "c", _task(upload), None)

    assert upload.exists()
    assert worker.feature_store.statuses[-1] == "completed"


@pytest.mark.asyncio
async def test_local_source_is_kept_after_processing_failure(monkeypatch, tmp_path):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    worker, _ = _worker(
        monkeypatch,
        processor=Processor(error=RuntimeError("transient")),
        object_storage=_local_storage(tmp_path),
    )

    with pytest.raises(RuntimeError, match="transient"):
        await worker._handle_video_task(
            "video-processing-tasks", "c", _task(upload), None
        )

    assert upload.exists()


@pytest.mark.asyncio
async def test_duplicate_delivery_keeps_source_for_the_lease_holder(
    monkeypatch, tmp_path
):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")

    class LeaseHeldStore:
        async def claim_content_processing_run(self, **_kwargs):
            return {"outcome": "lease_held"}

    processor = Processor()
    worker, _ = _worker(
        monkeypatch,
        processor=processor,
        system_store=LeaseHeldStore(),
        object_storage=_local_storage(tmp_path),
    )

    # Returns normally, so the consumer commits the offset; the reaper owns
    # recovery if the holder dies.
    await worker._handle_video_task("video-processing-tasks", "c", _task(upload), None)

    assert upload.exists()
    assert processor.paths == []
    assert ("duplicate_in_progress",) in worker.observability.named(
        "record_content_processing"
    )


@pytest.mark.asyncio
async def test_materialized_remote_copy_is_removed_but_source_is_not(
    monkeypatch, tmp_path
):
    temp_copy = tmp_path / "materialized.mp4"
    deleted = []

    class RemoteStorage:
        async def materialize_for_processing(self, storage_path, *, suggested_suffix):
            assert storage_path == "s3://bucket/uploads/content-1.mp4"
            temp_copy.write_bytes(b"video")
            return str(temp_copy), True

        async def delete_uploaded_object(self, path):
            deleted.append(path)

    worker, _ = _worker(
        monkeypatch,
        processor=Processor(error=RuntimeError("transient")),
        object_storage=RemoteStorage(),
    )

    with pytest.raises(RuntimeError):
        await worker._handle_video_task(
            "video-processing-tasks",
            "c",
            _task("s3://bucket/uploads/content-1.mp4"),
            None,
        )

    assert not temp_copy.exists()
    assert deleted == []


@pytest.mark.asyncio
async def test_pipeline_version_mismatch_is_non_retryable(monkeypatch, tmp_path):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    worker, _ = _worker(monkeypatch, processor=Processor())

    with pytest.raises(NonRetryableProcessingError) as exc:
        await worker._handle_video_task(
            "video-processing-tasks",
            "c",
            _task(upload, pipeline_version="pipeline-v0"),
            None,
        )

    assert isinstance(exc.value, ValueError)
    assert upload.exists()


@pytest.mark.asyncio
async def test_missing_local_source_is_retryable(monkeypatch, tmp_path):
    # A shared-volume file that is not visible yet gets the bounded consumer
    # retries before the DLQ, rather than going straight to the DLQ.
    worker, _ = _worker(monkeypatch, processor=Processor())

    with pytest.raises(FileNotFoundError) as exc:
        await worker._handle_video_task(
            "video-processing-tasks", "c", _task(tmp_path / "missing.mp4"), None
        )

    assert not isinstance(exc.value, NonRetryableProcessingError)


@pytest.mark.asyncio
async def test_unrelated_file_not_found_stays_retryable(monkeypatch, tmp_path):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    worker, _ = _worker(
        monkeypatch,
        processor=Processor(error=FileNotFoundError("/models/clip/weights.bin")),
    )

    with pytest.raises(FileNotFoundError) as exc:
        await worker._handle_video_task(
            "video-processing-tasks", "c", _task(upload), None
        )

    assert not isinstance(exc.value, NonRetryableProcessingError)


@pytest.mark.asyncio
async def test_missing_remote_object_is_non_retryable(monkeypatch):
    class NoSuchKey(Exception):
        response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }

    class RemoteStorage:
        async def materialize_for_processing(self, *_args, **_kwargs):
            raise NoSuchKey("gone")

    worker, _ = _worker(
        monkeypatch, processor=Processor(), object_storage=RemoteStorage()
    )

    with pytest.raises(NonRetryableProcessingError):
        await worker._handle_video_task(
            "video-processing-tasks", "c", _task("s3://bucket/uploads/x.mp4"), None
        )


@pytest.mark.asyncio
async def test_invalid_video_is_non_retryable(monkeypatch, tmp_path):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"not-a-video")
    worker, _ = _worker(
        monkeypatch, processor=Processor(error=InvalidVideoError("no keyframes"))
    )

    with pytest.raises(NonRetryableProcessingError) as exc:
        await worker._handle_video_task(
            "video-processing-tasks", "c", _task(upload), None
        )

    assert isinstance(exc.value.__cause__, InvalidVideoError)
    assert worker.feature_store.statuses[-1] == "failed"


@pytest.mark.asyncio
async def test_clip_failure_stays_retryable_and_is_counted(monkeypatch, tmp_path):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    worker, kafka = _worker(
        monkeypatch,
        processor=Processor(error=ContentModalityError("clip", "RuntimeError")),
    )

    with pytest.raises(ContentModalityError):
        await worker._handle_video_task(
            "video-processing-tasks", "c", _task(upload), None
        )

    assert ("clip", "failed") in worker.observability.named(
        "record_content_modality_status"
    )
    assert worker.feature_store.statuses[-1] == "failed"
    assert kafka.events == []


@pytest.mark.asyncio
async def test_zero_norm_embedding_is_not_reported_as_visual(monkeypatch, tmp_path):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    worker, kafka = _worker(
        monkeypatch,
        processor=Processor(
            embedding=(0.0, 0.0),
            modality_status={"clip": "completed", "ocr": "failed"},
        ),
    )

    await worker._handle_video_task("video-processing-tasks", "c", _task(upload), None)

    updates = kafka.events[0]["feature_updates"]
    assert updates["has_visual_embedding"] is False
    assert worker.observability.named("record_content_zero_norm_embedding") == [()]
    statuses = worker.observability.named("record_content_modality_status")
    assert ("clip", "zero_norm") in statuses
    assert ("ocr", "failed") in statuses


@pytest.mark.asyncio
async def test_real_embedding_reports_visual_and_modalities(monkeypatch, tmp_path):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    worker, kafka = _worker(monkeypatch, processor=Processor())

    await worker._handle_video_task("video-processing-tasks", "c", _task(upload), None)

    assert kafka.events[0]["feature_updates"]["has_visual_embedding"] is True
    assert worker.observability.named("record_content_zero_norm_embedding") == []
    assert worker.observability.named("record_content_modality_status") == [
        ("clip", "completed")
    ]


class _RenewStore:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def renew_content_processing_lease(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else True
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def _run_renewal(monkeypatch, worker, *, lease_seconds, max_sleeps):
    import asyncio

    clock = {"now": 1000.0}
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds
        if len(sleeps) > max_sleeps:
            raise asyncio.CancelledError()

    monkeypatch.setattr(video_processor.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(video_processor.time, "monotonic", lambda: clock["now"])
    lease_lost = asyncio.Event()
    try:
        await worker._renew_processing_lease(
            content_id="content-1",
            pipeline_version="pipeline-v1",
            lease_seconds=lease_seconds,
            lease_lost=lease_lost,
        )
    except asyncio.CancelledError:
        pass
    return lease_lost, sleeps


@pytest.mark.asyncio
async def test_lease_renewal_survives_transient_errors(monkeypatch):
    worker, _ = _worker(monkeypatch, processor=Processor())
    worker.system_store = _RenewStore(
        [RuntimeError("db blip"), RuntimeError("db blip"), True, True]
    )

    lease_lost, sleeps = await _run_renewal(
        monkeypatch, worker, lease_seconds=300, max_sleeps=4
    )

    assert not lease_lost.is_set()
    assert worker.system_store.calls == 4
    # Errors retry quickly (2s, 4s) instead of waiting a full interval.
    assert sleeps[:3] == [100.0, 2.0, 4.0]
    assert ("lease_renew_error",) in worker.observability.named(
        "record_content_processing"
    )


@pytest.mark.asyncio
async def test_lease_renewal_reports_lost_before_the_lease_expires(monkeypatch):
    worker, _ = _worker(monkeypatch, processor=Processor())
    worker.system_store = _RenewStore([RuntimeError("db down")] * 100)

    lease_lost, sleeps = await _run_renewal(
        monkeypatch, worker, lease_seconds=60, max_sleeps=100
    )

    assert lease_lost.is_set()
    # Gave up while the last renewed lease (60s) was still valid.
    assert sum(sleeps) < 60
    assert ("lease_lost",) in worker.observability.named(
        "record_content_processing"
    )


@pytest.mark.asyncio
async def test_lease_renewal_marks_lost_when_lease_taken(monkeypatch):
    worker, _ = _worker(monkeypatch, processor=Processor())
    worker.system_store = _RenewStore([False])

    lease_lost, _ = await _run_renewal(
        monkeypatch, worker, lease_seconds=60, max_sleeps=5
    )

    assert lease_lost.is_set()


@pytest.mark.asyncio
async def test_failed_renewal_task_does_not_mask_processing_outcome(
    monkeypatch, tmp_path
):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")

    class AcquiringStore:
        async def claim_content_processing_run(self, **_kwargs):
            return {"outcome": "acquired"}

        async def assert_content_processing_lease(self, **_kwargs):
            return True

        async def complete_content_processing_run(self, **_kwargs):
            return True

    worker, kafka = _worker(
        monkeypatch, processor=Processor(), system_store=AcquiringStore()
    )

    async def crashing_renewal(**_kwargs):
        raise RuntimeError("renewal crashed")

    monkeypatch.setattr(worker, "_renew_processing_lease", crashing_renewal)

    await worker._handle_video_task("video-processing-tasks", "c", _task(upload), None)

    assert worker.feature_store.statuses[-1] == "completed"
    assert kafka.events
