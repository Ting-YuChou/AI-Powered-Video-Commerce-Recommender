"""Direct uploads: POST /api/content/uploads and .../{content_id}/complete."""

import json
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from video_commerce.common.config import Config
from video_commerce.common.models import DirectUploadRequest
from video_commerce.data_plane.object_storage import ObjectHead, UploadObjectChanged
from video_commerce.data_plane.system_store import (
    ContentIdempotencyKeyInUse,
    ContentUploadReservationLimit,
)
from video_commerce.services.gateway import api as gateway_api

ADMIN_SCOPE = Config().security_config.admin_scope


class DirectStore:
    """In-memory content_jobs with the direct-upload store contract."""

    def __init__(self):
        self.jobs = {}
        self.reservations = []
        self.finalized = []
        self.published = []
        self.finalize_result = None
        self.finalize_error = None
        self.reserve_error = None
        self.lookup_error = None

    async def get_content_job_by_idempotency_key(self, *, user_id, idempotency_key):
        for job in self.jobs.values():
            if (
                job["user_id"] == user_id
                and job["idempotency_key"] == idempotency_key
            ):
                return json.loads(json.dumps(job))
        return None

    async def get_content_job(self, content_id):
        if self.lookup_error is not None:
            raise self.lookup_error
        job = self.jobs.get(content_id)
        return json.loads(json.dumps(job)) if job else None

    async def reserve_direct_upload(self, **kwargs):
        self.reservations.append(kwargs)
        if self.reserve_error is not None:
            error, commit = self.reserve_error
            if commit:
                self._reserve(**kwargs)
            raise error
        # Same rule as the Postgres query: a reservation counts until its URL
        # expiry plus the complete grace period.
        completable_after = time.time() - kwargs["complete_grace_seconds"]
        pending = [
            job
            for job in self.jobs.values()
            if job["user_id"] == kwargs["user_id"]
            and job["status"] == "awaiting_upload"
            and job["upload_expires_at"] > completable_after
        ]
        if len(pending) >= kwargs["max_pending_per_user"]:
            raise ContentUploadReservationLimit(kwargs["user_id"])
        return dict(self._reserve(**kwargs))

    def _reserve(self, **kwargs):
        job = {
            "content_id": kwargs["content_id"],
            "filename": kwargs["filename"],
            "storage_path": None,
            "user_id": kwargs["user_id"],
            "priority": kwargs["priority"],
            "status": "awaiting_upload",
            "payload": {
                "request_id": kwargs["request_id"],
                "incoming_uri": kwargs["incoming_uri"],
            },
            "task_event_id": None,
            "idempotency_key": kwargs["idempotency_key"],
            "request_fingerprint": kwargs["request_fingerprint"],
            "declared_size_bytes": kwargs["declared_size_bytes"],
            "declared_content_type": kwargs["declared_content_type"],
            "upload_expires_at": kwargs["upload_expires_at"],
        }
        self.jobs[job["content_id"]] = job
        return job

    async def finalize_direct_upload(self, task, *, storage_path):
        self.finalized.append((task, storage_path))
        if self.finalize_error is not None:
            error, commit = self.finalize_error
            if commit:
                self._finalize(task, storage_path)
            raise error
        if self.finalize_result is not None:
            return self.finalize_result
        job = self.jobs.get(task.content_id)
        if job is None:
            return False, None
        if job["status"] != "awaiting_upload":
            return False, dict(job)
        self._finalize(task, storage_path)
        return True, dict(job)

    def _finalize(self, task, storage_path):
        job = self.jobs[task.content_id]
        job.update(
            status="pending_publish",
            storage_path=storage_path,
            task_event_id=task.event_id,
        )

    async def claim_content_task_outbox_event(self, event_id, **_kwargs):
        return {"event_id": event_id}

    async def mark_content_task_outbox_published(self, event_id, *, worker_id):
        self.published.append(event_id)
        for job in self.jobs.values():
            if job["task_event_id"] == event_id:
                job["status"] = "queued"
        return True

    async def mark_content_task_outbox_failed(self, *_args, **_kwargs):
        return True


class DirectStorage:
    """Fake S3: objects keyed by URI, with ETags."""

    is_remote = True

    def __init__(self):
        self.objects = {}
        self.signed = []
        self.copies = []
        self.deleted = []
        self.copy_error = None
        self.presign_error = None
        self.before_copy = None

    def build_incoming_object_name(self, *, content_id, suffix):
        return f"uploads/incoming/{content_id}{suffix}"

    def build_object_name(self, *, content_id, suffix):
        return f"uploads/raw/{content_id}{suffix}"

    def object_uri(self, object_name):
        return f"s3://bucket/{object_name}"

    def presign_direct_upload(self, storage_uri, *, content_type, size_bytes, expires_in):
        if self.presign_error is not None:
            raise self.presign_error
        self.signed.append((storage_uri, content_type, size_bytes, expires_in))
        return {"url": "https://storage.example/bucket", "fields": {"key": storage_uri}}

    def put(self, uri, *, size, content_type="video/mp4", etag='"etag-1"'):
        self.objects[uri] = ObjectHead(size=size, etag=etag, content_type=content_type)

    async def head_upload_object(self, storage_uri):
        return self.objects.get(storage_uri)

    async def copy_upload_object(self, source_uri, *, object_name, if_match):
        self.copies.append((source_uri, object_name, if_match))
        if self.before_copy is not None:
            await self.before_copy()
        if self.copy_error is not None:
            raise self.copy_error
        source = self.objects.get(source_uri)
        if source is None or source.etag != if_match:
            raise UploadObjectChanged(source_uri)
        target = self.object_uri(object_name)
        self.objects[target] = source
        return target

    async def delete_uploaded_object(self, storage_path):
        self.deleted.append(storage_path)
        self.objects.pop(storage_path, None)


class Kafka:
    def __init__(self):
        self.acknowledged = True
        self.events = []

    async def publish_video_processing_task_payload(self, event):
        self.events.append(event)
        return self.acknowledged


class FeatureStore:
    def __init__(self):
        self.statuses = []

    async def update_content_status(self, content_id, status):
        self.statuses.append((content_id, status))


class Observability:
    def __init__(self):
        self.uploads = []

    def record_content_upload(self, priority, status):
        self.uploads.append((priority, status))


@pytest.fixture
def direct_env(monkeypatch, tmp_path):
    config = Config()
    config.data_config.upload_dir = str(tmp_path)
    config.data_config.direct_upload_enabled = True
    config.data_config.max_file_size = 1000
    config.kafka_config.enable = True
    runtime = SimpleNamespace(config=config, observability=Observability())
    env = SimpleNamespace(
        runtime=runtime,
        config=config,
        store=DirectStore(),
        storage=DirectStorage(),
        kafka=Kafka(),
        features=FeatureStore(),
    )
    monkeypatch.setattr(gateway_api.app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(gateway_api, "system_store", env.store)
    monkeypatch.setattr(gateway_api, "object_storage", env.storage)
    monkeypatch.setattr(gateway_api, "kafka_manager", env.kafka)
    monkeypatch.setattr(gateway_api, "feature_store", env.features)

    async def healthy(_runtime):
        return None

    monkeypatch.setattr(gateway_api, "_content_upload_dependency_error", healthy)
    return env


def _request(*, key=None, claims=None):
    headers = {gateway_api.IDEMPOTENCY_KEY_HEADER: key} if key is not None else {}
    state = SimpleNamespace(request_id="request-1")
    if claims is not None:
        state.auth_claims = claims
    return SimpleNamespace(headers=headers, state=state)


async def _initiate(
    *,
    key=None,
    claims=None,
    user_id="user-1",
    filename="clip.mp4",
    content_type="video/mp4",
    size_bytes=100,
    priority="normal",
):
    payload = DirectUploadRequest(
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        priority=priority,
        user_id=user_id,
    )
    return await gateway_api.initiate_direct_upload(
        _request(key=key, claims=claims), payload=payload
    )


async def _complete(content_id, *, user_id="user-1", claims=None):
    return await gateway_api.complete_direct_upload(
        _request(claims=claims), content_id=content_id, user_id=user_id
    )


def _body(response):
    return json.loads(response.body)


def _incoming(content_id):
    return f"s3://bucket/uploads/incoming/{content_id}.mp4"


async def _uploaded(env, **initiate_kwargs):
    ticket = await _initiate(**initiate_kwargs)
    content_id = ticket["content_id"]
    env.storage.put(_incoming(content_id), size=ticket["size_bytes"])
    return content_id


# --- reserve -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_reserve_is_404_while_direct_uploads_are_disabled(direct_env):
    direct_env.config.data_config.direct_upload_enabled = False

    response = await _initiate()

    assert response.status_code == 404
    assert _body(response)["error"]["code"] == "DIRECT_UPLOAD_DISABLED"
    assert direct_env.store.reservations == []


@pytest.mark.asyncio
async def test_reserve_creates_awaiting_job_and_signs_a_post(direct_env):
    before = time.time()
    ticket = await _initiate(content_type=" Video/MP4 ", size_bytes=100)

    content_id = ticket["content_id"]
    assert ticket["status"] == "awaiting_upload"
    assert ticket["upload"]["url"] == "https://storage.example/bucket"
    assert ticket["size_bytes"] == 100
    assert ticket["max_file_size"] == 1000
    assert ticket["effective_priority"] == "normal"
    assert ticket["complete_path"] == f"/api/content/uploads/{content_id}/complete"
    assert before + 900 <= ticket["expires_at"] <= time.time() + 900
    [reservation] = direct_env.store.reservations
    assert reservation["incoming_uri"] == _incoming(content_id)
    assert reservation["declared_content_type"] == "video/mp4"
    assert reservation["declared_size_bytes"] == 100
    assert reservation["max_pending_per_user"] == 5
    assert reservation["complete_grace_seconds"] == 3600
    # No key, no fingerprint: it only matters for replays.
    assert reservation["idempotency_key"] is None
    assert reservation["request_fingerprint"] is None
    assert direct_env.storage.signed == [(_incoming(content_id), "video/mp4", 100, 900)]
    assert ("normal", "direct_reserved") in direct_env.runtime.observability.uploads


@pytest.mark.asyncio
async def test_reserve_requires_a_user(direct_env):
    with pytest.raises(HTTPException) as exc:
        await _initiate(user_id=None)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_reserve_binds_the_owner_to_the_bearer_subject(direct_env):
    ticket = await _initiate(user_id=None, claims={"sub": "u1"})
    assert direct_env.store.jobs[ticket["content_id"]]["user_id"] == "u1"

    with pytest.raises(HTTPException) as exc:
        await _initiate(user_id="someone-else", claims={"sub": "u1"})
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,status",
    [
        ({"filename": "clip.exe"}, 400),
        ({"content_type": "application/octet-stream"}, 400),
        ({"size_bytes": 1001}, 413),
        ({"priority": "urgent"}, 400),
    ],
)
async def test_reserve_applies_the_multipart_upload_rules(direct_env, kwargs, status):
    with pytest.raises(HTTPException) as exc:
        await _initiate(**kwargs)

    assert exc.value.status_code == status
    assert direct_env.store.reservations == []


@pytest.mark.asyncio
async def test_reserve_lowers_high_priority_without_the_admin_scope(direct_env):
    lowered = await _initiate(priority="high", user_id=None, claims={"sub": "u1"})
    kept = await _initiate(
        priority="high", user_id=None, claims={"sub": "u2", "scope": ADMIN_SCOPE}
    )

    assert lowered["effective_priority"] == "normal"
    assert direct_env.store.jobs[lowered["content_id"]]["priority"] == "normal"
    assert kept["effective_priority"] == "high"


@pytest.mark.asyncio
async def test_reserve_caps_open_reservations_per_user(direct_env):
    direct_env.config.data_config.direct_upload_max_pending_per_user = 1
    await _initiate()

    limited = await _initiate()
    other_user = await _initiate(user_id="user-2")

    assert limited.status_code == 429
    assert _body(limited)["error"]["code"] == "TOO_MANY_PENDING_UPLOADS"
    assert _body(limited)["error"]["retryable"] is True
    assert other_user["status"] == "awaiting_upload"


@pytest.mark.asyncio
async def test_reserve_cap_counts_reservations_until_their_grace_ends(direct_env):
    direct_env.config.data_config.direct_upload_max_pending_per_user = 1
    grace = direct_env.config.data_config.direct_upload_complete_grace_seconds
    first = await _initiate()
    job = direct_env.store.jobs[first["content_id"]]

    # URL expired but /complete still accepts it: still counts.
    job["upload_expires_at"] = time.time() - grace + 60
    assert (await _initiate()).status_code == 429

    # Past the grace period it can no longer be completed.
    job["upload_expires_at"] = time.time() - grace - 60
    assert (await _initiate())["status"] == "awaiting_upload"


@pytest.mark.asyncio
async def test_reserve_signing_failure_leaves_no_reservation(direct_env):
    direct_env.storage.presign_error = RuntimeError("credentials unavailable")

    with pytest.raises(RuntimeError):
        await _initiate()

    assert direct_env.store.reservations == []


@pytest.mark.asyncio
async def test_reserve_retry_with_same_key_resigns_the_same_reservation(direct_env):
    first = await _initiate(key="k1")
    direct_env.store.jobs[first["content_id"]]["upload_expires_at"] = (
        time.time() + 300
    )

    second = await _initiate(key="k1")

    assert second["content_id"] == first["content_id"]
    assert second["idempotent_replay"] is True
    assert len(direct_env.store.reservations) == 1
    # Re-signed only for the time the reservation has left.
    assert 290 <= direct_env.storage.signed[-1][3] <= 300
    assert direct_env.store.jobs[first["content_id"]]["request_fingerprint"] == (
        gateway_api.direct_upload_request_fingerprint(
            filename="clip.mp4",
            content_type="video/mp4",
            size_bytes=100,
            priority="normal",
        )
    )


@pytest.mark.asyncio
async def test_reserve_retry_with_a_different_request_is_409(direct_env):
    first = await _initiate(key="k1")

    conflict = await _initiate(key="k1", size_bytes=101)

    assert conflict.status_code == 409
    assert _body(conflict)["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"
    assert _body(conflict)["content_id"] == first["content_id"]


@pytest.mark.asyncio
async def test_reserve_rejects_a_key_used_for_a_multipart_upload(direct_env):
    direct_env.store.jobs["multipart-1"] = {
        "content_id": "multipart-1",
        "user_id": "user-1",
        "status": "queued",
        "idempotency_key": "k1",
        "request_fingerprint": gateway_api.upload_request_fingerprint(
            body_sha256="0" * 64, size_bytes=100, filename="clip.mp4", priority="normal"
        ),
    }

    conflict = await _initiate(key="k1")

    assert conflict.status_code == 409
    assert _body(conflict)["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"


@pytest.mark.asyncio
async def test_reserve_retry_after_the_url_expired_is_409(direct_env):
    first = await _initiate(key="k1")
    direct_env.store.jobs[first["content_id"]]["upload_expires_at"] = time.time() + 30

    expired = await _initiate(key="k1")

    assert expired.status_code == 409
    assert _body(expired)["error"]["code"] == "DIRECT_UPLOAD_URL_EXPIRED"
    assert len(direct_env.storage.signed) == 1


@pytest.mark.asyncio
async def test_reserve_retry_after_complete_reports_the_job(direct_env):
    content_id = await _uploaded(direct_env, key="k1")
    await _complete(content_id)

    replay = await _initiate(key="k1")

    assert replay["content_id"] == content_id
    assert replay["status"] == "queued"
    assert replay["upload"] is None
    assert replay["idempotent_replay"] is True


@pytest.mark.asyncio
async def test_reserve_retry_after_expiry_is_409_job_failed(direct_env):
    first = await _initiate(key="k1")
    direct_env.store.jobs[first["content_id"]]["status"] = "upload_expired"

    replay = await _initiate(key="k1")

    assert replay.status_code == 409
    assert _body(replay)["error"]["code"] == "IDEMPOTENCY_KEY_JOB_FAILED"


@pytest.mark.asyncio
async def test_concurrent_reserve_with_same_key_returns_the_winner(direct_env):
    direct_env.store.reserve_error = (ContentIdempotencyKeyInUse("k1"), True)

    replay = await _initiate(key="k1")

    assert replay["idempotent_replay"] is True
    assert replay["content_id"] == direct_env.store.reservations[0]["content_id"]


def test_direct_upload_fingerprint_is_v2_and_covers_every_field():
    base = {
        "filename": "clip.mp4",
        "content_type": "video/mp4",
        "size_bytes": 100,
        "priority": "normal",
    }
    fingerprint = gateway_api.direct_upload_request_fingerprint(**base)

    assert fingerprint.startswith("v2:")
    for field, value in [
        ("filename", "other.mp4"),
        ("content_type", "video/webm"),
        ("size_bytes", 101),
        ("priority", "high"),
    ]:
        assert (
            gateway_api.direct_upload_request_fingerprint(**{**base, field: value})
            != fingerprint
        )


# --- complete ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_copies_to_raw_stages_publishes_and_cleans_up(direct_env):
    content_id = await _uploaded(direct_env)

    response = await _complete(content_id)

    assert response["status"] == "queued"
    assert response["size_bytes"] == 100
    [(source, raw_object, if_match)] = direct_env.storage.copies
    assert source == _incoming(content_id)
    assert raw_object.startswith(f"uploads/raw/{content_id}-")
    assert raw_object.endswith(".mp4")
    assert if_match == '"etag-1"'
    [(task, storage_path)] = direct_env.store.finalized
    assert storage_path == f"s3://bucket/{raw_object}"
    assert task.payload["file_path"] == storage_path
    assert task.payload["user_id"] == "user-1"
    assert direct_env.store.jobs[content_id]["status"] == "queued"
    assert direct_env.kafka.events == [{"event_id": task.event_id}]
    assert direct_env.features.statuses == [(content_id, "queued")]
    # The incoming object is gone; the raw copy stays.
    assert direct_env.storage.deleted == [_incoming(content_id)]
    assert storage_path in direct_env.storage.objects
    assert ("normal", "queued") in direct_env.runtime.observability.uploads


@pytest.mark.asyncio
async def test_complete_before_the_file_arrives_is_retryable_409(direct_env):
    ticket = await _initiate()

    response = await _complete(ticket["content_id"])

    assert response.status_code == 409
    assert _body(response)["error"]["code"] == "DIRECT_UPLOAD_OBJECT_MISSING"
    assert _body(response)["error"]["retryable"] is True
    assert direct_env.store.jobs[ticket["content_id"]]["status"] == "awaiting_upload"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,content_type",
    [(99, "video/mp4"), (100, "video/webm"), (100, "application/octet-stream")],
)
async def test_complete_rejects_an_object_that_does_not_match(
    direct_env, size, content_type
):
    ticket = await _initiate()
    direct_env.storage.put(
        _incoming(ticket["content_id"]), size=size, content_type=content_type
    )

    response = await _complete(ticket["content_id"])

    assert response.status_code == 422
    assert _body(response)["error"]["code"] == "DIRECT_UPLOAD_MISMATCH"
    assert direct_env.storage.copies == []
    assert direct_env.store.finalized == []


@pytest.mark.asyncio
async def test_complete_accepts_content_type_parameters(direct_env):
    ticket = await _initiate()
    direct_env.storage.put(
        _incoming(ticket["content_id"]), size=100, content_type="Video/MP4; codecs=avc1"
    )

    response = await _complete(ticket["content_id"])

    assert response["status"] == "queued"


@pytest.mark.asyncio
async def test_complete_rejects_an_object_replaced_after_the_check(direct_env):
    content_id = await _uploaded(direct_env)
    direct_env.storage.copy_error = UploadObjectChanged(_incoming(content_id))

    response = await _complete(content_id)

    assert response.status_code == 409
    assert _body(response)["error"]["code"] == "DIRECT_UPLOAD_CHANGED"
    assert _body(response)["error"]["retryable"] is True
    assert direct_env.store.finalized == []


@pytest.mark.asyncio
async def test_complete_whose_source_was_taken_by_a_concurrent_complete(direct_env):
    content_id = await _uploaded(direct_env)

    async def concurrent_complete():
        # The other request stages the job and deletes the incoming object.
        direct_env.store.jobs[content_id].update(
            status="queued", storage_path="s3://bucket/uploads/raw/winner.mp4"
        )
        direct_env.storage.objects.pop(_incoming(content_id))

    direct_env.storage.before_copy = concurrent_complete

    response = await _complete(content_id)

    assert response["status"] == "queued"
    assert response["already_completed"] is True
    assert direct_env.store.finalized == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "failed_missing_object"])
async def test_complete_of_a_failed_job_is_409(direct_env, status):
    content_id = await _uploaded(direct_env)
    direct_env.store.jobs[content_id]["status"] = status

    response = await _complete(content_id)

    assert response.status_code == 409
    assert _body(response)["error"]["code"] == "CONTENT_JOB_FAILED"
    assert _body(response)["status"] == status


@pytest.mark.asyncio
async def test_complete_hides_other_users_uploads(direct_env):
    content_id = await _uploaded(direct_env)

    for kwargs in [
        {"user_id": "user-2"},
        {"user_id": None, "claims": {"sub": "user-2"}},
    ]:
        with pytest.raises(HTTPException) as exc:
            await _complete(content_id, **kwargs)
        assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        await _complete(content_id, user_id=None)
    assert exc.value.status_code == 400
    assert direct_env.storage.copies == []


@pytest.mark.asyncio
async def test_complete_is_404_for_a_multipart_upload(direct_env):
    direct_env.store.jobs["multipart-1"] = {
        "content_id": "multipart-1",
        "user_id": "user-1",
        "status": "queued",
        "payload": {"request_id": "r"},
        "declared_size_bytes": None,
    }

    with pytest.raises(HTTPException) as exc:
        await _complete("multipart-1")

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_complete_after_the_grace_period_is_410(direct_env):
    content_id = await _uploaded(direct_env)
    grace = direct_env.config.data_config.direct_upload_complete_grace_seconds
    direct_env.store.jobs[content_id]["upload_expires_at"] = time.time() - grace - 1

    response = await _complete(content_id)

    assert response.status_code == 410
    assert _body(response)["error"]["code"] == "DIRECT_UPLOAD_EXPIRED"
    assert direct_env.storage.copies == []


@pytest.mark.asyncio
async def test_complete_within_the_grace_period_succeeds(direct_env):
    content_id = await _uploaded(direct_env)
    direct_env.store.jobs[content_id]["upload_expires_at"] = time.time() - 60

    response = await _complete(content_id)

    assert response["status"] == "queued"


@pytest.mark.asyncio
async def test_complete_of_an_expired_job_is_410(direct_env):
    content_id = await _uploaded(direct_env)
    direct_env.store.jobs[content_id]["status"] = "upload_expired"

    response = await _complete(content_id)

    assert response.status_code == 410


@pytest.mark.asyncio
async def test_repeated_complete_reports_the_current_status(direct_env):
    content_id = await _uploaded(direct_env)
    await _complete(content_id)

    again = await _complete(content_id)

    assert again["status"] == "queued"
    assert again["already_completed"] is True
    assert len(direct_env.storage.copies) == 1


@pytest.mark.asyncio
async def test_repeated_complete_republishes_a_pending_job(direct_env):
    content_id = await _uploaded(direct_env)
    direct_env.kafka.acknowledged = False
    pending = await _complete(content_id)
    assert pending.status_code == 503
    assert _body(pending)["error"]["code"] == "CONTENT_ENQUEUE_PENDING"
    assert direct_env.store.jobs[content_id]["status"] == "pending_publish"
    # The job is durable, so the incoming object is no longer needed.
    assert _incoming(content_id) in direct_env.storage.deleted

    direct_env.kafka.acknowledged = True
    again = await _complete(content_id)

    assert again["status"] == "queued"
    assert len(direct_env.storage.copies) == 1


@pytest.mark.asyncio
async def test_complete_that_loses_the_race_deletes_only_its_own_copy(direct_env):
    content_id = await _uploaded(direct_env)
    winner = dict(direct_env.store.jobs[content_id])
    winner.update(
        status="queued", storage_path="s3://bucket/uploads/raw/winner.mp4"
    )
    direct_env.store.finalize_result = (False, winner)

    response = await _complete(content_id)

    [(_, own_copy, _)] = direct_env.storage.copies
    assert direct_env.storage.deleted == [f"s3://bucket/{own_copy}"]
    assert response["status"] == "queued"
    assert response["already_completed"] is True


@pytest.mark.asyncio
async def test_complete_keeps_the_copy_when_staging_committed_despite_an_error(
    direct_env,
):
    content_id = await _uploaded(direct_env)
    direct_env.store.finalize_error = (RuntimeError("connection lost"), True)

    response = await _complete(content_id)

    assert response.status_code == 503
    assert _body(response)["error"]["code"] == "CONTENT_ENQUEUE_PENDING"
    raw_uri = direct_env.store.jobs[content_id]["storage_path"]
    assert raw_uri in direct_env.storage.objects
    assert raw_uri not in direct_env.storage.deleted


@pytest.mark.asyncio
async def test_complete_deletes_the_copy_when_staging_failed(direct_env):
    content_id = await _uploaded(direct_env)
    direct_env.store.finalize_error = (RuntimeError("database down"), False)

    with pytest.raises(RuntimeError):
        await _complete(content_id)

    [(_, own_copy, _)] = direct_env.storage.copies
    assert direct_env.storage.deleted == [f"s3://bucket/{own_copy}"]
    assert direct_env.store.jobs[content_id]["status"] == "awaiting_upload"


@pytest.mark.asyncio
async def test_complete_keeps_the_copy_when_the_staging_outcome_is_unknown(
    direct_env,
):
    content_id = await _uploaded(direct_env)

    async def lookup_fails_after_finalize(task, *, storage_path):
        direct_env.store.lookup_error = RuntimeError("database still down")
        raise RuntimeError("connection lost")

    direct_env.store.finalize_direct_upload = lookup_fails_after_finalize

    with pytest.raises(RuntimeError, match="connection lost"):
        await _complete(content_id)

    # The job may point at the copy, so only the orphan reconciler may delete it.
    assert direct_env.storage.deleted == []


@pytest.mark.asyncio
async def test_complete_still_works_after_the_flag_is_turned_off(direct_env):
    content_id = await _uploaded(direct_env)
    direct_env.config.data_config.direct_upload_enabled = False

    response = await _complete(content_id)

    assert response["status"] == "queued"


@pytest.mark.asyncio
async def test_direct_uploads_need_s3_storage(direct_env, monkeypatch):
    content_id = await _uploaded(direct_env)
    monkeypatch.setattr(direct_env.storage, "is_remote", False)

    for call in (_initiate(), _complete(content_id)):
        with pytest.raises(HTTPException) as exc:
            await call
        assert exc.value.status_code == 503
