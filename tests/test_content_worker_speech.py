from types import SimpleNamespace

import pytest

from video_commerce.common.models import ContentFeatures
from video_commerce.services.content_worker import video_processor


class FakeFeatureStore:
    def __init__(self):
        self.statuses = []

    async def update_content_status(self, content_id, status):
        self.statuses.append((content_id, status))

    async def store_content_features(self, content_id, features):
        self.features = features


class FakeContentProcessor:
    async def process_video(self, _path, content_id):
        return ContentFeatures(
            content_id=content_id,
            visual_embedding=[0.1, 0.2],
            audio_features={
                "has_audio": True,
                "audio_transcript": "private transcript 手机",
                "transcription_status": "completed",
                "speech_categories": ["electronics"],
                "transcription_time_seconds": 0.3,
            },
        )


class FakeVectorSearch:
    async def add_content_embedding(self, _content_id, _embedding):
        return None


class FakeKafkaManager:
    def __init__(self):
        self.events = []

    async def send_feature_update(self, **kwargs):
        self.events.append(kwargs)


@pytest.mark.asyncio
async def test_content_worker_publishes_speech_metadata_without_transcript(monkeypatch, tmp_path):
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    config = SimpleNamespace(
        kafka_config=SimpleNamespace(),
        data_config=SimpleNamespace(cleanup_temp_files=False),
    )
    worker = video_processor.VideoProcessorWorker(config)
    worker.feature_store = FakeFeatureStore()
    worker.content_processor = FakeContentProcessor()
    worker.vector_search = FakeVectorSearch()
    worker.system_store = None
    worker.object_storage = None
    kafka_manager = FakeKafkaManager()
    monkeypatch.setattr(
        video_processor,
        "get_kafka_manager",
        lambda observability=None: kafka_manager,
    )

    await worker._handle_video_task(
        "video-processing-tasks",
        "content-1",
        {
            "content_id": "content-1",
            "file_path": str(upload),
            "filename": "upload.mp4",
            "request_id": "request-1",
        },
        None,
    )

    updates = kafka_manager.events[0]["feature_updates"]
    assert updates["has_transcript"] is True
    assert updates["transcription_status"] == "completed"
    assert updates["has_speech_categories"] is True
    assert "private transcript" not in str(kafka_manager.events)
