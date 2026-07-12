from types import SimpleNamespace
import os
import uuid

import pytest
from sqlalchemy import text

from video_commerce.common.config import DatabaseConfig
from video_commerce.common.feature_history_contracts import payload_sha256
from video_commerce.data_plane.system_store import (
    SystemStore,
    prepare_catalog_activation,
)
from video_commerce.services.catalog_event_publisher.main import CatalogEventPublisher


def test_prepare_catalog_activation_is_deterministic_across_retries():
    first = prepare_catalog_activation(
        source_version="catalog-v42",
        metadata_map={
            "product-2": {"price": 20.0},
            "product-1": {"price": 10.0},
        },
        event_time=100.0,
        available_at=105.0,
    )
    second = prepare_catalog_activation(
        source_version="catalog-v42",
        metadata_map={
            "product-1": {"price": 10.0},
            "product-2": {"price": 20.0},
        },
        event_time=100.0,
        available_at=105.0,
    )

    assert first == second
    assert first.activation_id
    assert first.expected_count == 2
    assert len(first.manifest_hash) == 64
    assert [row["product_id"] for row in first.outbox_rows] == [
        "product-1",
        "product-2",
    ]
    assert first.outbox_rows[0]["payload_hash"] == payload_sha256(
        first.outbox_rows[0]["payload"]
    )


def test_catalog_outbox_migration_declares_operational_tables():
    migration = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "migrations/postgres/004_catalog_feature_outbox.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS catalog_activations" in migration
    assert "CREATE TABLE IF NOT EXISTS catalog_feature_outbox" in migration
    assert "payload_hash" in migration
    assert "claim_expires_at" in migration
    assert "published_at" in migration


class FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.published = []
        self.failed = []

    async def claim_catalog_outbox(self, *, worker_id, batch_size, lease_seconds):
        assert worker_id == "publisher-1"
        assert batch_size == 10
        assert lease_seconds == 60
        return list(self.rows)

    async def mark_catalog_outbox_published(self, event_id, *, worker_id):
        self.published.append((event_id, worker_id))

    async def mark_catalog_outbox_failed(self, event_id, error, *, worker_id):
        self.failed.append((event_id, error, worker_id))


class FakeKafka:
    def __init__(self, succeeds=True):
        self.succeeds = succeeds
        self.events = []

    async def send_catalog_feature_event(self, event):
        self.events.append(event)
        return self.succeeds


@pytest.mark.asyncio
async def test_catalog_publisher_marks_rows_only_after_broker_ack():
    event = {"event_id": "event-1", "entity_id": "product-1"}
    store = FakeStore([event])
    kafka = FakeKafka(succeeds=True)
    publisher = CatalogEventPublisher(
        system_store=store,
        kafka_manager=kafka,
        config=SimpleNamespace(
            catalog_outbox_batch_size=10,
            catalog_outbox_lease_seconds=60,
        ),
        worker_id="publisher-1",
    )

    published = await publisher.publish_once()

    assert published == 1
    assert kafka.events == [event]
    assert store.published == [("event-1", "publisher-1")]
    assert store.failed == []


@pytest.mark.asyncio
async def test_catalog_publisher_releases_failed_rows_for_retry():
    event = {"event_id": "event-1", "entity_id": "product-1"}
    store = FakeStore([event])
    publisher = CatalogEventPublisher(
        system_store=store,
        kafka_manager=FakeKafka(succeeds=False),
        config=SimpleNamespace(
            catalog_outbox_batch_size=10,
            catalog_outbox_lease_seconds=60,
        ),
        worker_id="publisher-1",
    )

    published = await publisher.publish_once()

    assert published == 0
    assert store.published == []
    assert store.failed[0][0] == "event-1"
    assert "Kafka" in store.failed[0][1]


@pytest.mark.asyncio
async def test_catalog_activation_outbox_is_idempotent_and_lease_safe():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for Postgres outbox integration test")
    source_version = f"catalog-test-{uuid.uuid4().hex}"
    store = SystemStore(
        DatabaseConfig(
            enable=True,
            url=database_url,
            auto_create_schema=True,
            enable_retention_cleanup=False,
        )
    )
    await store.initialize()
    activation_id = None
    try:
        activation_id = await store.activate_product_catalog(
            source_version,
            {
                "test-product-1": {"price": 10.0},
                "test-product-2": {"price": 20.0},
            },
            event_time=100.0,
            available_at=105.0,
            batch_size=1,
        )
        assert (
            await store.activate_product_catalog(
                source_version,
                {
                    "test-product-2": {"price": 20.0},
                    "test-product-1": {"price": 10.0},
                },
                event_time=100.0,
                available_at=105.0,
                batch_size=2,
            )
            == activation_id
        )
        with pytest.raises(RuntimeError, match="does not match"):
            await store.activate_product_catalog(
                source_version,
                {
                    "test-product-2": {"price": 999.0},
                    "test-product-1": {"price": 10.0},
                },
                event_time=100.0,
                available_at=105.0,
            )

        first_claim = await store.claim_catalog_outbox(
            worker_id="worker-1", batch_size=10, lease_seconds=60
        )
        assert len(first_claim) == 2
        assert (
            await store.claim_catalog_outbox(
                worker_id="worker-2", batch_size=10, lease_seconds=60
            )
            == []
        )

        await store.mark_catalog_outbox_failed(
            first_claim[0]["event_id"], "retry", worker_id="worker-1"
        )
        retry_claim = await store.claim_catalog_outbox(
            worker_id="worker-2", batch_size=10, lease_seconds=60
        )
        assert [event["event_id"] for event in retry_claim] == [
            first_claim[0]["event_id"]
        ]
        await store.mark_catalog_outbox_published(
            retry_claim[0]["event_id"], worker_id="worker-2"
        )

        async with store.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT count(*) FROM catalog_feature_outbox "
                    "WHERE activation_id = :activation_id"
                ),
                {"activation_id": activation_id},
            )
            assert result.scalar_one() == 2
    finally:
        if activation_id:
            async with store.session_factory.begin() as session:
                await session.execute(
                    text(
                        "DELETE FROM catalog_feature_outbox "
                        "WHERE activation_id = :activation_id"
                    ),
                    {"activation_id": activation_id},
                )
                await session.execute(
                    text(
                        "DELETE FROM product_catalog_snapshot "
                        "WHERE activation_id = :activation_id"
                    ),
                    {"activation_id": activation_id},
                )
                await session.execute(
                    text(
                        "DELETE FROM catalog_activations "
                        "WHERE activation_id = :activation_id"
                    ),
                    {"activation_id": activation_id},
                )
        await store.close()
