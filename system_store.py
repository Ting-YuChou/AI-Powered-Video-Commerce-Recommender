"""
Durable Postgres-backed system store for operational state and training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, desc, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import DatabaseConfig


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base metadata for system store tables."""


class InteractionEvent(Base):
    __tablename__ = "interaction_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    product_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    context: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ContentJob(Base):
    __tablename__ = "content_jobs"

    content_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    priority: Mapped[str] = mapped_column(String(32), nullable=False, default="normal")
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ProductCatalogSnapshot(Base):
    __tablename__ = "product_catalog_snapshot"

    product_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


Index("ix_interaction_events_occurred_at_desc", InteractionEvent.occurred_at.desc())
Index(
    "ix_model_checkpoints_latest",
    ModelCheckpoint.model_name,
    ModelCheckpoint.created_at.desc(),
    ModelCheckpoint.id.desc(),
)


@dataclass
class DatabaseHealth:
    status: str
    response_time_ms: float
    error: Optional[str] = None


class SystemStore:
    """Operational persistence layer backed by Postgres."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker | None = None
        self.is_connected = False

    @property
    def enabled(self) -> bool:
        return self.config.enable

    async def initialize(self) -> None:
        if not self.enabled:
            return

        self.engine = create_async_engine(
            self.config.url,
            pool_pre_ping=True,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            echo=self.config.echo_sql,
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

        if self.config.auto_create_schema:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                await self._ensure_operational_indexes(conn)

        self.is_connected = True

    async def _ensure_operational_indexes(self, conn) -> None:
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_interaction_events_occurred_at_desc "
                "ON interaction_events (occurred_at DESC)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_model_checkpoints_latest "
                "ON model_checkpoints (model_name, created_at DESC, id DESC)"
            )
        )

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()
        self.engine = None
        self.session_factory = None
        self.is_connected = False

    async def health_check(self) -> DatabaseHealth:
        if not self.enabled:
            return DatabaseHealth(status="healthy", response_time_ms=0.0)

        started_at = time.time()
        try:
            async with self.session_factory() as session:
                await session.execute(text("SELECT 1"))
            return DatabaseHealth(
                status="healthy",
                response_time_ms=round((time.time() - started_at) * 1000, 2),
            )
        except Exception as exc:
            return DatabaseHealth(
                status="unhealthy",
                response_time_ms=round((time.time() - started_at) * 1000, 2),
                error=str(exc),
            )

    async def record_interaction_events_batch(
        self,
        interactions: Iterable[Dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return

        rows = []
        for interaction in interactions:
            occurred_at = interaction.get("occurred_at")
            if isinstance(occurred_at, (int, float)):
                occurred_at = datetime.fromtimestamp(occurred_at, tz=timezone.utc)
            elif not isinstance(occurred_at, datetime):
                occurred_at = _utc_now()

            rows.append(
                {
                    "event_id": interaction["event_id"],
                    "schema_version": int(interaction.get("schema_version", 1)),
                    "request_id": interaction.get("request_id"),
                    "user_id": interaction["user_id"],
                    "product_id": interaction["product_id"],
                    "action": interaction["action"],
                    "context": interaction.get("context", {}),
                    "occurred_at": occurred_at,
                }
            )

        if not rows:
            return

        stmt = pg_insert(InteractionEvent).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=[InteractionEvent.event_id])
        async with self.session_factory.begin() as session:
            await session.execute(stmt)

    async def get_training_interactions(self, limit: int = 50000) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        async with self.session_factory() as session:
            result = await session.execute(
                select(InteractionEvent)
                .order_by(InteractionEvent.occurred_at.desc())
                .limit(limit)
            )
            rows = result.scalars().all()

        interactions: List[Dict[str, Any]] = []
        for row in rows:
            interactions.append(
                {
                    "event_id": row.event_id,
                    "schema_version": row.schema_version,
                    "request_id": row.request_id,
                    "user_id": row.user_id,
                    "product_id": row.product_id,
                    "action": row.action,
                    "context": row.context or {},
                    "timestamp": row.occurred_at.timestamp(),
                    "occurred_at": row.occurred_at.timestamp(),
                }
            )
        return interactions

    async def get_analytics_summary(self) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "total_interactions": 0,
                "unique_users": 0,
                "unique_products": 0,
                "action_counts": {},
                "ctr": 0.0,
                "conversion_rate": 0.0,
                "timestamp": time.time(),
                "source": "postgres",
            }

        async with self.session_factory() as session:
            totals_result = await session.execute(
                select(
                    func.count(InteractionEvent.event_id),
                    func.count(func.distinct(InteractionEvent.user_id)),
                    func.count(func.distinct(InteractionEvent.product_id)),
                )
            )
            total_interactions, unique_users, unique_products = totals_result.one()

            action_rows = await session.execute(
                select(InteractionEvent.action, func.count(InteractionEvent.event_id))
                .group_by(InteractionEvent.action)
            )

        action_counts = Counter({action: count for action, count in action_rows.all()})
        clicks = action_counts.get("click", 0)
        purchases = action_counts.get("purchase", 0)
        views = action_counts.get("view", 0)
        ctr = clicks / max(views, 1)
        conversion_rate = purchases / max(clicks, 1)

        return {
            "total_interactions": int(total_interactions or 0),
            "unique_users": int(unique_users or 0),
            "unique_products": int(unique_products or 0),
            "action_counts": dict(action_counts),
            "ctr": round(ctr, 4),
            "conversion_rate": round(conversion_rate, 4),
            "timestamp": time.time(),
            "source": "postgres",
        }

    async def upsert_content_job(
        self,
        content_id: str,
        filename: Optional[str],
        storage_path: Optional[str],
        user_id: Optional[str],
        priority: str,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        stmt = pg_insert(ContentJob).values(
            content_id=content_id,
            filename=filename,
            storage_path=storage_path,
            user_id=user_id,
            priority=priority,
            status=status,
            error_message=error_message,
            payload=payload or {},
            updated_at=_utc_now(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ContentJob.content_id],
            set_={
                "filename": filename,
                "storage_path": storage_path,
                "user_id": user_id,
                "priority": priority,
                "status": status,
                "error_message": error_message,
                "payload": payload or {},
                "updated_at": _utc_now(),
            },
        )
        async with self.session_factory.begin() as session:
            await session.execute(stmt)

    async def update_content_job_status(
        self,
        content_id: str,
        status: str,
        *,
        error_message: Optional[str] = None,
        storage_path: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        async with self.session_factory.begin() as session:
            current = await session.get(ContentJob, content_id)
            if current is None:
                current = ContentJob(
                    content_id=content_id,
                    filename=None,
                    storage_path=storage_path,
                    user_id=None,
                    priority="normal",
                    status=status,
                    error_message=error_message,
                    payload=payload or {},
                    updated_at=_utc_now(),
                )
                session.add(current)
                return

            if storage_path is not None:
                current.storage_path = storage_path
            current.status = status
            current.error_message = error_message
            if payload is not None:
                current.payload = payload
            current.updated_at = _utc_now()

    async def get_content_job(self, content_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        async with self.session_factory() as session:
            row = await session.get(ContentJob, content_id)
        if row is None:
            return None
        return {
            "content_id": row.content_id,
            "filename": row.filename,
            "storage_path": row.storage_path,
            "user_id": row.user_id,
            "priority": row.priority,
            "status": row.status,
            "error_message": row.error_message,
            "payload": row.payload or {},
            "created_at": row.created_at.timestamp() if row.created_at else None,
            "updated_at": row.updated_at.timestamp() if row.updated_at else None,
        }

    async def store_product_catalog_snapshot_batch(
        self,
        metadata_map: Dict[str, Dict[str, Any]],
    ) -> None:
        if not self.enabled or not metadata_map:
            return

        rows = [
            {
                "product_id": product_id,
                "snapshot": metadata,
                "updated_at": _utc_now(),
            }
            for product_id, metadata in metadata_map.items()
        ]
        stmt = pg_insert(ProductCatalogSnapshot).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[ProductCatalogSnapshot.product_id],
            set_={
                "snapshot": stmt.excluded.snapshot,
                "updated_at": _utc_now(),
            },
        )
        async with self.session_factory.begin() as session:
            await session.execute(stmt)

    async def record_model_checkpoint(
        self,
        model_name: str,
        model_version: str,
        checkpoint_path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        async with self.session_factory.begin() as session:
            session.add(
                ModelCheckpoint(
                    model_name=model_name,
                    model_version=model_version,
                    checkpoint_path=checkpoint_path,
                    payload=payload or {},
                )
            )

    async def get_latest_model_checkpoint(self, model_name: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        async with self.session_factory() as session:
            result = await session.execute(
                select(ModelCheckpoint)
                .where(ModelCheckpoint.model_name == model_name)
                .order_by(desc(ModelCheckpoint.created_at), desc(ModelCheckpoint.id))
                .limit(1)
            )
            row = result.scalar_one_or_none()

        if row is None:
            return None

        return {
            "id": row.id,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "checkpoint_path": row.checkpoint_path,
            "payload": row.payload or {},
            "created_at": row.created_at.timestamp() if row.created_at else None,
        }
