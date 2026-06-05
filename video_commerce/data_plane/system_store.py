"""
Durable Postgres-backed system store for operational state and training data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, delete, desc, event, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from video_commerce.common.config import DatabaseConfig

logger = logging.getLogger(__name__)

POSITIVE_SEQUENCE_ACTIONS = ("view", "click", "add_to_cart", "purchase")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _event_value(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _event_to_sequence_dict(event: Any) -> Dict[str, Any]:
    occurred_at = _coerce_datetime(_event_value(event, "occurred_at")) or _utc_now()
    return {
        "event_id": _event_value(event, "event_id"),
        "schema_version": int(_event_value(event, "schema_version", 1) or 1),
        "request_id": _event_value(event, "request_id"),
        "user_id": _event_value(event, "user_id"),
        "product_id": _event_value(event, "product_id"),
        "action": _event_value(event, "action"),
        "context": _event_value(event, "context", {}) or {},
        "timestamp": occurred_at.timestamp(),
        "occurred_at": occurred_at.timestamp(),
    }


def build_chronological_user_sequences(
    events: Iterable[Any],
    *,
    max_events_per_user: int,
    min_sequence_length: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build bounded positive product sequences keyed by user id."""
    if max_events_per_user <= 0:
        return {}

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        user_id = _event_value(event, "user_id")
        product_id = _event_value(event, "product_id")
        action = _event_value(event, "action")
        if not user_id or not product_id or action not in POSITIVE_SEQUENCE_ACTIONS:
            continue
        grouped.setdefault(user_id, []).append(_event_to_sequence_dict(event))

    min_sequence_length = max(1, int(min_sequence_length))
    bounded: Dict[str, List[Dict[str, Any]]] = {}
    for user_id, sequence in grouped.items():
        sequence.sort(
            key=lambda item: (
                float(item.get("occurred_at") or 0.0),
                str(item.get("event_id") or ""),
            )
        )
        sequence = sequence[-max_events_per_user:]
        if len(sequence) >= min_sequence_length:
            bounded[user_id] = sequence
    return bounded


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
    "ix_interaction_events_action_occurred_at",
    InteractionEvent.action,
    InteractionEvent.occurred_at.desc(),
)
Index(
    "ix_interaction_events_user_sequence",
    InteractionEvent.user_id,
    InteractionEvent.occurred_at,
    InteractionEvent.event_id,
)
Index(
    "ix_interaction_events_positive_user_sequence",
    InteractionEvent.user_id,
    InteractionEvent.occurred_at.desc(),
    InteractionEvent.event_id.desc(),
    postgresql_where=InteractionEvent.action.in_(POSITIVE_SEQUENCE_ACTIONS),
)
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

    def __init__(self, config: DatabaseConfig, observability=None):
        self.config = config
        self.observability = observability
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker | None = None
        self.is_connected = False
        self._retention_cleanup_task: asyncio.Task | None = None
        self._analytics_summary_cache: Dict[int, tuple[float, Dict[str, Any]]] = {}

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
        self._install_metrics_listeners()
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

        if self.config.auto_create_schema:
            async with self.engine.begin() as conn:
                if self.config.interaction_events_partitioned:
                    await conn.run_sync(
                        lambda sync_conn: Base.metadata.create_all(
                            sync_conn,
                            tables=[
                                table
                                for table in Base.metadata.sorted_tables
                                if table.name != InteractionEvent.__tablename__
                            ],
                        )
                    )
                    await self._ensure_partitioned_interaction_events(conn)
                else:
                    await conn.run_sync(Base.metadata.create_all)
                await self._ensure_operational_indexes(conn)

        self.is_connected = True
        self._update_pool_metrics()
        if self.config.enable_retention_cleanup:
            await self.prune_interaction_events()
            self._retention_cleanup_task = asyncio.create_task(
                self._run_periodic_retention_cleanup(),
                name="postgres-interaction-retention-cleanup",
            )

    async def _ensure_operational_indexes(self, conn) -> None:
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_interaction_events_occurred_at_desc "
                "ON interaction_events (occurred_at DESC)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_interaction_events_action_occurred_at "
                "ON interaction_events (action, occurred_at DESC)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_interaction_events_user_sequence "
                "ON interaction_events (user_id, occurred_at, event_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_interaction_events_positive_user_sequence "
                "ON interaction_events (user_id, occurred_at DESC, event_id DESC) "
                "WHERE action IN ('view', 'click', 'add_to_cart', 'purchase')"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_model_checkpoints_latest "
                "ON model_checkpoints (model_name, created_at DESC, id DESC)"
            )
        )

    async def _ensure_partitioned_interaction_events(self, conn) -> None:
        relation_result = await conn.execute(
            text(
                """
                SELECT c.relkind
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema()
                  AND c.relname = 'interaction_events'
                """
            )
        )
        relkind = relation_result.scalar_one_or_none()
        if relkind == "r":
            raise RuntimeError(
                "interaction_events already exists as a non-partitioned table; "
                "run migrations/postgres/001_partition_interaction_events.sql before "
                "setting DATABASE_INTERACTION_EVENTS_PARTITIONED=true"
            )

        if relkind is None:
            await conn.execute(
                text(
                    """
                    CREATE TABLE interaction_events (
                        event_id VARCHAR(64) NOT NULL,
                        schema_version INTEGER NOT NULL DEFAULT 1,
                        request_id VARCHAR(64),
                        user_id VARCHAR(255) NOT NULL,
                        product_id VARCHAR(255) NOT NULL,
                        action VARCHAR(64) NOT NULL,
                        context JSON NOT NULL DEFAULT '{}'::json,
                        occurred_at TIMESTAMPTZ NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        PRIMARY KEY (event_id, occurred_at)
                    ) PARTITION BY RANGE (occurred_at)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS interaction_events_default
                    PARTITION OF interaction_events DEFAULT
                    """
                )
            )
        elif relkind != "p":
            raise RuntimeError(
                f"interaction_events has unsupported Postgres relkind {relkind!r}"
            )

        await self._ensure_interaction_event_partitions(conn)

    async def _ensure_interaction_event_partitions(self, conn) -> None:
        backfill_months = max(0, int(self.config.partition_backfill_months))
        premake_months = max(1, int(self.config.partition_premake_months))
        await conn.execute(
            text(
                f"""
                DO $$
                DECLARE
                    partition_start DATE := date_trunc('month', now())::date - INTERVAL '{backfill_months} months';
                    partition_end DATE := date_trunc('month', now())::date + INTERVAL '{premake_months} months';
                    current_start DATE;
                    current_end DATE;
                    partition_name TEXT;
                BEGIN
                    current_start := partition_start;
                    WHILE current_start < partition_end LOOP
                        current_end := current_start + INTERVAL '1 month';
                        partition_name := format('interaction_events_%s', to_char(current_start, 'YYYY_MM'));
                        EXECUTE format(
                            'CREATE TABLE IF NOT EXISTS %I PARTITION OF interaction_events FOR VALUES FROM (%L) TO (%L)',
                            partition_name,
                            current_start,
                            current_end
                        );
                        current_start := current_end;
                    END LOOP;
                END $$;
                """
            )
        )

    async def close(self) -> None:
        if self._retention_cleanup_task:
            self._retention_cleanup_task.cancel()
            try:
                await self._retention_cleanup_task
            except asyncio.CancelledError:
                pass
            self._retention_cleanup_task = None
        if self.engine is not None:
            await self.engine.dispose()
        self.engine = None
        self.session_factory = None
        self.is_connected = False
        self._analytics_summary_cache.clear()

    def _install_metrics_listeners(self) -> None:
        if self.engine is None or self.observability is None:
            return

        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault("video_commerce_query_start_time", []).append(time.perf_counter())

        @event.listens_for(self.engine.sync_engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            started_stack = conn.info.get("video_commerce_query_start_time", [])
            started_at = started_stack.pop(-1) if started_stack else time.perf_counter()
            operation = _sql_operation(statement)
            self.observability.record_database_query(
                operation,
                time.perf_counter() - started_at,
                "success",
            )

        @event.listens_for(self.engine.sync_engine, "handle_error")
        def handle_error(exception_context):
            operation = _sql_operation(exception_context.statement)
            self.observability.record_database_query(operation, 0.0, "error")

    def _update_pool_metrics(self) -> None:
        if self.observability is not None:
            self.observability.update_database_pool_metrics(self)

    async def health_check(self) -> DatabaseHealth:
        if not self.enabled:
            return DatabaseHealth(status="healthy", response_time_ms=0.0)

        started_at = time.time()
        try:
            async with self.session_factory() as session:
                await session.execute(text("SELECT 1"))
            self._update_pool_metrics()
            return DatabaseHealth(
                status="healthy",
                response_time_ms=round((time.time() - started_at) * 1000, 2),
            )
        except Exception as exc:
            self._update_pool_metrics()
            return DatabaseHealth(
                status="unhealthy",
                response_time_ms=round((time.time() - started_at) * 1000, 2),
                error=str(exc),
            )

    async def prune_interaction_events(self) -> int:
        """Delete raw interaction events older than the configured retention window."""
        if not self.enabled:
            return 0
        retention_days = int(self.config.interaction_retention_days)
        if retention_days <= 0:
            return 0

        cutoff = _utc_now() - timedelta(days=retention_days)
        stmt = delete(InteractionEvent).where(InteractionEvent.occurred_at < cutoff)
        async with self.session_factory.begin() as session:
            result = await session.execute(stmt)
        deleted_count = int(result.rowcount or 0)
        self._update_pool_metrics()
        return deleted_count

    async def _run_periodic_retention_cleanup(self) -> None:
        interval_seconds = max(60, int(self.config.retention_cleanup_interval_seconds))
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                deleted_count = await self.prune_interaction_events()
                if deleted_count:
                    logger.info(
                        "interaction_events_retention_deleted",
                        extra={"deleted_count": deleted_count},
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("interaction_events_retention_cleanup_failed: %s", exc)
                await asyncio.sleep(min(interval_seconds, 300))

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
        if self.config.interaction_events_partitioned:
            stmt = stmt.on_conflict_do_nothing()
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=[InteractionEvent.event_id])
        async with self.session_factory.begin() as session:
            await session.execute(stmt)
        self._update_pool_metrics()

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
        self._update_pool_metrics()

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

    async def get_user_training_sequences(
        self,
        *,
        max_users: int = 10000,
        max_events_per_user: int = 200,
        since: Optional[Any] = None,
        min_sequence_length: int = 2,
        actions: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return bounded positive chronological interaction sequences per user."""
        if not self.enabled or max_users <= 0 or max_events_per_user <= 0:
            return {}

        selected_actions = tuple(actions or POSITIVE_SEQUENCE_ACTIONS)
        if not selected_actions:
            return {}

        conditions = [InteractionEvent.action.in_(selected_actions)]
        since_dt = _coerce_datetime(since)
        if since_dt is not None:
            conditions.append(InteractionEvent.occurred_at >= since_dt)

        min_sequence_length = max(1, int(min_sequence_length))
        eligible_users = (
            select(InteractionEvent.user_id)
            .where(*conditions)
            .group_by(InteractionEvent.user_id)
            .having(func.count(InteractionEvent.event_id) >= min_sequence_length)
            .order_by(func.max(InteractionEvent.occurred_at).desc(), InteractionEvent.user_id.asc())
            .limit(max_users)
            .subquery()
        )
        event_rank = func.row_number().over(
            partition_by=InteractionEvent.user_id,
            order_by=(InteractionEvent.occurred_at.desc(), InteractionEvent.event_id.desc()),
        ).label("event_rank")
        ranked_events = (
            select(
                InteractionEvent.event_id,
                InteractionEvent.schema_version,
                InteractionEvent.request_id,
                InteractionEvent.user_id,
                InteractionEvent.product_id,
                InteractionEvent.action,
                InteractionEvent.context,
                InteractionEvent.occurred_at,
                event_rank,
            )
            .where(
                InteractionEvent.user_id.in_(select(eligible_users.c.user_id)),
                *conditions,
            )
            .subquery()
        )
        stmt = (
            select(ranked_events)
            .where(ranked_events.c.event_rank <= max_events_per_user)
            .order_by(
                ranked_events.c.user_id.asc(),
                ranked_events.c.occurred_at.asc(),
                ranked_events.c.event_id.asc(),
            )
        )

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            rows = result.mappings().all()
        self._update_pool_metrics()

        return build_chronological_user_sequences(
            rows,
            max_events_per_user=max_events_per_user,
            min_sequence_length=min_sequence_length,
        )

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

        window_hours = max(1, int(self.config.analytics_window_hours))
        ttl_seconds = max(0, int(self.config.analytics_summary_cache_ttl_seconds))
        now = time.time()
        if ttl_seconds > 0:
            cached = self._analytics_summary_cache.get(window_hours)
            if cached is not None:
                cached_at, cached_summary = cached
                if now - cached_at < ttl_seconds:
                    return _copy_analytics_summary(cached_summary)

        cutoff = _utc_now() - timedelta(hours=window_hours)

        async with self.session_factory() as session:
            totals_result = await session.execute(
                select(
                    func.count(InteractionEvent.event_id),
                    func.count(func.distinct(InteractionEvent.user_id)),
                    func.count(func.distinct(InteractionEvent.product_id)),
                )
                .where(InteractionEvent.occurred_at >= cutoff)
            )
            total_interactions, unique_users, unique_products = totals_result.one()

            action_rows = await session.execute(
                select(InteractionEvent.action, func.count(InteractionEvent.event_id))
                .where(InteractionEvent.occurred_at >= cutoff)
                .group_by(InteractionEvent.action)
            )

        action_counts = Counter({action: count for action, count in action_rows.all()})
        self._update_pool_metrics()
        clicks = action_counts.get("click", 0)
        purchases = action_counts.get("purchase", 0)
        views = action_counts.get("view", 0)
        ctr = clicks / max(views, 1)
        conversion_rate = purchases / max(clicks, 1)

        summary = {
            "total_interactions": int(total_interactions or 0),
            "unique_users": int(unique_users or 0),
            "unique_products": int(unique_products or 0),
            "action_counts": dict(action_counts),
            "ctr": round(ctr, 4),
            "conversion_rate": round(conversion_rate, 4),
            "timestamp": now,
            "source": "postgres",
            "window_hours": window_hours,
        }
        if ttl_seconds > 0:
            self._analytics_summary_cache[window_hours] = (
                now,
                _copy_analytics_summary(summary),
            )
        return summary

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
        self._update_pool_metrics()

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

        updated_at = _utc_now()
        stmt = pg_insert(ContentJob).values(
            content_id=content_id,
            filename=None,
            storage_path=storage_path,
            user_id=None,
            priority="normal",
            status=status,
            error_message=error_message,
            payload=payload or {},
            updated_at=updated_at,
        )
        update_values = {
            "status": status,
            "error_message": error_message,
            "updated_at": updated_at,
        }
        if storage_path is not None:
            update_values["storage_path"] = storage_path
        if payload is not None:
            update_values["payload"] = payload
        stmt = stmt.on_conflict_do_update(
            index_elements=[ContentJob.content_id],
            set_=update_values,
        )
        async with self.session_factory.begin() as session:
            await session.execute(stmt)
        self._update_pool_metrics()

    async def get_content_job(self, content_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        async with self.session_factory() as session:
            row = await session.get(ContentJob, content_id)
        self._update_pool_metrics()
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
        self._update_pool_metrics()

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
        self._update_pool_metrics()

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
        self._update_pool_metrics()

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


def _sql_operation(statement: Optional[str]) -> str:
    if not statement:
        return "unknown"
    return statement.strip().split(None, 1)[0].lower() or "unknown"


def _copy_analytics_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(summary)
    copied["action_counts"] = dict(summary.get("action_counts") or {})
    return copied
