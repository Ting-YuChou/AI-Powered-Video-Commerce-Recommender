"""Disk-backed, deterministic PIT row sources for bounded-memory training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import uuid
from typing import Any, Callable, Iterable, Iterator, Mapping


PIT_STREAM_CONTRACT_VERSION = "pit_stream_v1"
VALUE_NORMALIZATION_ALGORITHM_VERSION = "exact_linear_quantile_welford_v1"


class PitSpoolError(ValueError):
    """Raised when an immutable PIT generation cannot be verified."""


def iter_arrow_batch_rows(batch) -> Iterator[dict[str, Any]]:
    """Yield Arrow rows one at a time without allocating a batch-sized list."""
    names = tuple(str(name) for name in batch.schema.names)
    columns = tuple(batch.column(index) for index in range(len(names)))
    for row_index in range(batch.num_rows):
        yield {name: column[row_index].as_py() for name, column in zip(names, columns)}


def _stable_order(group_id: str, seed: int, epoch: int) -> int:
    payload = f"{int(seed)}:{int(epoch)}:{group_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & (
        (1 << 63) - 1
    )


class PitSpoolBuilder:
    """Build one generation privately and publish it with one atomic rename."""

    def __init__(
        self,
        root: str | Path,
        *,
        generation_id: str,
        kind: str,
        max_group_rows: int,
        max_bytes: int,
        lineage: Mapping[str, Any] | None = None,
    ) -> None:
        if not generation_id or kind not in {"ranking", "retrieval"}:
            raise PitSpoolError("PIT spool generation and kind are required")
        if int(max_group_rows) <= 0 or int(max_bytes) <= 0:
            raise PitSpoolError("PIT spool limits must be positive")
        self.root = Path(root)
        self.generation_id = str(generation_id)
        self.kind = str(kind)
        self.max_group_rows = int(max_group_rows)
        self.max_bytes = int(max_bytes)
        self.lineage = dict(lineage or {})
        self._staging = self.root / f".staging-{self.generation_id}-{uuid.uuid4().hex}"
        self._target = self.root / "verified" / self.generation_id
        self._staging.mkdir(parents=True, exist_ok=False)
        self._connection = sqlite3.connect(self._staging / "rows.sqlite3")
        self._connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE rows (
                sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                row_id TEXT NOT NULL,
                as_of_ts REAL NOT NULL,
                payload_json TEXT NOT NULL,
                business_value REAL,
                value_bucket INTEGER,
                is_purchase INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE groups (
                group_id TEXT PRIMARY KEY,
                max_as_of_ts REAL NOT NULL,
                row_count INTEGER NOT NULL,
                split TEXT NOT NULL DEFAULT 'all'
            );
            CREATE INDEX rows_group_idx ON rows(group_id, row_id, sequence_id);
            CREATE INDEX groups_split_time_idx
                ON groups(split, max_as_of_ts, group_id);
            """
        )
        self._closed = False

    @property
    def staging_path(self) -> Path:
        return self._staging

    def add_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self._ensure_open()
        for raw in rows:
            group_id = str(raw.get("group_id") or "").strip()
            row_id = str(raw.get("row_id") or "").strip()
            try:
                as_of_ts = float(raw["as_of_ts"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PitSpoolError("PIT row has invalid as_of_ts") from exc
            if not group_id or not row_id or not math.isfinite(as_of_ts):
                raise PitSpoolError("PIT row identity and as_of_ts are required")
            payload = raw.get("payload")
            if not isinstance(payload, Mapping):
                raise PitSpoolError("PIT row payload must be an object")
            try:
                payload_json = json.dumps(
                    dict(payload),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise PitSpoolError("PIT row payload is not canonical JSON") from exc
            business_value = raw.get("business_value")
            if business_value is not None:
                business_value = float(business_value)
                if not math.isfinite(business_value):
                    raise PitSpoolError("PIT business value must be finite")
            self._connection.execute(
                """
                INSERT INTO rows(
                    group_id, row_id, as_of_ts, payload_json,
                    business_value, value_bucket, is_purchase
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group_id,
                    row_id,
                    as_of_ts,
                    payload_json,
                    business_value,
                    int(raw.get("value_bucket", -1)),
                    1 if raw.get("is_purchase") else 0,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO groups(group_id, max_as_of_ts, row_count)
                VALUES (?, ?, 1)
                ON CONFLICT(group_id) DO UPDATE SET
                    max_as_of_ts = MAX(groups.max_as_of_ts, excluded.max_as_of_ts),
                    row_count = groups.row_count + 1
                """,
                (group_id, as_of_ts),
            )
        self._connection.commit()
        self._check_size()

    def finalize(self, *, expected_rows: int) -> "VerifiedPitSource":
        self._ensure_open()
        try:
            row_count = int(
                self._connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
            )
            if row_count != int(expected_rows):
                raise PitSpoolError(
                    f"PIT row count mismatch: expected {expected_rows}, got {row_count}"
                )
            largest = self._connection.execute(
                "SELECT group_id, row_count FROM groups ORDER BY row_count DESC LIMIT 1"
            ).fetchone()
            if largest and int(largest[1]) > self.max_group_rows:
                raise PitSpoolError(
                    "PIT group exceeds maximum row count: "
                    f"group={largest[0]} rows={largest[1]} max={self.max_group_rows}"
                )
            group_count = int(
                self._connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
            )
            self._connection.execute("PRAGMA optimize")
            self._connection.commit()
            self._connection.close()
            self._closed = True
            metadata = {
                "contract_version": PIT_STREAM_CONTRACT_VERSION,
                "generation_id": self.generation_id,
                "kind": self.kind,
                "row_count": row_count,
                "group_count": group_count,
                "max_group_rows": self.max_group_rows,
                "largest_group_rows": int(largest[1]) if largest else 0,
                "lineage": self.lineage,
            }
            (self._staging / "metadata.json").write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            self._check_size()
            self._target.parent.mkdir(parents=True, exist_ok=True)
            if self._target.exists():
                existing = VerifiedPitSource.open(self._target)
                if existing.metadata != metadata:
                    raise PitSpoolError("verified PIT generation conflicts with retry")
                shutil.rmtree(self._staging, ignore_errors=True)
                return existing
            os.replace(self._staging, self._target)
            return VerifiedPitSource.open(self._target)
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True
        shutil.rmtree(self._staging, ignore_errors=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise PitSpoolError("PIT spool builder is closed")

    def _check_size(self) -> None:
        size = sum(
            path.stat().st_size for path in self._staging.rglob("*") if path.is_file()
        )
        if size > self.max_bytes:
            raise PitSpoolError(
                f"PIT staging exceeds byte limit: bytes={size} max={self.max_bytes}"
            )


@dataclass(frozen=True)
class VerifiedPitSource:
    path: Path
    metadata: Mapping[str, Any]

    @classmethod
    def open(cls, path: str | Path) -> "VerifiedPitSource":
        root = Path(path)
        try:
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PitSpoolError("verified PIT metadata is unavailable") from exc
        if metadata.get("contract_version") != PIT_STREAM_CONTRACT_VERSION:
            raise PitSpoolError("verified PIT contract version is incompatible")
        if not (root / "rows.sqlite3").is_file():
            raise PitSpoolError("verified PIT row store is unavailable")
        return cls(root, metadata)

    @property
    def row_count(self) -> int:
        return int(self.metadata["row_count"])

    @property
    def group_count(self) -> int:
        return int(self.metadata["group_count"])

    @property
    def staging_bytes(self) -> int:
        return sum(
            path.stat().st_size for path in self.path.rglob("*") if path.is_file()
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path / 'rows.sqlite3'}?mode=rw", uri=True)

    def assign_ranking_splits(
        self,
        *,
        holdout_days: int,
        attribution_cutoff: float,
        validation_fraction: float,
    ) -> None:
        if int(holdout_days) <= 0 or not 0.0 <= float(validation_fraction) < 1.0:
            raise PitSpoolError("PIT split settings are invalid")
        connection = self._connect()
        try:
            maximum = connection.execute(
                "SELECT MAX(max_as_of_ts) FROM groups"
            ).fetchone()[0]
            if maximum is None or float(maximum) > float(attribution_cutoff):
                raise PitSpoolError("PIT holdout exceeds attribution cutoff")
            holdout_start = float(maximum) - int(holdout_days) * 86400.0
            connection.execute("UPDATE groups SET split='training'")
            connection.execute(
                "UPDATE groups SET split='holdout' WHERE max_as_of_ts >= ?",
                (holdout_start,),
            )
            pre_holdout = int(
                connection.execute(
                    "SELECT COUNT(*) FROM groups WHERE split='training'"
                ).fetchone()[0]
            )
            if pre_holdout >= 2 and validation_fraction > 0.0:
                validation_count = min(
                    pre_holdout - 1,
                    max(1, int(pre_holdout * float(validation_fraction))),
                )
                connection.execute(
                    """
                    UPDATE groups SET split='validation' WHERE group_id IN (
                        SELECT group_id FROM groups WHERE split='training'
                        ORDER BY max_as_of_ts DESC, group_id DESC LIMIT ?
                    )
                    """,
                    (validation_count,),
                )
            connection.commit()
        finally:
            connection.close()

    def group_ids(self, split: str) -> tuple[str, ...]:
        connection = self._connect()
        try:
            if split == "all":
                rows = connection.execute(
                    "SELECT group_id FROM groups ORDER BY max_as_of_ts, group_id"
                )
            else:
                rows = connection.execute(
                    "SELECT group_id FROM groups WHERE split=? "
                    "ORDER BY max_as_of_ts, group_id",
                    (str(split),),
                )
            return tuple(str(row[0]) for row in rows)
        finally:
            connection.close()

    def row_count_for_split(self, split: str) -> int:
        connection = self._connect()
        try:
            if split == "all":
                return int(
                    connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
                )
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM rows r JOIN groups g "
                    "ON g.group_id=r.group_id WHERE g.split=?",
                    (str(split),),
                ).fetchone()[0]
            )
        finally:
            connection.close()

    def reindex_value_buckets(
        self,
        resolver: Callable[[Mapping[str, Any]], int],
    ) -> None:
        """Assign model-specific value buckets without materializing PIT rows."""
        connection = self._connect()
        try:
            cursor = connection.execute("SELECT sequence_id, payload_json FROM rows")
            updates = []
            for sequence_id, payload_json in cursor:
                payload = json.loads(payload_json)
                updates.append((int(resolver(payload)), int(sequence_id)))
                if len(updates) >= 1024:
                    connection.executemany(
                        "UPDATE rows SET value_bucket=? WHERE sequence_id=?", updates
                    )
                    updates.clear()
            if updates:
                connection.executemany(
                    "UPDATE rows SET value_bucket=? WHERE sequence_id=?", updates
                )
            connection.commit()
        finally:
            connection.close()

    def iter_group_batches(
        self,
        *,
        split: str,
        batch_rows: int,
        seed: int,
        epoch: int = 0,
    ) -> Iterator[list[dict[str, Any]]]:
        if int(batch_rows) <= 0:
            raise PitSpoolError("PIT batch size must be positive")
        connection = self._connect()
        connection.create_function(
            "pit_order",
            1,
            lambda group_id: _stable_order(str(group_id), int(seed), int(epoch)),
            deterministic=True,
        )
        where = "" if split == "all" else "WHERE g.split=?"
        parameters = () if split == "all" else (str(split),)
        cursor = connection.execute(
            f"""
            SELECT r.group_id, r.payload_json
            FROM groups g JOIN rows r ON r.group_id=g.group_id
            {where}
            ORDER BY pit_order(g.group_id), g.group_id, r.row_id, r.sequence_id
            """,
            parameters,
        )
        pending: list[dict[str, Any]] = []
        current_group: str | None = None
        current_rows: list[dict[str, Any]] = []
        try:
            for group_id, payload_json in cursor:
                group_id = str(group_id)
                if current_group is not None and group_id != current_group:
                    if pending and len(pending) + len(current_rows) > int(batch_rows):
                        yield pending
                        pending = []
                    pending.extend(current_rows)
                    current_rows = []
                current_group = group_id
                payload = dict(json.loads(payload_json))
                payload.setdefault("group_id", group_id)
                current_rows.append(payload)
            if current_rows:
                if pending and len(pending) + len(current_rows) > int(batch_rows):
                    yield pending
                    pending = []
                pending.extend(current_rows)
            if pending:
                yield pending
        finally:
            connection.close()

    def compute_value_transform_stats(
        self,
        *,
        split: str,
        quantile: float,
        min_bucket_purchases: int = 20,
    ) -> dict[str, Any]:
        if not 0.0 <= float(quantile) <= 1.0:
            raise PitSpoolError("value quantile must be between zero and one")
        connection = self._connect()
        try:
            global_stats = self._value_stats_for_bucket(
                connection, split=split, quantile=float(quantile), bucket=None
            )
            where, parameters = self._split_where(split)
            bucket_rows = connection.execute(
                "SELECT r.value_bucket, COUNT(*) FROM rows r "
                "JOIN groups g ON g.group_id=r.group_id "
                f"WHERE r.is_purchase=1 {where} GROUP BY r.value_bucket",
                parameters,
            )
            buckets = {}
            for bucket_id, count in bucket_rows:
                if int(count) >= int(min_bucket_purchases):
                    buckets[str(int(bucket_id))] = self._value_stats_for_bucket(
                        connection,
                        split=split,
                        quantile=float(quantile),
                        bucket=int(bucket_id),
                    )
            return {
                "global": global_stats,
                "buckets": buckets,
                "algorithm_version": VALUE_NORMALIZATION_ALGORITHM_VERSION,
            }
        finally:
            connection.close()

    def value_mask_coverage(self, split: str) -> float:
        connection = self._connect()
        try:
            where, parameters = self._split_where(split)
            row = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(r.is_purchase), 0) FROM rows r "
                "JOIN groups g ON g.group_id=r.group_id "
                f"WHERE 1=1 {where}",
                parameters,
            ).fetchone()
            return float(row[1]) / max(int(row[0]), 1)
        finally:
            connection.close()

    @staticmethod
    def _split_where(split: str) -> tuple[str, tuple[Any, ...]]:
        if split == "all":
            return "", ()
        return "AND g.split=?", (str(split),)

    def _value_stats_for_bucket(
        self,
        connection: sqlite3.Connection,
        *,
        split: str,
        quantile: float,
        bucket: int | None,
    ) -> dict[str, Any]:
        split_clause, parameters = self._split_where(split)
        bucket_clause = "" if bucket is None else "AND r.value_bucket=?"
        all_parameters = parameters + (() if bucket is None else (int(bucket),))
        base = (
            "FROM rows r JOIN groups g ON g.group_id=r.group_id "
            f"WHERE r.is_purchase=1 {split_clause} {bucket_clause}"
        )
        count = int(
            connection.execute(f"SELECT COUNT(*) {base}", all_parameters).fetchone()[0]
        )
        if count == 0:
            return {"count": 0, "clip": 0.0, "mean": 0.0, "std": 1.0}
        position = (count - 1) * float(quantile)
        lower_index = int(math.floor(position))
        upper_index = int(math.ceil(position))
        lower = float(
            connection.execute(
                f"SELECT MAX(0.0, r.business_value) {base} "
                "ORDER BY MAX(0.0, r.business_value) LIMIT 1 OFFSET ?",
                all_parameters + (lower_index,),
            ).fetchone()[0]
        )
        upper = float(
            connection.execute(
                f"SELECT MAX(0.0, r.business_value) {base} "
                "ORDER BY MAX(0.0, r.business_value) LIMIT 1 OFFSET ?",
                all_parameters + (upper_index,),
            ).fetchone()[0]
        )
        clip = lower + (upper - lower) * (position - lower_index)
        mean = 0.0
        m2 = 0.0
        seen = 0
        for (value,) in connection.execute(
            f"SELECT MAX(0.0, r.business_value) {base}", all_parameters
        ):
            logged = math.log1p(min(float(value), clip))
            seen += 1
            delta = logged - mean
            mean += delta / seen
            m2 += delta * (logged - mean)
        std = math.sqrt(m2 / seen) if seen else 1.0
        if not math.isfinite(std) or std < 1e-6:
            std = 1.0
        return {"count": seen, "clip": clip, "mean": mean, "std": std}
