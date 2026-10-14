"""Frozen categorical feature contract for the ranking v3 challenger."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import torch
import numpy as np
from torch import nn


FEATURE_CONTRACT_VERSION = "ranking_ltr_v3_categorical_sources"
PAD_ID = 0
MISSING_ID = 1
OOV_ID = 2
_FIRST_VALUE_ID = 3

FIELD_CAPACITIES = {
    "category": 4096,
    "brand": 16384,
    "tags": 32768,
    "device": 32,
    "surface": 64,
    "time_of_day": 16,
}
MULTI_HOT_LIMITS = {"preferred_categories": 10, "tags": 20}
PACKED_INPUT_DIM = 40
PACKED_FIELDS = (
    "category",
    "brand",
    "device",
    "surface",
    "time_of_day",
    "request_category",
)


def pack_categorical_inputs(vocabulary, records):
    rows = []
    for record in records:
        encoded = vocabulary.encode_record(record)
        rows.append(
            [encoded[field] for field in PACKED_FIELDS]
            + encoded["preferred_categories"]
            + encoded["tags"]
            + [
                encoded["price_cross"],
                encoded["retrieval_source_overlap"],
                encoded["preferred_category_match"],
                encoded["request_category_match"],
            ]
        )
    return np.asarray(rows, dtype=np.float32).reshape(-1, PACKED_INPUT_DIM)


def unpack_categorical_inputs(values: torch.Tensor):
    identifiers = values[:, :36].to(torch.long)
    inputs = {field: identifiers[:, index] for index, field in enumerate(PACKED_FIELDS)}
    inputs["preferred_categories"] = identifiers[:, 6:16]
    inputs["tags"] = identifiers[:, 16:36]
    inputs["price_cross"] = values[:, 36:37]
    inputs["retrieval_source_overlap"] = values[:, 37:38]
    inputs["preferred_category_match"] = values[:, 38:39]
    inputs["request_category_match"] = values[:, 39:40]
    return inputs


@dataclass(frozen=True)
class DenseFeatureNormalization:
    lower: Sequence[float]
    upper: Sequence[float]
    mean: Sequence[float]
    std: Sequence[float]

    @classmethod
    def fit(cls, vectors: Iterable[Sequence[float]], *, dimension: int):
        """Exact linear quantiles and clipped Welford statistics using disk sort."""
        with tempfile.TemporaryDirectory(prefix="ranking-normalization-") as directory:
            connection = sqlite3.connect(f"{directory}/values.sqlite3")
            try:
                connection.execute(
                    "CREATE TABLE values_by_feature (feature INTEGER, value REAL)"
                )
                for vector in vectors:
                    values = np.asarray(vector, dtype=np.float64)
                    if values.shape != (dimension,):
                        raise ValueError("normalization feature dimension mismatch")
                    connection.executemany(
                        "INSERT INTO values_by_feature VALUES (?,?)",
                        [
                            (index, float(value))
                            for index, value in enumerate(values)
                            if np.isfinite(value)
                        ],
                    )
                connection.execute(
                    "CREATE INDEX feature_value ON values_by_feature(feature,value)"
                )
                lower, upper, means, deviations = [], [], [], []
                for index in range(dimension):
                    count = connection.execute(
                        "SELECT COUNT(*) FROM values_by_feature WHERE feature=?",
                        (index,),
                    ).fetchone()[0]
                    if not count:
                        lower.append(0.0)
                        upper.append(0.0)
                        means.append(0.0)
                        deviations.append(1.0)
                        continue

                    def quantile(q):
                        position = (count - 1) * q
                        lo, hi = math.floor(position), math.ceil(position)
                        a = connection.execute(
                            "SELECT value FROM values_by_feature WHERE feature=? ORDER BY value LIMIT 1 OFFSET ?",
                            (index, lo),
                        ).fetchone()[0]
                        b = connection.execute(
                            "SELECT value FROM values_by_feature WHERE feature=? ORDER BY value LIMIT 1 OFFSET ?",
                            (index, hi),
                        ).fetchone()[0]
                        return a + (b - a) * (position - lo)

                    low, high = quantile(0.01), quantile(0.99)
                    mean = m2 = 0.0
                    for n, (value,) in enumerate(
                        connection.execute(
                            "SELECT value FROM values_by_feature WHERE feature=?",
                            (index,),
                        ),
                        start=1,
                    ):
                        clipped = min(max(value, low), high)
                        delta = clipped - mean
                        mean += delta / n
                        m2 += delta * (clipped - mean)
                    lower.append(low)
                    upper.append(high)
                    means.append(mean)
                    deviations.append(max(math.sqrt(m2 / count), 1e-6))
                return cls(lower, upper, means, deviations)
            finally:
                connection.close()

    def apply(self, vectors):
        values = np.asarray(vectors, dtype=np.float32)
        missing = ~np.isfinite(values)
        filled = np.where(missing, np.asarray(self.mean), values)
        normalized = (np.clip(filled, self.lower, self.upper) - self.mean) / self.std
        return np.concatenate([normalized, missing.astype(np.float32)], axis=-1).astype(
            np.float32
        )

    def to_manifest(self):
        payload = {
            "algorithm_version": "clipped_welford_linear_quantile_v1",
            "lower": list(self.lower),
            "upper": list(self.upper),
            "mean": list(self.mean),
            "std": list(self.std),
        }
        payload["sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload

    @classmethod
    def from_manifest(cls, manifest):
        payload = dict(manifest)
        expected = payload.pop("sha256", "")
        if (
            hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            != expected
        ):
            raise ValueError("normalization checksum mismatch")
        if payload.get("algorithm_version") != "clipped_welford_linear_quantile_v1":
            raise ValueError("normalization algorithm is incompatible")
        arrays = [
            np.asarray(payload[key], dtype=float)
            for key in ("lower", "upper", "mean", "std")
        ]
        if (
            not arrays[0].size
            or any(
                array.shape != arrays[0].shape or not np.isfinite(array).all()
                for array in arrays
            )
            or (arrays[3] <= 0).any()
        ):
            raise ValueError("invalid normalization statistics")
        return cls(*[array.tolist() for array in arrays])


def categorical_record_from_bundle(bundle: Any) -> dict[str, Any]:
    metadata = dict(bundle.product_metadata)
    context = dict(bundle.context)
    candidate = bundle.candidate
    hour_bin = int((float(bundle.as_of_ts) % 86400.0) / 5400.0)
    signals = getattr(candidate, "retrieval_signals", {}) or {}
    return {
        "category": metadata.get("category"),
        "brand": metadata.get("brand"),
        "tags": metadata.get("tags"),
        "preferred_categories": list(bundle.user_features.preferred_categories[:10]),
        "request_category": context.get("product_category")
        or context.get("category")
        or context.get("request_category"),
        "device": context.get("device") or context.get("device_type"),
        "surface": context.get("surface") or context.get("placement"),
        "time_of_day": context.get("time_of_day", str(hour_bin)),
        "price_sensitivity": float(bundle.user_features.price_sensitivity),
        "normalized_log_price": math.log1p(
            max(float(metadata.get("price", 0.0) or 0.0), 0.0)
        ),
        "retrieval_source_overlap": len(signals)
        if signals
        else len(
            {part for part in str(getattr(candidate, "source", "")).split("+") if part}
        ),
    }


def _token(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


@dataclass(frozen=True)
class RankingCategoricalVocabulary:
    fields: Mapping[str, Mapping[str, int]]
    min_frequency: int = 5
    contract_version: str = FEATURE_CONTRACT_VERSION

    @classmethod
    def build(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        min_frequency: int = 5,
        capacities: Mapping[str, int] | None = None,
    ) -> "RankingCategoricalVocabulary":
        if int(min_frequency) < 1:
            raise ValueError(
                "categorical vocabulary minimum frequency must be positive"
            )
        limits = dict(FIELD_CAPACITIES)
        limits.update(
            {key: int(value) for key, value in dict(capacities or {}).items()}
        )
        if set(limits) != set(FIELD_CAPACITIES) or any(
            value < 1 for value in limits.values()
        ):
            raise ValueError("categorical vocabulary capacities are invalid")
        vocabularies: dict[str, dict[str, int]] = {}
        with tempfile.TemporaryDirectory(prefix="ranking-vocabulary-") as directory:
            connection = sqlite3.connect(f"{directory}/counts.sqlite3")
            try:
                connection.execute(
                    "CREATE TABLE counts (field TEXT, token TEXT, frequency INTEGER NOT NULL, "
                    "PRIMARY KEY(field,token)) WITHOUT ROWID"
                )
                for record in records:
                    counts = Counter()
                    for field in limits:
                        raw = record.get(field)
                        if field in MULTI_HOT_LIMITS and isinstance(
                            raw, (list, tuple, set)
                        ):
                            values = list(raw)[: MULTI_HOT_LIMITS[field]]
                        else:
                            values = [raw]
                        for value in values:
                            normalized = _token(value)
                            if normalized is not None:
                                counts[(field, normalized)] += 1
                    preferred = record.get("preferred_categories")
                    if isinstance(preferred, (list, tuple, set)):
                        for value in list(preferred)[:10]:
                            normalized = _token(value)
                            if normalized is not None:
                                counts[("category", normalized)] += 1
                    connection.executemany(
                        "INSERT INTO counts VALUES (?,?,?) ON CONFLICT(field,token) "
                        "DO UPDATE SET frequency=frequency+excluded.frequency",
                        [
                            (field, token, count)
                            for (field, token), count in counts.items()
                        ],
                    )
                connection.commit()
                for field, capacity in limits.items():
                    eligible = connection.execute(
                        "SELECT token FROM counts WHERE field=? AND frequency>=? "
                        "ORDER BY frequency DESC,token ASC LIMIT ?",
                        (field, int(min_frequency), int(capacity)),
                    )
                    vocabularies[field] = {
                        token: index + _FIRST_VALUE_ID
                        for index, (token,) in enumerate(eligible)
                    }
            finally:
                connection.close()
        return cls(fields=vocabularies, min_frequency=int(min_frequency))

    def lookup(self, field: str, value: Any) -> int:
        normalized = _token(value)
        if normalized is None:
            return MISSING_ID
        return int(self.fields.get(field, {}).get(normalized, OOV_ID))

    def encode_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        encoded: dict[str, Any] = {}
        for field in ("category", "brand", "device", "surface", "time_of_day"):
            encoded[field] = self.lookup(field, record.get(field))
        encoded["request_category"] = self.lookup(
            "category", record.get("request_category")
        )
        for field, limit in MULTI_HOT_LIMITS.items():
            raw = record.get(field)
            values = list(raw) if isinstance(raw, (list, tuple, set)) else []
            vocabulary_field = "category" if field == "preferred_categories" else field
            identifiers = [
                self.lookup(vocabulary_field, value) for value in values[:limit]
            ]
            if raw is None:
                identifiers = [MISSING_ID]
            identifiers.extend([PAD_ID] * (limit - len(identifiers)))
            encoded[field] = identifiers
        encoded["price_cross"] = float(
            record.get("price_sensitivity", 0.0) or 0.0
        ) * float(record.get("normalized_log_price", 0.0) or 0.0)
        encoded["retrieval_source_overlap"] = float(
            record.get("retrieval_source_overlap", 0.0) or 0.0
        )
        category = _token(record.get("category"))
        encoded["preferred_category_match"] = float(
            category is not None
            and category
            in {
                _token(value)
                for value in list(record.get("preferred_categories") or [])[:10]
            }
        )
        encoded["request_category_match"] = float(
            category is not None and category == _token(record.get("request_category"))
        )
        return encoded

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "contract_version": self.contract_version,
            "min_frequency": int(self.min_frequency),
            "special_ids": {"pad": PAD_ID, "missing": MISSING_ID, "oov": OOV_ID},
            "fields": {
                field: dict(sorted(values.items()))
                for field, values in sorted(self.fields.items())
            },
            "multi_hot_limits": dict(MULTI_HOT_LIMITS),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload

    @classmethod
    def from_manifest(
        cls, manifest: Mapping[str, Any]
    ) -> "RankingCategoricalVocabulary":
        payload = dict(manifest)
        expected = str(payload.pop("sha256", ""))
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if len(expected) != 64 or hashlib.sha256(canonical).hexdigest() != expected:
            raise ValueError("categorical vocabulary manifest checksum mismatch")
        if payload.get("contract_version") != FEATURE_CONTRACT_VERSION:
            raise ValueError("categorical vocabulary contract is incompatible")
        fields = dict(payload.get("fields") or {})
        if set(fields) != set(FIELD_CAPACITIES) or payload.get("special_ids") != {
            "pad": PAD_ID,
            "missing": MISSING_ID,
            "oov": OOV_ID,
        }:
            raise ValueError(
                "categorical vocabulary fields or special IDs are incompatible"
            )
        for field, mapping in fields.items():
            identifiers = sorted(dict(mapping).values())
            if (
                identifiers
                != list(range(_FIRST_VALUE_ID, _FIRST_VALUE_ID + len(identifiers)))
                or len(identifiers) > FIELD_CAPACITIES[field]
            ):
                raise ValueError("categorical vocabulary IDs are invalid")
        return cls(
            fields={
                str(field): {
                    str(key): int(value) for key, value in dict(values).items()
                }
                for field, values in dict(payload.get("fields") or {}).items()
            },
            min_frequency=int(payload.get("min_frequency", 5)),
        )

    def tensorize_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        encoded = [self.encode_record(record) for record in records]
        if not encoded:
            return {}
        tensors = {
            field: torch.tensor(
                [row[field] for row in encoded], dtype=torch.long, device=device
            )
            for field in (
                "category",
                "brand",
                "device",
                "surface",
                "time_of_day",
                "request_category",
                "preferred_categories",
                "tags",
            )
        }
        tensors["price_cross"] = torch.tensor(
            [[row["price_cross"]] for row in encoded],
            dtype=torch.float32,
            device=device,
        )
        tensors["retrieval_source_overlap"] = torch.tensor(
            [[row["retrieval_source_overlap"]] for row in encoded],
            dtype=torch.float32,
            device=device,
        )
        for field in ("preferred_category_match", "request_category_match"):
            tensors[field] = torch.tensor(
                [[row[field]] for row in encoded], dtype=torch.float32, device=device
            )
        return tensors


class RankingCategoricalEncoder(nn.Module):
    """Learned embeddings plus explicit v3 cross features."""

    output_dim = 124

    def __init__(self, vocabulary: RankingCategoricalVocabulary) -> None:
        super().__init__()
        self.vocabulary = vocabulary

        def size(field: str) -> int:
            values = vocabulary.fields.get(field, {})
            return max([OOV_ID, *values.values()]) + 1

        self.category = nn.Embedding(size("category"), 16, padding_idx=PAD_ID)
        self.brand = nn.Embedding(size("brand"), 16, padding_idx=PAD_ID)
        self.tags = nn.Embedding(size("tags"), 16, padding_idx=PAD_ID)
        self.device = nn.Embedding(size("device"), 8, padding_idx=PAD_ID)
        self.surface = nn.Embedding(size("surface"), 8, padding_idx=PAD_ID)
        self.time_of_day = nn.Embedding(size("time_of_day"), 8, padding_idx=PAD_ID)

    @staticmethod
    def _mean_pool(embedding: nn.Embedding, identifiers: torch.Tensor) -> torch.Tensor:
        values = embedding(identifiers)
        mask = identifiers.ne(PAD_ID).unsqueeze(-1)
        return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        category = self.category(inputs["category"])
        request_category = self.category(inputs["request_category"])
        preferred = self._mean_pool(self.category, inputs["preferred_categories"])
        tags = self._mean_pool(self.tags, inputs["tags"])
        output = torch.cat(
            [
                category,
                self.brand(inputs["brand"]),
                tags,
                preferred,
                self.device(inputs["device"]),
                self.surface(inputs["surface"]),
                self.time_of_day(inputs["time_of_day"]),
                preferred * category,
                request_category * category,
                inputs["price_cross"],
                inputs["retrieval_source_overlap"],
                inputs["preferred_category_match"],
                inputs["request_category_match"],
            ],
            dim=1,
        )
        if output.shape[1] != self.output_dim:
            raise ValueError("categorical encoder output dimension is invalid")
        return output
