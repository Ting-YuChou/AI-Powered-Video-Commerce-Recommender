"""Release-bound performance evidence required by Phase C promotion."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping


def quality_gate_snapshot_valid(
    gate_config: Mapping, *, experiment_kind: str = "combined"
) -> bool:
    """A v2 promotion cannot silently weaken the versioned policy."""
    from dataclasses import asdict
    from video_commerce.ml.model_release_quality import RankingQualityGateConfig

    expected = asdict(RankingQualityGateConfig())
    expected["experiment_kind"] = experiment_kind
    dynamic_fields = {
        "candidate_recall_at_500_champion",
        "candidate_recall_at_500_challenger",
    }
    return all(
        gate_config.get(key) == value
        for key, value in expected.items()
        if key not in dynamic_fields
    )


def bundle_digest(manifest: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def active_rrf_release_valid(checkpoint) -> bool:
    payload = dict((checkpoint or {}).get("payload") or {})
    return (
        bool(payload.get("model_release_id"))
        and int(payload.get("active_generation") or 0) > 0
        and payload.get("quality_gate_policy_version") == "ranking_quality_gate_v2"
        and payload.get("candidate_fusion_policy_version") == "candidate_rrf_v1"
        and payload.get("candidate_rrf_k") == 60
        and payload.get("candidate_source_registry_version") == "retrieval_sources_v1"
    )


def performance_evidence_valid(evidence, *, release_id: str, manifest: Mapping) -> bool:
    """Fail closed on missing identity, unmeasured resources or failing SLOs."""
    if not isinstance(evidence, dict):
        return False
    if evidence.get("release_id") != release_id or evidence.get(
        "bundle_digest"
    ) != bundle_digest(manifest):
        return False
    if evidence.get("platform") != "linux" or not evidence.get("resources"):
        return False
    try:
        resources = evidence["resources"]
        if not all(
            math.isfinite(float(resources[key])) and float(resources[key]) > 0.0
            for key in ("cpus", "memory_bytes")
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    checksum = str(evidence.get("results_sha256") or "")
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        return False
    for mode in ("hot", "unique"):
        result = dict(evidence.get(mode) or {})
        required = (
            "error_rate",
            "five_xx_rate",
            "successful_p95_ms",
            "successful_p99_ms",
            "p95_relative_regression",
            "successful_qps_relative_regression",
        )
        try:
            values = {key: float(result[key]) for key in required}
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in values.values()):
            return False
        if (
            not 0.0 <= values["error_rate"] < 0.005
            or not 0.0 <= values["five_xx_rate"] < 0.001
            or not 0.0 <= values["successful_p95_ms"] < 1000.0
            or not 0.0 <= values["successful_p99_ms"] < 2000.0
            or values["p95_relative_regression"] > 0.1
            or values["successful_qps_relative_regression"] > 0.1
        ):
            return False
    return True
