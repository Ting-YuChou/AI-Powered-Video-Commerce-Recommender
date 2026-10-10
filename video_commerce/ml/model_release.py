"""Release lifecycle policy and audited promotion orchestration."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ReleaseLifecycleState(str, Enum):
    REGISTERED = "registered"
    VALIDATED = "validated"
    STAGING = "staging"
    ACTIVE = "active"
    RETIRED = "retired"


class ReleaseValidationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReleasePromotionError(RuntimeError):
    """Raised when a release transition violates the durable policy."""


def validate_release_transition(
    current: ReleaseLifecycleState | str,
    target: ReleaseLifecycleState | str,
    *,
    validation_status: ReleaseValidationStatus | str,
    bootstrap: bool = False,
) -> bool:
    current_state = ReleaseLifecycleState(current)
    target_state = ReleaseLifecycleState(target)
    status = ReleaseValidationStatus(validation_status)
    if bootstrap:
        if (
            current_state is not ReleaseLifecycleState.REGISTERED
            or target_state is not ReleaseLifecycleState.ACTIVE
        ):
            raise ReleasePromotionError("bootstrap only permits registered to active")
        return True
    allowed = {
        ReleaseLifecycleState.REGISTERED: {ReleaseLifecycleState.VALIDATED},
        ReleaseLifecycleState.VALIDATED: {ReleaseLifecycleState.STAGING},
        ReleaseLifecycleState.STAGING: {ReleaseLifecycleState.ACTIVE},
        ReleaseLifecycleState.ACTIVE: {ReleaseLifecycleState.RETIRED},
        ReleaseLifecycleState.RETIRED: {ReleaseLifecycleState.ACTIVE},
    }
    if target_state not in allowed[current_state]:
        raise ReleasePromotionError(
            f"release transition {current_state.value} to {target_state.value} is not allowed"
        )
    if (
        target_state
        in {
            ReleaseLifecycleState.VALIDATED,
            ReleaseLifecycleState.STAGING,
            ReleaseLifecycleState.ACTIVE,
        }
        and status is not ReleaseValidationStatus.PASSED
    ):
        raise ReleasePromotionError("release quality gate has not passed")
    return True


class ModelReleaseController:
    def __init__(self, store: Any) -> None:
        self.store = store

    async def promote(
        self,
        *,
        model_name: str,
        model_version: str,
        environment: str,
        expected_generation: int,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        normalized_actor = str(actor or "").strip()
        normalized_reason = str(reason or "").strip()
        if not normalized_actor:
            raise ValueError("promotion actor is required")
        if not normalized_reason:
            raise ValueError("promotion reason is required")
        result = await self.store.activate_model_release(
            model_name=str(model_name),
            model_version=str(model_version),
            environment=str(environment),
            expected_generation=int(expected_generation),
            actor=normalized_actor,
            reason=normalized_reason,
            rollback=False,
            bootstrap=False,
        )
        if result is None:
            raise ReleasePromotionError(
                "release activation failed because state or generation changed"
            )
        return dict(result)

    async def rollback(
        self,
        *,
        model_name: str,
        model_version: str,
        environment: str,
        expected_generation: int,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        normalized_actor = str(actor or "").strip()
        normalized_reason = str(reason or "").strip()
        if not normalized_actor:
            raise ValueError("rollback actor is required")
        if not normalized_reason:
            raise ValueError("rollback reason is required")
        result = await self.store.activate_model_release(
            model_name=str(model_name),
            model_version=str(model_version),
            environment=str(environment),
            expected_generation=int(expected_generation),
            actor=normalized_actor,
            reason=normalized_reason,
            rollback=True,
            bootstrap=False,
        )
        if result is None:
            raise ReleasePromotionError(
                "release rollback failed because state or generation changed"
            )
        return dict(result)

    async def bootstrap_active(
        self,
        *,
        model_name: str,
        model_version: str,
        environment: str,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        normalized_actor = str(actor or "").strip()
        normalized_reason = str(reason or "").strip()
        if not normalized_actor or not normalized_reason:
            raise ValueError("bootstrap actor and reason are required")
        result = await self.store.activate_model_release(
            model_name=str(model_name),
            model_version=str(model_version),
            environment=str(environment),
            expected_generation=0,
            actor=normalized_actor,
            reason=normalized_reason,
            rollback=False,
            bootstrap=True,
        )
        if result is None:
            raise ReleasePromotionError("active release already exists")
        return dict(result)
