"""Rollback-only adapter from current-state legacy rows to typed examples."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from video_commerce.common.feature_history_contracts import (
    RANKING_LTR_FEATURE_DEFINITION_VERSION,
)
from video_commerce.ml.ranking_features import FeatureBundle
from video_commerce.ml.ranking_history import (
    RANKING_HISTORY_CONTEXT_KEY,
    build_training_history_contexts,
    merge_item_embedding_maps,
    ranking_history_config_from_settings,
)
from video_commerce.ml.ranking_training import AttributionFacts, RankingTrainingExample


class LegacyTrainingDatasetAdapter:
    """The only component allowed to read current Redis/catalog training state."""

    def __init__(
        self,
        *,
        feature_store,
        vector_search,
        ranking_model,
        recommendation_engine,
    ) -> None:
        self.feature_store = feature_store
        self.vector_search = vector_search
        self.ranking_model = ranking_model
        self.recommendation_engine = recommendation_engine

    async def build(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        training_sample_source: str,
    ) -> list[RankingTrainingExample]:
        user_features_map = await self.feature_store.get_all_user_features_map()
        product_metadata_map = dict(
            getattr(self.vector_search, "product_metadata", None) or {}
        )
        cf_engine = getattr(self.recommendation_engine, "cf_engine", None)
        item_embedding_map = merge_item_embedding_maps(
            getattr(cf_engine, "trained_item_embeddings", None),
            getattr(cf_engine, "synthetic_item_embeddings", None),
        )
        two_tower_model_version = getattr(
            self.recommendation_engine, "loaded_two_tower_version", None
        )
        return self.build_from_maps(
            rows,
            user_features_map=user_features_map,
            product_metadata_map=product_metadata_map,
            item_embedding_map=item_embedding_map,
            two_tower_model_version=two_tower_model_version,
            training_sample_source=training_sample_source,
        )

    def build_from_maps(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        user_features_map: Mapping[str, Any],
        product_metadata_map: Mapping[str, Mapping[str, Any]],
        item_embedding_map: Mapping[str, Any],
        two_tower_model_version: str | None,
        training_sample_source: str,
    ) -> list[RankingTrainingExample]:
        history_contexts = {}
        ranking_config = getattr(self.ranking_model, "config", None)
        if getattr(ranking_config, "history_embeddings_enabled", False):
            history_contexts = build_training_history_contexts(
                list(rows),
                item_embedding_map,
                config=ranking_history_config_from_settings(ranking_config),
                two_tower_model_version=two_tower_model_version,
            )

        examples = []
        for index, original in enumerate(rows):
            sample = dict(original)
            product_id = str(sample.get("product_id") or "").strip()
            user_id = str(sample.get("user_id") or "unknown").strip()
            if not product_id:
                continue
            as_of_ts = self.ranking_model._training_as_of_timestamp(sample)
            user_features = self.ranking_model._training_user_features(
                sample, user_features_map
            )
            product_metadata = self.ranking_model._training_product_metadata(
                sample, product_metadata_map
            )
            context = dict(sample.get("context") or {})
            if index in history_contexts:
                context[RANKING_HISTORY_CONTEXT_KEY] = history_contexts[index]
            candidate = self.ranking_model._training_candidate(sample)
            action, clicked, purchased = self._attribution_state(sample)
            value = None
            value_source = None
            if purchased:
                value = self.ranking_model._training_business_value_label(
                    sample, product_metadata, action
                )
                value_source = "legacy_heuristic"
            impression_id = str(
                sample.get("impression_id") or context.get("impression_id") or ""
            ).strip()
            is_slate = bool(
                impression_id
                and training_sample_source
                in {"recommendation_impressions", "feature_lake_pit"}
            )
            if not impression_id:
                impression_id = f"legacy:{user_id}:{int(as_of_ts // 1800)}"
            observation_id = str(
                sample.get("observation_id")
                or sample.get("event_id")
                or f"legacy:{index}:{user_id}:{product_id}"
            )
            examples.append(
                RankingTrainingExample(
                    observation_id=observation_id,
                    impression_id=impression_id,
                    bundle=FeatureBundle(
                        as_of_ts=as_of_ts,
                        feature_definition_version=(
                            RANKING_LTR_FEATURE_DEFINITION_VERSION
                        ),
                        user_features=user_features,
                        product_metadata=product_metadata,
                        context=context,
                        candidate=candidate,
                    ),
                    attribution=AttributionFacts(
                        attributed_action=action,
                        attributed_click=clicked,
                        attributed_purchase=purchased,
                        attributed_value=value,
                        attributed_value_source=value_source,
                    ),
                    is_slate_sample=is_slate,
                )
            )
        return examples

    @staticmethod
    def _attribution_state(sample: Mapping[str, Any]) -> tuple[str, bool, bool]:
        context = sample.get("context") or {}
        action = str(sample.get("action") or "view").lower()
        purchased = action == "purchase" or bool(context.get("attributed_purchase"))
        clicked = (
            purchased
            or action in {"click", "add_to_cart"}
            or bool(context.get("attributed_click"))
        )
        if purchased:
            return "purchase", True, True
        if action == "add_to_cart":
            return "add_to_cart", True, False
        if clicked:
            return "click", True, False
        return "view", False, False
