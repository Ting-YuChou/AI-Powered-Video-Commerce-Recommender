from config import CacheConfig, RedisConfig
from feature_store import FeatureStore


def test_feature_store_routes_cache_keys_to_cache_redis_client():
    store = FeatureStore(
        RedisConfig(cache_host="redis-cache", cache_db=1),
        CacheConfig(),
    )
    state_client = object()
    cache_client = object()
    store.redis_client = state_client
    store.cache_redis_client = cache_client

    assert store._client_for_key_type("recommendations_cache") is cache_client
    assert store._client_for_key_type("candidate_cache") is cache_client
    assert store._client_for_key_type("product_metadata") is cache_client
    assert store._client_for_key_type("trending_products") is cache_client
    assert store._client_for_key_type("category_pool") is cache_client

    assert store._client_for_key_type("user_features") is state_client
    assert store._client_for_key_type("user_interactions") is state_client
    assert store._client_for_key_type("content_status") is state_client
    assert store._client_for_key_type("health") is state_client
