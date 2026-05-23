from types import SimpleNamespace

from video_commerce.services.feature_worker.feature_updater import should_run_python_feature_worker
from scripts import promote_flink_feature_shadow


class FakeRedis:
    def __init__(self):
        self.store = {
            b"flink:shadow:rtwf:user:u1:5m": b"payload",
            b"other:key": b"ignored",
        }
        self.ttls = {b"flink:shadow:rtwf:user:u1:5m": 60000}
        self.restored = {}

    def scan_iter(self, match=None, count=None):
        prefix = match.rstrip(b"*")
        return (key for key in list(self.store) if key.startswith(prefix))

    def pttl(self, key):
        return self.ttls.get(key, -1)

    def dump(self, key):
        return self.store.get(key)

    def restore(self, key, ttl, dumped, replace=False):
        self.restored[key] = {
            "ttl": ttl,
            "dumped": dumped,
            "replace": replace,
        }


def _config(mode):
    return SimpleNamespace(
        feature_pipeline_config=SimpleNamespace(mode=mode),
    )


def test_python_feature_worker_runs_in_shadow_mode():
    assert should_run_python_feature_worker(_config("python")) is True
    assert should_run_python_feature_worker(_config("flink_shadow")) is True


def test_python_feature_worker_skips_when_flink_is_official():
    assert should_run_python_feature_worker(_config("flink")) is False


def test_promote_flink_shadow_dry_run_does_not_write_official_keys():
    fake = FakeRedis()

    promoted = promote_flink_feature_shadow.promote(
        execute=False,
        batch_size=100,
        client=fake,
    )

    assert promoted == 1
    assert fake.restored == {}


def test_promote_flink_shadow_execute_copies_to_official_keys():
    fake = FakeRedis()

    promoted = promote_flink_feature_shadow.promote(
        execute=True,
        batch_size=100,
        client=fake,
    )

    assert promoted == 1
    assert fake.restored[b"rtwf:user:u1:5m"] == {
        "ttl": 60000,
        "dumped": b"payload",
        "replace": True,
    }
