from types import SimpleNamespace

import pytest

from video_commerce.services.model_trainer import main as trainer_module
from video_commerce.services.model_trainer.main import ModelTrainerService


def _service(tmp_path, manager):
    service = object.__new__(ModelTrainerService)
    service.config = SimpleNamespace(
        ranking_config=SimpleNamespace(
            feature_contract_version="ranking_ltr_v1",
            copy=lambda deep=True: SimpleNamespace(
                feature_contract_version="ranking_ltr_v1",
                probability_calibration="none",
                din_enabled=False,
                trimodal_enabled=False,
            ),
        ),
        model_config=SimpleNamespace(ranking_model_path=str(tmp_path / "ranking.pt")),
        model_release_config=SimpleNamespace(gate_mode="enforced", environment="production"),
        feature_lake_config=SimpleNamespace(
            attribution_window_hours=168, allowed_lateness_hours=1
        ),
    )
    service.ranking_model = SimpleNamespace(feature_schema_version="schema-v1")
    service.artifact_manager = manager
    service.observability = None
    return service


class _Manager:
    def __init__(self, record):
        self.record = record
        self.calls = 0

    async def sync_selected_ranking_checkpoint(self, **_kwargs):
        self.calls += 1
        return self.record


def _active_record(path, release_id="release-active"):
    return SimpleNamespace(
        model_version="ranking-active",
        checkpoint_path=str(path),
        payload={
            "active_generation": 3,
            "model_release_id": release_id,
            "feature_definition_version": "ranking_ltr_v1",
            "probability_calibration": {"temperature": 1.2},
        },
    )


@pytest.mark.asyncio
async def test_champion_is_reloaded_from_the_active_release_each_round(
    tmp_path, monkeypatch
):
    active_path = tmp_path / "generations" / "ranking_model" / "3-release" / "ranking.pt"
    active_path.parent.mkdir(parents=True)
    active_path.write_bytes(b"checkpoint")
    manager = _Manager(_active_record(active_path))
    service = _service(tmp_path, manager)
    loaded = []

    class FakeRankingModel:
        def __init__(self, config, observability=None):
            self.config = config

        async def load_model(self, path):
            loaded.append((path, self.config.probability_calibration))

    monkeypatch.setattr(trainer_module, "RankingModel", FakeRankingModel)

    record = await service._refresh_active_ranking_checkpoint()
    champion = await service._load_champion_model(record)

    assert manager.calls == 1
    assert service.loaded_ranking_checkpoint is record
    assert loaded == [(str(active_path), "temperature_v1")]
    assert champion.config.feature_contract_version == "ranking_ltr_v1"
    # The trainer's own output path is never the champion source.
    assert service._challenger_checkpoint_path() == str(tmp_path / "ranking.challenger.pt")
    assert service._challenger_checkpoint_path() != str(active_path)


@pytest.mark.asyncio
async def test_latest_fallback_without_active_generation_is_not_a_champion(tmp_path):
    record = _active_record(tmp_path / "ranking.pt")
    record.payload.pop("active_generation")
    service = _service(tmp_path, _Manager(record))

    assert await service._refresh_active_ranking_checkpoint() is None


@pytest.mark.asyncio
async def test_missing_active_checkpoint_file_refuses_to_build_a_champion(tmp_path):
    service = _service(tmp_path, _Manager(None))

    with pytest.raises(RuntimeError, match="not available locally"):
        await service._load_champion_model(_active_record(tmp_path / "missing.pt"))


def test_label_purge_covers_attribution_window_and_lateness(tmp_path):
    service = _service(tmp_path, _Manager(None))

    assert service._label_purge_seconds() == (168 + 1) * 3600.0
