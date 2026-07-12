from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_compose_pit_production_preset_is_pit_first():
    preset = {}
    for raw_line in (
        (ROOT / "deploy/compose/pit-production.conf").read_text().splitlines()
    ):
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            preset[key] = value

    assert preset["FEATURE_LAKE_ENABLED"] == "true"
    assert preset["FEATURE_LAKE_TRAINING_SOURCE"] == "pit"
    assert preset["FEATURE_LAKE_PIT_SHADOW_ENABLED"] == "false"
    assert preset["FEATURE_LAKE_RANKING_PIT_DATASET_URI"] == (
        "s3://video-commerce-features/training/ranking-pit/latest.json"
    )
    assert preset["FEATURE_LAKE_PIT_ORCHESTRATOR_ENABLED"] == "true"


def test_compose_defines_resumable_pit_orchestrator_service():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = compose["services"]["pit-dataset-orchestrator"]

    assert "pit-production" in service["profiles"]
    assert service["build"]["target"] == "pit-orchestrator"
    assert service["environment"]["FEATURE_LAKE_PIT_EXPORT_URI"].endswith(
        "/training/ranking-pit"
    )
    assert service["environment"]["FEATURE_LAKE_FLINK_JOBMANAGER"] == (
        "flink-jobmanager:8081"
    )
    assert "feature-lake-readiness-gate" in service["depends_on"]


def test_compose_flink_cluster_uses_feature_image_with_system_hadoop_libraries():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = compose["services"]

    for name in ("flink-jobmanager", "flink-taskmanager"):
        service = services[name]
        assert service["image"] == "${FLINK_FEATURE_IMAGE:-video-commerce-flink-feature:local}"
        assert service["build"]["context"] == "./flink-jobs/interaction-features"
        assert service["build"]["dockerfile"] == "Dockerfile"


def test_helm_pit_production_preset_and_cronjob_contract():
    values = yaml.safe_load(
        (ROOT / "charts/video-commerce/values-pit-production.yaml").read_text()
    )
    lake = values["external"]["featureLake"]
    cron = values["pitOrchestrator"]

    assert lake["enabled"] is True
    assert lake["trainingSource"] == "pit"
    assert lake["pitShadowEnabled"] is False
    assert lake["rankingPitDatasetUri"].endswith("/latest.json")
    assert cron["enabled"] is True
    assert cron["schedule"] == "0 2 * * *"
    assert cron["concurrencyPolicy"] == "Forbid"
    assert values["backend"]["workloads"]["pitStateExporter"]["enabled"] is True

    template = (
        ROOT / "charts/video-commerce/templates/pit-orchestrator-cronjob.yaml"
    ).read_text()
    assert 'required "external.featureLake.catalogUri' in template
    assert 'required "external.featureLake.flinkJobmanager' in template
    assert 'required "external.featureLake.warehouseUri' in template
    assert "concurrencyPolicy:" in template
    assert "--once" in template


def test_pit_run_migration_and_orm_share_resume_lineage_columns():
    migration = (
        ROOT / "migrations/postgres/006_pit_materialization_runs.sql"
    ).read_text()
    orm = (ROOT / "video_commerce/data_plane/system_store.py").read_text()

    for column in (
        "run_id",
        "cutoff_at",
        "status",
        "phase",
        "attempts",
        "export_attempt",
        "flink_job_id",
        "lease_expires_at",
        "snapshot_id",
        "manifest_uri",
    ):
        assert column in migration
    assert "class PitMaterializationRun(Base):" in orm
    assert "export_attempt: Mapped[Optional[int]]" in orm
    assert "flink_job_id: Mapped[Optional[str]]" in orm
