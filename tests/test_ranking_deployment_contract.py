from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_ranking_high_concurrency_defaults_are_production_safe():
    env_example = (ROOT / ".env.example").read_text()
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    values = yaml.safe_load((ROOT / "charts/video-commerce/values.yaml").read_text())

    assert "RANKING_ALLOW_UNTRAINED_FALLBACK=true" in env_example
    assert "RANKING_REQUIRE_VERIFIED_ARTIFACT=false" in env_example
    assert "RANKING_BATCH_RUNNER_COUNT=4" in env_example
    assert "SERVICE_RANKING_WORKERS=2" in env_example
    assert (
        compose["services"]["ranking-service"]["environment"]["SERVICE_RANKING_WORKERS"]
        == "${SERVICE_RANKING_WORKERS:-2}"
    )
    assert values["backend"]["workloads"]["rankingRunner"]["replicaCount"] == 4
    assert values["backend"]["workloads"]["rankingRunner"]["pdb"]["minAvailable"] == 3
    assert values["backend"]["workloads"]["rankingRunner"]["strategy"] == {
        "maxUnavailable": 1,
        "maxSurge": 1,
    }
    assert (
        values["backend"]["workloads"]["rankingRunner"]["probes"]["readinessType"]
        == "exec"
    )
    assert (
        values["backend"]["workloads"]["rankingCoordinator"]["probes"]["readinessType"]
        == "exec"
    )
    assert values["backend"]["workloads"]["rankingRunner"]["lifecycle"]["preStop"][
        "exec"
    ]["command"]


def test_ranking_loadtest_encodes_acceptance_slos_and_overload_mode():
    script = (ROOT / "loadtest/k6/ranking.js").read_text()

    assert '"p(95)<400"' in script
    assert '"p(99)<600"' in script
    assert "unexpected_errors" in script
    assert "five_xx" in script
    assert 'mode === "overload"' in script


def test_triton_deployment_is_opt_in_immutable_and_single_replica():
    env_example = (ROOT / ".env.example").read_text()
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    values = yaml.safe_load((ROOT / "charts/video-commerce/values.yaml").read_text())
    template = (
        ROOT / "charts/video-commerce/templates/ranking-triton.yaml"
    ).read_text()

    assert "RANKING_INFERENCE_BACKEND=legacy" in env_example
    assert "RANKING_ONNX_EXPORT_ENABLED=false" in env_example
    assert values["rankingTriton"]["enabled"] is False
    assert values["rankingTriton"]["image"]["tag"] == "26.07-py3"
    assert values["appConfig"]["ranking"]["inferenceBackend"] == "legacy"
    assert compose["services"]["ranking-triton"]["profiles"] == ["triton"]
    assert compose["services"]["ranking-triton-materializer"]["profiles"] == [
        "triton"
    ]
    volume_init = compose["services"]["ranking-triton-volume-init"]
    assert volume_init["profiles"] == ["triton"]
    assert volume_init["user"] == "0:0"
    assert "models_data:/app/models" in volume_init["volumes"]
    assert "triton_models:/models" in volume_init["volumes"]
    assert "/app/models /models" in " ".join(volume_init["command"])
    assert compose["services"]["ranking-triton-materializer"]["depends_on"][
        "ranking-triton-volume-init"
    ]["condition"] == "service_completed_successfully"
    assert "kafka-init" not in compose["services"]["ranking-triton-materializer"].get(
        "depends_on", {}
    )
    assert "ranking-coordinator" not in compose["services"]["ranking-service"].get(
        "depends_on", {}
    )
    assert "--model-control-mode=none" in template
    assert "--strict-readiness=true" in template
    assert "replicas: 1" in template
    assert "kind: ServiceMonitor" in template
    assert "kind: NetworkPolicy" in template
    assert "rankingCoordinatorDirectEnabled=false" in template
