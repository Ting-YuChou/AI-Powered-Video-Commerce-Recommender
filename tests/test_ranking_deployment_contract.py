import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "charts/video-commerce"


def _helm_template(*args):
    helm = os.getenv("HELM_BIN") or shutil.which("helm")
    if not helm:
        pytest.skip("helm binary not available")
    rendered = subprocess.run(
        [helm, "template", "video-commerce", str(CHART), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [doc for doc in yaml.safe_load_all(rendered) if doc]


def _network_policies(docs):
    return {
        doc["metadata"]["name"]: doc for doc in docs if doc["kind"] == "NetworkPolicy"
    }


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


def test_ranking_runtime_auth_mode_is_wired_with_permissive_default():
    env_example = (ROOT / ".env.example").read_text()
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    configmap = (CHART / "templates/backend-configmap.yaml").read_text()

    assert "RANKING_RUNTIME_AUTH_MODE=permissive" in env_example
    for service in ("ranking-coordinator", "ranking-runner", "ranking-service"):
        assert (
            compose["services"][service]["environment"]["RANKING_RUNTIME_AUTH_MODE"]
            == "${RANKING_RUNTIME_AUTH_MODE:-permissive}"
        )
    assert values["appConfig"]["ranking"]["runtimeAuthMode"] == "permissive"
    assert "RANKING_RUNTIME_AUTH_MODE:" in configmap


def test_ranking_runner_prestop_drains_over_loopback():
    runner = yaml.safe_load((CHART / "values.yaml").read_text())["backend"][
        "workloads"
    ]["rankingRunner"]
    command = runner["lifecycle"]["preStop"]["exec"]["command"]

    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--operation") + 1] == "drain"


def test_ranking_network_policies_are_toggleable_and_scoped():
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    template = (CHART / "templates/ranking-network-policies.yaml").read_text()

    assert values["networkPolicy"]["rankingRuntime"]["enabled"] is True
    assert values["networkPolicy"]["rankingRuntime"]["metricsFrom"] == {
        "namespaceSelector": {},
        "podSelector": {},
    }
    assert "kind: NetworkPolicy" in template
    assert ".Values.networkPolicy.rankingRuntime" in template
    assert "$workloads.recommendationService.name" in template
    assert "$workloads.rankingService.name" in template


def test_ranking_network_policies_render_default_ingress_rules():
    policies = _network_policies(_helm_template())

    coordinator = policies["video-commerce-ranking-coordinator"]["spec"]
    runner = policies["video-commerce-ranking-runner"]["spec"]

    assert coordinator["podSelector"]["matchLabels"][
        "app.kubernetes.io/component"
    ] == "ranking-coordinator"
    assert coordinator["policyTypes"] == ["Ingress"]
    assert len(coordinator["ingress"]) == 1
    assert sorted(
        peer["podSelector"]["matchLabels"]["app.kubernetes.io/component"]
        for peer in coordinator["ingress"][0]["from"]
    ) == ["ranking-service", "recommendation-service"]
    assert coordinator["ingress"][0]["ports"] == [{"protocol": "TCP", "port": 8013}]

    assert runner["podSelector"]["matchLabels"][
        "app.kubernetes.io/component"
    ] == "ranking-runner"
    assert len(runner["ingress"]) == 1
    assert [
        peer["podSelector"]["matchLabels"]["app.kubernetes.io/component"]
        for peer in runner["ingress"][0]["from"]
    ] == ["ranking-coordinator"]
    assert runner["ingress"][0]["ports"] == [{"protocol": "TCP", "port": 8014}]


def test_ranking_network_policies_render_metrics_peer_and_toggle_off():
    with_metrics = _network_policies(
        _helm_template(
            "--set",
            "networkPolicy.rankingRuntime.metricsFrom.namespaceSelector.matchLabels.team=monitoring",
            "--set",
            "networkPolicy.rankingRuntime.metricsFrom.podSelector.matchLabels.app=prometheus",
        )
    )
    for name, port in (
        ("video-commerce-ranking-coordinator", 8013),
        ("video-commerce-ranking-runner", 8014),
    ):
        metrics_rule = with_metrics[name]["spec"]["ingress"][1]
        # Both selectors in one peer: pods matching the label in matching namespaces.
        assert metrics_rule["from"] == [
            {
                "namespaceSelector": {"matchLabels": {"team": "monitoring"}},
                "podSelector": {"matchLabels": {"app": "prometheus"}},
            }
        ]
        assert metrics_rule["ports"] == [{"protocol": "TCP", "port": port}]

    disabled = _network_policies(
        _helm_template("--set", "networkPolicy.rankingRuntime.enabled=false")
    )
    assert "video-commerce-ranking-coordinator" not in disabled
    assert "video-commerce-ranking-runner" not in disabled
