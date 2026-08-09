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
