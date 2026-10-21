"""Deployment contract for the edge: Caddy probes, body limits, scratch bounds."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "charts" / "video-commerce"


def _chart_caddyfile() -> str:
    template = (CHART / "templates" / "caddy-configmap.yaml").read_text()
    body = template.split("Caddyfile: |", 1)[1].split("{{- end }}", 1)[0]
    lines = [line[4:] if line.startswith("    ") else line for line in body.splitlines()]
    return "\n".join(lines).strip()


def _route_block(caddyfile: str) -> str:
    return caddyfile.split("route {", 1)[1]


def test_caddy_readiness_probes_caddy_itself():
    deployment = (CHART / "templates" / "caddy-deployment.yaml").read_text()
    readiness = deployment.split("readinessProbe:", 1)[1].split("resources:", 1)[0]

    assert "path: /caddy-healthz" in readiness
    assert "respond /caddy-healthz 200" in _chart_caddyfile()


def test_compose_and_chart_caddyfiles_share_the_same_route_rules():
    compose_caddyfile = (ROOT / "Code.Frontend" / "Caddyfile").read_text()
    chart_route = _route_block(_chart_caddyfile()).replace(
        '{{ include "video-commerce.componentName" (dict "root" . "name" "gateway-api") }}',
        "gateway-api",
    )

    assert _route_block(compose_caddyfile).split() == chart_route.split()
    assert "max_size {$EDGE_MAX_UPLOAD_BODY:501MiB}" in compose_caddyfile
    assert "max_size {$EDGE_MAX_JSON_BODY:1MiB}" in compose_caddyfile


def test_backend_scratch_volumes_are_bounded():
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    empty_dirs = values["backend"]["emptyDirs"]

    for name in ("tmp", "uploads", "objectStorage", "logs", "pitStaging"):
        assert empty_dirs[name].get("sizeLimit"), name
    gateway = values["backend"]["workloads"]["gatewayApi"]["resources"]
    assert gateway["limits"]["ephemeral-storage"]


def test_helm_topic_list_includes_the_flink_late_events_topic():
    values = yaml.safe_load((CHART / "values.yaml").read_text())

    assert "late-interaction-events" in values["kafkaTopicInitJob"]["topics"]
