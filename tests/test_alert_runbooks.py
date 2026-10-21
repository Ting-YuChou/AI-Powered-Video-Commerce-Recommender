"""Every alert routes somewhere and links to a runbook section that exists."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "monitoring" / "prometheus-rules.yml"
RUNBOOKS = ROOT / "docs" / "operations" / "alert-runbooks.md"
RUNBOOK_BASE = (
    "https://github.com/Ting-YuChou/AI-Powered-Video-Commerce-Recommender/"
    "blob/main/docs/operations/alert-runbooks.md#"
)


def _alerts():
    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    return [
        rule
        for group in rules["groups"]
        for rule in group["rules"]
        if "alert" in rule
    ]


def _github_anchor(heading: str) -> str:
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\- ]", "", slug)
    return slug.replace(" ", "-")


def _runbook_sections():
    sections = {}
    current = None
    for line in RUNBOOKS.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(body) for name, body in sections.items()}


def test_every_alert_has_severity_and_a_resolvable_runbook_url():
    anchors = {_github_anchor(name) for name in _runbook_sections()}
    alerts = _alerts()

    assert alerts
    for alert in alerts:
        name = alert["alert"]
        assert alert["labels"]["severity"] in {"critical", "warning"}, name
        runbook_url = alert["annotations"].get("runbook_url", "")
        assert runbook_url.startswith(RUNBOOK_BASE), name
        anchor = runbook_url[len(RUNBOOK_BASE):]
        assert anchor == _github_anchor(name), name
        assert anchor in anchors, f"{name}: no '## {name}' section in alert-runbooks.md"


def test_runbook_sections_are_complete_and_not_stale():
    alert_names = {alert["alert"] for alert in _alerts()}
    sections = _runbook_sections()

    assert set(sections) == alert_names
    for name, body in sections.items():
        for field in ("**Symptom:**", "**Impact:**", "**First checks:**", "**Remediation:**"):
            assert field in body, f"{name} runbook is missing {field}"


def test_chart_rules_copy_matches_monitoring_rules():
    chart_copy = ROOT / "charts" / "video-commerce" / "files" / "prometheus-rules.yml"

    assert chart_copy.read_bytes() == RULES.read_bytes(), (
        "run: cp monitoring/prometheus-rules.yml charts/video-commerce/files/"
    )


def test_prometheus_rule_template_is_optional_and_reads_the_chart_copy():
    template = (
        ROOT / "charts/video-commerce/templates/prometheusrule.yaml"
    ).read_text(encoding="utf-8")
    values = yaml.safe_load(
        (ROOT / "charts/video-commerce/values.yaml").read_text(encoding="utf-8")
    )

    assert values["monitoring"]["prometheusRule"]["enabled"] is False
    assert "kind: PrometheusRule" in template
    assert '.Files.Get "files/prometheus-rules.yml"' in template


def test_new_and_changed_alerts_have_promtool_cases():
    cases = (ROOT / "tests/monitoring/prometheus-rules.test.yml").read_text(
        encoding="utf-8"
    )
    for name in (
        "WorkerHeartbeatMissing",
        "WorkerMetricsTargetDown",
        "KafkaProducerDisconnected",
        "RateLimiterFailingOpen",
        "DependencyUnhealthy",
        "RankingRuntimeUnauthenticatedConnections",
        "RankingRuntimeAuthFailures",
        "RecommendationOutboxTerminal",
        "RecommendationImpressionAttributionDegraded",
        "CatalogOutboxTerminal",
        "ContentProcessingRedrivesExhausted",
        "ContentZeroNormEmbeddings",
        "PitRunWaitingForMaterializer",
        "KnownUserFeaturesMissing",
    ):
        assert f"alertname: {name}" in cases, name
