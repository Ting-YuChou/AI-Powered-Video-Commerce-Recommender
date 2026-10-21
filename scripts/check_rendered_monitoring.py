"""Check a `helm template` render with ServiceMonitor and PrometheusRule enabled.

Usage: python scripts/check_rendered_monitoring.py /tmp/video-commerce-monitoring.yaml

- The PrometheusRule spec equals monitoring/prometheus-rules.yml.
- Every ServiceMonitor endpoint relabels `job`, so rule matchers such as
  job="gateway-api" work on Kubernetes as they do in Compose.
- ranking-service also scrapes the proxy's own /metrics/proxy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(rendered_path: str) -> int:
    documents = [
        doc
        for doc in yaml.safe_load_all(Path(rendered_path).read_text(encoding="utf-8"))
        if doc
    ]
    errors = []

    rules = [doc for doc in documents if doc.get("kind") == "PrometheusRule"]
    expected = yaml.safe_load((ROOT / "monitoring/prometheus-rules.yml").read_text())
    if len(rules) != 1:
        errors.append(f"expected one PrometheusRule, found {len(rules)}")
    elif rules[0]["spec"] != expected:
        errors.append("PrometheusRule spec differs from monitoring/prometheus-rules.yml")

    monitors = [doc for doc in documents if doc.get("kind") == "ServiceMonitor"]
    if not monitors:
        errors.append("no ServiceMonitor rendered")
    jobs = {}
    for monitor in monitors:
        for endpoint in monitor["spec"]["endpoints"]:
            relabel = [
                item
                for item in endpoint.get("relabelings", [])
                if item.get("targetLabel") == "job"
            ]
            if len(relabel) != 1:
                errors.append(
                    f"{monitor['metadata']['name']} {endpoint.get('path')}: no job relabel"
                )
                continue
            jobs[relabel[0]["replacement"]] = endpoint.get("path")
    if jobs.get("ranking-service-proxy") != "/metrics/proxy":
        errors.append("ranking-service-proxy endpoint missing")
    for job in ("gateway-api", "recommendation-service", "ranking-service"):
        if jobs.get(job) != "/metrics":
            errors.append(f"{job} endpoint missing")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if not errors:
        print(f"ok: 1 PrometheusRule, {len(monitors)} ServiceMonitors, jobs={sorted(jobs)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
