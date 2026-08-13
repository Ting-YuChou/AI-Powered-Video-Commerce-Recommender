#!/usr/bin/env python3
"""Create JSON and Markdown tables from recommendation k6 summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_commerce.loadtest_support import summarize_k6_result


def parse_result(value: str) -> tuple[int, str, Path]:
    target, duration, path = value.split(":", 2)
    return int(target), duration, Path(path)


def format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="candidate_cache_rank")
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        summarize_k6_result(
            path,
            target_qps=target,
            duration=duration,
            scenario=args.scenario,
        )
        for target, duration, path in map(parse_result, args.result)
    ]
    checkpoint = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "git_commit": commit,
        "checkpoint": checkpoint,
        "scope": "gateway_to_recommendation_to_ranking_coordinator_and_runners",
        "quality_claim_valid": False,
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Recommendation Path Capacity Results",
        "",
        f"- Git commit: `{commit}`",
        f"- Checkpoint: `{checkpoint['model_version']}` (`{checkpoint['artifact_sha256']}`)",
        "- Scope: gateway → recommendation → coordinator → ranking runners",
        "- Model: real forward pass with deterministic synthetic weights; capacity only, not ranking quality",
        "",
        "| Target QPS | Offered QPS | Successful QPS | Errors | 5xx | Dropped | p50 | p95 | p99 | Batch avg | Forward p95 | Candidate-cache→rank | Model-forward evidence | Final-cache |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target_qps} | {actual_qps:.1f} | {successful_qps:.1f} | {error_rate:.2%} | "
            "{five_xx_rate:.2%} | {dropped_iterations} | {p50_ms:.1f} ms | "
            "{p95_ms:.1f} ms | {p99_display} | {batch_requests_avg:.1f} | "
            "{model_forward_p95_ms:.1f} ms | {candidate_cache_rank_rate:.2%} | "
            "{model_forward_evidence_rate:.2%} | {recommendation_cache_rate:.2%} |".format(
                p99_display=format_ms(row["p99_ms"]), **row
            )
        )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
