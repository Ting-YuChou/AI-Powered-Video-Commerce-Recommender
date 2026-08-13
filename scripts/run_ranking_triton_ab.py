#!/usr/bin/env python3
"""Run an interleaved matched A/B against already deployed ranking backends."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_commerce.loadtest_ranking_ab import build_run_plan, execute_run_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-url", required=True)
    parser.add_argument("--triton-url", required=True)
    parser.add_argument("--rates", nargs="+", type=int, default=[500, 1000, 1500, 2000])
    parser.add_argument("--duration", default="5m")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("loadtest/results/ranking-triton-ab")
    )
    parser.add_argument("--k6-binary", default="k6")
    parser.add_argument("--script", type=Path, default=Path("loadtest/k6/ranking.js"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = build_run_plan(
        legacy_url=args.legacy_url,
        triton_url=args.triton_url,
        rates=args.rates,
        duration=args.duration,
        repetitions=args.repetitions,
        output_dir=args.output_dir,
    )
    execute_run_plan(
        plan,
        k6_binary=args.k6_binary,
        script_path=args.script,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
