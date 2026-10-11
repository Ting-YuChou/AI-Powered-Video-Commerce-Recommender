#!/usr/bin/env python3
"""Manual bounded-memory acceptance benchmark for the PIT spool contract."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_commerce.ml.pit_streaming import PitSpoolBuilder


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def run_case(rows: int, batch_rows: int) -> dict:
    import torch

    # Some virtualized CI CPUs do not expose a vendor string that oneDNN can
    # use to build a matmul primitive. The benchmark measures the PIT reader's
    # memory slope, so the portable eager CPU kernel is sufficient here.
    torch.backends.mkldnn.enabled = False

    root = tempfile.mkdtemp(prefix=f"pit-memory-{rows}-")
    started = time.monotonic()
    try:
        builder = PitSpoolBuilder(
            root,
            generation_id=f"synthetic-{rows}",
            kind="ranking",
            max_group_rows=8,
            max_bytes=8 * 1024**3,
            lineage={"benchmark": True},
        )
        chunk = []
        for index in range(rows):
            chunk.append(
                {
                    "group_id": f"g-{index // 4}",
                    "row_id": f"r-{index}",
                    "as_of_ts": float(index // 4),
                    "business_value": float(index % 17) if index % 11 == 0 else None,
                    "is_purchase": index % 11 == 0,
                    "payload": {"x": float(index % 101), "y": float(index % 2)},
                }
            )
            if len(chunk) == 2048:
                builder.add_rows(chunk)
                chunk.clear()
        if chunk:
            builder.add_rows(chunk)
        source = builder.finalize(expected_rows=rows)
        weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        optimizer = torch.optim.SGD([weight, bias], lr=0.001)
        seen = 0
        for batch in source.iter_group_batches(
            split="all", batch_rows=batch_rows, seed=42, epoch=0
        ):
            values = torch.tensor([[float(row["x"])] for row in batch])
            labels = torch.tensor([[float(row["y"])] for row in batch])
            prediction = values * weight + bias
            loss = torch.nn.functional.mse_loss(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            seen += len(batch)
        return {
            "rows": rows,
            "rows_seen": seen,
            "peak_rss_bytes": _peak_rss_bytes(),
            "staging_bytes": source.staging_bytes,
            "duration_seconds": time.monotonic() - started,
            "residual_staging_generations": len(
                list(source.path.parent.parent.glob(".staging-*"))
            ),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[100_000, 500_000])
    parser.add_argument("--batch-rows", type=int, default=1024)
    parser.add_argument("--case", type=int)
    args = parser.parse_args()
    if args.case is not None:
        print(json.dumps(run_case(args.case, args.batch_rows), sort_keys=True))
        return 0
    results = []
    for rows in args.rows:
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--case",
                str(rows),
                "--batch-rows",
                str(args.batch_rows),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stderr)
            return completed.returncode
        results.append(json.loads(completed.stdout))
    baseline, largest = results[0], results[-1]
    growth = largest["peak_rss_bytes"] / max(baseline["peak_rss_bytes"], 1) - 1.0
    report = {"results": results, "peak_rss_growth_ratio": growth}
    print(json.dumps(report, indent=2, sort_keys=True))
    if any(result["peak_rss_bytes"] >= 2 * 1024**3 for result in results):
        return 1
    if growth > 0.25:
        return 1
    if any(result["residual_staging_generations"] for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
