"""Prepare ``PROMETHEUS_MULTIPROC_DIR`` before uvicorn forks its workers.

Service commands run ``python -m video_commerce.common.prometheus_multiproc &&
exec python -m uvicorn ...`` so stale per-process files from a previous
container in the same tmpfs/emptyDir never leak into the new aggregate. It is a
no-op when the variable is unset.
"""

from __future__ import annotations

import sys

from video_commerce.common.observability import (
    prepare_prometheus_multiprocess_dir,
    prometheus_multiprocess_dir,
)


def main() -> int:
    directory = prometheus_multiprocess_dir()
    if not directory:
        return 0
    removed = prepare_prometheus_multiprocess_dir(directory)
    print(
        f"prometheus multiprocess dir {directory} ready ({removed} stale files removed)",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
