"""Reproducible, interleaved legacy-versus-Triton ranking load runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable, List


@dataclass(frozen=True)
class RankingLoadRun:
    backend: str
    base_url: str
    rate: int
    duration: str
    repetition: int
    output_path: Path

    def command(self, *, k6_binary: str, script_path: Path) -> List[str]:
        return [
            k6_binary,
            "run",
            "--summary-export",
            str(self.output_path),
            "-e",
            f"BASE_URL={self.base_url}",
            "-e",
            f"RATE={self.rate}",
            "-e",
            f"DURATION={self.duration}",
            "-e",
            "MODE=acceptance",
            str(script_path),
        ]


def build_run_plan(
    *,
    legacy_url: str,
    triton_url: str,
    rates: Iterable[int],
    duration: str,
    repetitions: int,
    output_dir: Path,
) -> List[RankingLoadRun]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    targets = {"legacy": legacy_url, "triton": triton_url}
    plan: List[RankingLoadRun] = []
    for rate in rates:
        if int(rate) < 1:
            raise ValueError("rates must be positive")
        for repetition in range(1, repetitions + 1):
            order = ("legacy", "triton") if repetition % 2 else ("triton", "legacy")
            for backend in order:
                plan.append(
                    RankingLoadRun(
                        backend=backend,
                        base_url=targets[backend],
                        rate=int(rate),
                        duration=duration,
                        repetition=repetition,
                        output_path=output_dir
                        / f"{backend}-{int(rate)}qps-run{repetition}.json",
                    )
                )
    return plan


def execute_run_plan(
    plan: Iterable[RankingLoadRun],
    *,
    k6_binary: str,
    script_path: Path,
    dry_run: bool,
) -> None:
    for run in plan:
        command = run.command(k6_binary=k6_binary, script_path=script_path)
        print(" ".join(command))
        if dry_run:
            continue
        run.output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
