#!/usr/bin/env python3
"""CPU-only smoke check for ranking torch.compile.

This verifies local correctness, status fields, and fallback accounting. CPU
timings are regression signals only; they are not evidence of GPU kernel-fusion
performance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_commerce.common.config import RankingConfig
from video_commerce.ml.ranking import RankingModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only ranking torch.compile correctness smoke"
    )
    parser.add_argument("--backend", default="inductor", help="torch.compile backend")
    parser.add_argument("--mode", default="default", help="torch.compile mode")
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass dynamic=True/False to torch.compile",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def _ranking_config(args: argparse.Namespace, *, compile_enabled: bool) -> RankingConfig:
    return RankingConfig(
        architecture="dcn",
        hidden_dims=[16, 8],
        batch_target_requests=args.batch_size,
        torch_compile_enabled=compile_enabled,
        torch_compile_backend=args.backend,
        torch_compile_mode=args.mode,
        torch_compile_dynamic=args.dynamic,
    )


async def _load_ranking_model(config: RankingConfig, *, seed: int) -> RankingModel:
    torch.manual_seed(seed)
    model = RankingModel(config)
    await model.load_model()
    return model


def _time_inference(
    model: RankingModel,
    feature_matrix: np.ndarray,
    iterations: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    durations_ms: List[float] = []
    profile: Dict[str, Any] = {}
    iteration_count = max(1, iterations)
    for _ in range(iteration_count):
        started_at = time.perf_counter()
        _, profile = model.run_inference_batch(feature_matrix)
        durations_ms.append((time.perf_counter() - started_at) * 1000)
    return {
        "iterations": iteration_count,
        "average_ms": round(statistics.fmean(durations_ms), 3),
        "min_ms": round(min(durations_ms), 3),
        "max_ms": round(max(durations_ms), 3),
        "last_model_forward_ms": float(profile.get("model_forward_ms", 0.0)),
        "last_tensor_prep_ms": float(profile.get("tensor_prep_ms", 0.0)),
    }, profile


def _parity_report(
    eager_predictions: Dict[str, np.ndarray],
    compiled_predictions: Dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    parity_passed = True
    for name in sorted(eager_predictions):
        eager = eager_predictions[name]
        compiled = compiled_predictions[name]
        max_abs_diff = float(np.max(np.abs(eager - compiled)))
        close = bool(np.allclose(eager, compiled, rtol=rtol, atol=atol))
        outputs[name] = {
            "allclose": close,
            "max_abs_diff": max_abs_diff,
        }
        parity_passed = parity_passed and close
    return {
        "passed": parity_passed,
        "rtol": rtol,
        "atol": atol,
        "outputs": outputs,
    }


async def run(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    if not hasattr(torch, "compile"):
        return 2, {
            "status": "skipped",
            "reason": "torch.compile unavailable",
            "disclaimer": "CPU smoke only; no GPU performance claim.",
        }

    rng = np.random.default_rng(args.seed)
    feature_dim = RankingModel(
        _ranking_config(args, compile_enabled=False)
    ).feature_extractor.total_feature_dim
    feature_matrix = rng.normal(
        size=(args.batch_size, feature_dim),
    ).astype(np.float32)

    eager = await _load_ranking_model(
        _ranking_config(args, compile_enabled=False),
        seed=args.seed,
    )
    compiled = await _load_ranking_model(
        _ranking_config(args, compile_enabled=True),
        seed=args.seed,
    )

    eager_predictions, eager_profile = eager.run_inference_batch(feature_matrix)
    compiled_predictions, compiled_profile = compiled.run_inference_batch(
        feature_matrix
    )
    parity = _parity_report(
        eager_predictions,
        compiled_predictions,
        rtol=args.rtol,
        atol=args.atol,
    )

    eager_timing, eager_last_profile = _time_inference(
        eager,
        feature_matrix,
        args.iterations,
    )
    compiled_timing, compiled_last_profile = _time_inference(
        compiled,
        feature_matrix,
        args.iterations,
    )
    compiled_health = compiled.health_check()
    fallback_free = (
        compiled_health["torch_compile_active"] is True
        and compiled_health["torch_compile_error"] is None
        and compiled_health["torch_compile_fallback_count"] == 0
    )

    result = {
        "status": "passed" if parity["passed"] and fallback_free else "failed",
        "disclaimer": "CPU smoke/regression only; no GPU performance claim.",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(compiled.device),
        "compile": {
            "backend": args.backend,
            "mode": args.mode,
            "dynamic": args.dynamic,
            "enabled": True,
        },
        "batch_size": args.batch_size,
        "feature_dim": feature_matrix.shape[1],
        "parity": parity,
        "profiles": {
            "eager_initial": eager_profile,
            "compiled_initial": compiled_profile,
            "eager_last": eager_last_profile,
            "compiled_last": compiled_last_profile,
        },
        "timing": {
            "eager": eager_timing,
            "compiled": compiled_timing,
        },
        "compiled_health": {
            "torch_compile_active": compiled_health["torch_compile_active"],
            "torch_compile_error": compiled_health["torch_compile_error"],
            "torch_compile_warmup_ms": compiled_health["torch_compile_warmup_ms"],
            "torch_compile_fallback_count": compiled_health[
                "torch_compile_fallback_count"
            ],
            "torch_compile_last_fallback_error": compiled_health[
                "torch_compile_last_fallback_error"
            ],
            "torch_compile_last_inference_path": compiled_health[
                "torch_compile_last_inference_path"
            ],
        },
    }
    return (0 if result["status"] == "passed" else 1), result


def main() -> int:
    args = parse_args()
    exit_code, result = asyncio.run(run(args))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
