"""Offline ONNX export and parity validation for ranking models."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


OUTPUT_NAMES = ["ctr", "cvr", "ctcvr", "gmv", "ranking_score"]
VALIDATION_BATCH_SIZES = [1, 20, 64]


class _BaseOnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, base_features: torch.Tensor):
        predictions = self.model.forward_onnx(base_features)
        return tuple(predictions[name] for name in OUTPUT_NAMES)


class _DinOnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        base_features: torch.Tensor,
        candidate_indices: torch.Tensor,
        history_indices: torch.Tensor,
        history_recency: torch.Tensor,
        history_mask: torch.Tensor,
        summary_features: torch.Tensor,
    ):
        predictions = self.model.forward_onnx(
            base_features,
            candidate_indices=candidate_indices.squeeze(-1),
            history_indices=history_indices,
            history_recency=history_recency,
            history_mask=history_mask,
            summary_features=summary_features,
        )
        return tuple(predictions[name] for name in OUTPUT_NAMES)


def _sample_inputs(
    batch_size: int,
    input_dim: int,
    *,
    din_enabled: bool,
    history_mode: str = "full",
    item_count: int = 32,
) -> Tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(19 + batch_size)
    base = torch.randn(batch_size, input_dim, generator=generator)
    if not din_enabled:
        return (base,)
    candidate = torch.randint(
        1, max(2, item_count), (batch_size, 1), generator=generator
    )
    history = torch.randint(
        1, max(2, item_count), (batch_size, 3, 60), generator=generator
    )
    recency = torch.rand(batch_size, 3, 60, generator=generator)
    if history_mode == "empty":
        mask = torch.zeros(batch_size, 3, 60, dtype=torch.bool)
    elif history_mode == "sparse":
        mask = torch.zeros(batch_size, 3, 60, dtype=torch.bool)
        mask[..., -7:] = True
    else:
        mask = torch.ones(batch_size, 3, 60, dtype=torch.bool)
    history = torch.where(mask, history, torch.zeros_like(history))
    summary = torch.randn(batch_size, 12, generator=generator)
    return base, candidate, history, recency, mask, summary


def _torch_outputs(
    wrapper: nn.Module, inputs: Sequence[torch.Tensor]
) -> List[np.ndarray]:
    with torch.inference_mode():
        return [value.detach().cpu().numpy() for value in wrapper(*inputs)]


def export_ranking_onnx(
    ranking_model: Any,
    target_path: Path | str,
    *,
    opset: int = 17,
) -> Dict[str, Any]:
    """Export a trained verified ranker and validate numerical/top-k parity."""
    if not getattr(ranking_model, "is_trained", False):
        raise ValueError("ONNX export requires a trained ranking model")
    if not getattr(ranking_model, "artifact_verified", False):
        raise ValueError("ONNX export requires a verified ranking artifact")
    if bool(getattr(ranking_model.config, "trimodal_enabled", False)):
        raise ValueError("trimodal ranking is not supported by ONNX export")
    architecture = str(getattr(ranking_model.model, "architecture", ""))
    if architecture != "dcn":
        raise ValueError("ONNX export currently supports the DCN architecture only")
    if int(opset) != 17:
        raise ValueError("ranking ONNX export requires opset 17")
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnx and onnxruntime are required for export") from exc

    model = copy.deepcopy(ranking_model.model).to(torch.device("cpu")).eval()
    din_enabled = bool(getattr(ranking_model.config, "din_enabled", False))
    wrapper: nn.Module = (
        _DinOnnxWrapper(model) if din_enabled else _BaseOnnxWrapper(model)
    )
    wrapper.eval()
    input_dim = int(ranking_model.feature_extractor.total_feature_dim)
    item_count = (
        int(model.din.item_embedding.num_embeddings)
        if din_enabled and getattr(model, "din", None) is not None
        else 32
    )
    sample = _sample_inputs(
        20,
        input_dim,
        din_enabled=din_enabled,
        history_mode="full",
        item_count=item_count,
    )
    input_names = ["base_features"]
    if din_enabled:
        input_names.extend(
            [
                "candidate_indices",
                "history_indices",
                "history_recency",
                "history_mask",
                "summary_features",
            ]
        )
    dynamic_axes = {name: {0: "candidate_rows"} for name in input_names + OUTPUT_NAMES}
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.onnx.export(
            wrapper,
            sample,
            str(temporary),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
        )
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(
            str(temporary), providers=["CPUExecutionProvider"]
        )
        history_modes = ("empty", "sparse", "full") if din_enabled else ("full",)
        for batch_size in VALIDATION_BATCH_SIZES:
            for history_mode in history_modes:
                inputs = _sample_inputs(
                    batch_size,
                    input_dim,
                    din_enabled=din_enabled,
                    history_mode=history_mode,
                    item_count=item_count,
                )
                expected = _torch_outputs(wrapper, inputs)
                actual = session.run(
                    OUTPUT_NAMES,
                    {
                        name: value.detach().cpu().numpy()
                        for name, value in zip(input_names, inputs)
                    },
                )
                for name, torch_value, onnx_value in zip(
                    OUTPUT_NAMES, expected, actual
                ):
                    if not np.isfinite(onnx_value).all():
                        raise ValueError(f"ONNX {name} contains non-finite values")
                    np.testing.assert_allclose(
                        onnx_value,
                        torch_value,
                        atol=1e-5,
                        rtol=1e-4,
                        err_msg=f"ONNX parity failed for {name}",
                    )
                if not np.array_equal(
                    np.argsort(-expected[-1].reshape(-1)),
                    np.argsort(-actual[-1].reshape(-1)),
                ):
                    raise ValueError("ONNX ranking order parity failed")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "opset": 17,
        "input_dim": input_dim,
        "din_enabled": din_enabled,
        "din_sidecar_sha256": (
            str(
                (getattr(ranking_model, "din_sidecar_metadata", {}) or {}).get("sha256")
                or ""
            )
            if din_enabled
            else None
        ),
        "feature_schema_version": ranking_model.feature_schema_version,
        "model_architecture": architecture,
        "input_contract": {
            "base_features": [None, input_dim],
            **(
                {
                    "candidate_indices": [None, 1],
                    "history_indices": [None, 3, 60],
                    "history_recency": [None, 3, 60],
                    "history_mask": [None, 3, 60],
                    "summary_features": [None, 12],
                }
                if din_enabled
                else {}
            ),
        },
        "output_names": list(OUTPUT_NAMES),
        "output_contract": {name: [None, 1] for name in OUTPUT_NAMES},
        "validated_batch_sizes": list(VALIDATION_BATCH_SIZES),
        "parity_verified": True,
        "export_torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "target_triton_version": "26.07",
        "target_onnxruntime_version": "1.27.0",
    }
