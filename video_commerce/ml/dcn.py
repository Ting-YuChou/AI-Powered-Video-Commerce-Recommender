"""
Reusable Deep & Cross Network building blocks.

The implementation uses DCN v1 vector cross layers to keep parameter growth
small enough for latency-sensitive retrieval and ranking paths.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set

import torch
import torch.nn as nn

SUPPORTED_ARCHITECTURES = {"mlp", "dcn"}
RANKING_ARCHITECTURES = SUPPORTED_ARCHITECTURES | {"dcn_v2_low_rank"}


def normalize_architecture(
    value: Optional[str],
    default: str = "dcn",
    *,
    supported: Optional[Set[str]] = None,
) -> str:
    """Normalize and validate model architecture names."""
    architecture = (value or default).strip().lower()
    supported_architectures = supported or SUPPORTED_ARCHITECTURES
    if architecture not in supported_architectures:
        raise ValueError(
            f"Unsupported architecture '{value}'. "
            f"Expected one of: {', '.join(sorted(supported_architectures))}"
        )
    return architecture


def build_deep_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    *,
    dropout: float,
    use_batch_norm: bool = True,
) -> tuple[nn.Module, int]:
    """Build the existing Linear/BatchNorm/ReLU/Dropout deep branch."""
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        prev_dim = hidden_dim

    if not layers:
        return nn.Identity(), input_dim
    return nn.Sequential(*layers), prev_dim


class VectorCrossLayer(nn.Module):
    """One DCN v1 vector cross layer: x_{l+1} = x0 * (xl dot w) + b + xl."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        cross_scalar = torch.sum(xl * self.weight, dim=-1, keepdim=True)
        return x0 * cross_scalar + self.bias + xl


class VectorCrossNetwork(nn.Module):
    """Stack of vector cross layers."""

    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            VectorCrossLayer(input_dim) for _ in range(max(0, int(num_layers)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class LowRankCrossLayer(nn.Module):
    """One DCN-v2 low-rank cross layer: x0 * (V(U(xl)) + b) + xl."""

    def __init__(self, input_dim: int, low_rank_dim: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.low_rank_dim = max(1, int(low_rank_dim))
        self.u = nn.Linear(input_dim, self.low_rank_dim, bias=False)
        self.v = nn.Linear(self.low_rank_dim, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.u.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        cross_vector = self.v(self.u(xl)) + self.bias
        return x0 * cross_vector + xl


class LowRankCrossNetwork(nn.Module):
    """Stack of DCN-v2 low-rank matrix cross layers."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        low_rank_dim: int = 8,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            LowRankCrossLayer(input_dim, low_rank_dim)
            for _ in range(max(0, int(num_layers)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class DeepAndCrossNetwork(nn.Module):
    """LayerNorm input, parallel cross/deep branches, concat, projection."""

    def __init__(
        self,
        input_dim: int,
        deep_hidden_dims: Iterable[int],
        output_dim: int,
        *,
        cross_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cross_layers = max(0, int(cross_layers))

        self.input_norm = nn.LayerNorm(input_dim)
        self.cross_network = VectorCrossNetwork(input_dim, self.cross_layers)
        self.deep_network, deep_output_dim = build_deep_mlp(
            input_dim,
            deep_hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.output_projection = nn.Linear(input_dim + deep_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(x)
        cross_features = self.cross_network(normalized)
        deep_features = self.deep_network(normalized)
        return self.output_projection(torch.cat([cross_features, deep_features], dim=-1))


class LowRankDeepAndCrossNetwork(nn.Module):
    """LayerNorm input with DCN-v2 low-rank cross and parallel deep branch."""

    def __init__(
        self,
        input_dim: int,
        deep_hidden_dims: Iterable[int],
        output_dim: int,
        *,
        cross_layers: int = 3,
        low_rank_dim: int = 8,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cross_layers = max(0, int(cross_layers))
        self.low_rank_dim = max(1, int(low_rank_dim))

        self.input_norm = nn.LayerNorm(input_dim)
        self.cross_network = LowRankCrossNetwork(
            input_dim,
            self.cross_layers,
            self.low_rank_dim,
        )
        self.deep_network, deep_output_dim = build_deep_mlp(
            input_dim,
            deep_hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.output_projection = nn.Linear(input_dim + deep_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(x)
        cross_features = self.cross_network(normalized)
        deep_features = self.deep_network(normalized)
        return self.output_projection(torch.cat([cross_features, deep_features], dim=-1))
