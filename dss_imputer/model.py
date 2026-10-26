from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = time.float().unsqueeze(1) * 1_000 * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.projection(embedding)


class MaskConditionedDenoiser(nn.Module):
    """Column-token denoiser conditioned on time and a per-cell target mask."""

    def __init__(
        self,
        numerical_features: int,
        categorical_sizes: list[int],
        d_model: int = 192,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.05,
        ff_multiplier: int = 4,
        architecture: str = "transformer",
        mlp_width: int = 512,
    ) -> None:
        super().__init__()
        if numerical_features + len(categorical_sizes) < 1:
            raise ValueError("The denoiser needs at least one feature.")
        if d_model < 1:
            raise ValueError("d_model must be positive.")
        if layers < 1:
            raise ValueError("layers must be positive.")
        if heads < 1:
            raise ValueError("heads must be positive.")
        if architecture not in {"transformer", "hybrid", "mlp"}:
            raise ValueError(
                "architecture must be 'transformer', 'hybrid', or 'mlp'."
            )
        if architecture != "mlp" and d_model % heads:
            raise ValueError("d_model must be divisible by heads.")
        if mlp_width < 1:
            raise ValueError("mlp_width must be positive.")

        self.numerical_features = numerical_features
        self.categorical_sizes = list(categorical_sizes)
        self.feature_count = numerical_features + len(categorical_sizes)
        self.d_model = d_model
        self.architecture = architecture
        self.mlp_width = mlp_width
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.ff_multiplier = ff_multiplier

        if numerical_features:
            self.numerical_weight = nn.Parameter(
                torch.randn(numerical_features, d_model) / math.sqrt(d_model)
            )
            self.numerical_bias = nn.Parameter(torch.zeros(numerical_features, d_model))
            self.numerical_output_weight = nn.Parameter(
                torch.randn(numerical_features, d_model) / math.sqrt(d_model)
            )
            self.numerical_output_bias = nn.Parameter(torch.zeros(numerical_features))
        else:
            self.register_parameter("numerical_weight", None)
            self.register_parameter("numerical_bias", None)
            self.register_parameter("numerical_output_weight", None)
            self.register_parameter("numerical_output_bias", None)

        offsets: list[int] = []
        offset = 0
        for size in categorical_sizes:
            offsets.append(offset)
            offset += size + 1
        self.register_buffer(
            "categorical_offsets", torch.tensor(offsets, dtype=torch.long)
        )
        self.categorical_embedding = (
            nn.Embedding(offset, d_model) if categorical_sizes else None
        )

        # self.column_embedding = nn.Embedding(self.feature_count, d_model)
        # self.type_embedding = nn.Embedding(2, d_model)
        self.status_embedding = nn.Embedding(2, d_model)
        if architecture == "transformer":
            # Keep the original module names so existing checkpoints still load.
            self.time_embedding = TimeEmbedding(d_model)
            self.transformer = self._make_transformer(
                d_model, layers, heads, dropout, ff_multiplier
            )
        elif architecture == "hybrid":
            self.pre_transformer = self._make_transformer(
                d_model, layers, heads, dropout, ff_multiplier
            )
            flat_dimension = self.feature_count * d_model
            self.mlp_input = nn.Linear(flat_dimension, mlp_width)
            self.time_embedding = TimeEmbedding(mlp_width)
            self.mlp = nn.Sequential(
                nn.Linear(mlp_width, mlp_width * 2),
                nn.SiLU(),
                nn.Linear(mlp_width * 2, mlp_width * 2),
                nn.SiLU(),
                nn.Linear(mlp_width * 2, mlp_width),
                nn.SiLU(),
                nn.Linear(mlp_width, flat_dimension),
            )
            self.post_transformer = self._make_transformer(
                d_model, layers, heads, dropout, ff_multiplier
            )
        else:
            flat_dimension = self.feature_count * d_model
            self.mlp_input = nn.Linear(flat_dimension, mlp_width)
            self.time_embedding = TimeEmbedding(mlp_width)
            blocks: list[nn.Module] = []
            for _ in range(layers):
                blocks.extend(
                    [
                        nn.Linear(mlp_width, mlp_width),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                    ]
                )
            self.mlp = nn.Sequential(*blocks)
            self.mlp_output = nn.Linear(mlp_width, flat_dimension)
        self.categorical_heads = nn.ModuleList(
            [nn.Linear(d_model, size) for size in categorical_sizes]
        )

    @staticmethod
    def _make_transformer(
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        ff_multiplier: int,
    ) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(d_model)
        )

    def config(self) -> dict[str, Any]:
        config = {
            "numerical_features": self.numerical_features,
            "categorical_sizes": self.categorical_sizes,
            "d_model": self.d_model,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "ff_multiplier": self.ff_multiplier,
        }
        if self.architecture != "transformer":
            config.update(
                {
                    "architecture": self.architecture,
                    "mlp_width": self.mlp_width,
                }
            )
        return config

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        time: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch = target_mask.shape[0]
        tokens: list[torch.Tensor] = []
        types: list[int] = []

        if self.numerical_features:
            numerical_tokens = numerical.unsqueeze(
                -1
            ) * self.numerical_weight.unsqueeze(0) + self.numerical_bias.unsqueeze(0)
            tokens.append(numerical_tokens)
            types.extend([0] * self.numerical_features)

        if self.categorical_sizes:
            global_ids = categorical + self.categorical_offsets.unsqueeze(0)
            tokens.append(self.categorical_embedding(global_ids))
            types.extend([1] * len(self.categorical_sizes))

        hidden = torch.cat(tokens, dim=1)
        columns = torch.arange(self.feature_count, device=hidden.device)
        type_ids = torch.tensor(types, dtype=torch.long, device=hidden.device)
        # hidden = hidden + self.column_embedding(columns).unsqueeze(0)
        # hidden = hidden + self.type_embedding(type_ids).unsqueeze(0)
        hidden = hidden + self.status_embedding(target_mask.long())
        if self.architecture == "transformer":
            hidden = hidden + self.time_embedding(time).view(batch, 1, self.d_model)
            hidden = self.transformer(hidden)
        elif self.architecture == "hybrid":
            hidden = self.pre_transformer(hidden)
            flat = hidden.reshape(batch, -1)
            flat = self.mlp_input(flat) + self.time_embedding(time)
            hidden = self.mlp(flat).reshape(batch, self.feature_count, self.d_model)
            hidden = self.post_transformer(hidden)
        else:
            flat = hidden.reshape(batch, -1)
            flat = self.mlp_input(flat) + self.time_embedding(time)
            flat = self.mlp(flat)
            hidden = self.mlp_output(flat).reshape(
                batch, self.feature_count, self.d_model
            )

        if self.numerical_features:
            numerical_hidden = hidden[:, : self.numerical_features]
            numerical_prediction = (
                numerical_hidden * self.numerical_output_weight.unsqueeze(0)
            ).sum(dim=-1) + self.numerical_output_bias.unsqueeze(0)
        else:
            numerical_prediction = numerical.new_zeros((batch, 0))

        categorical_hidden = hidden[:, self.numerical_features :]
        categorical_predictions = [
            head(categorical_hidden[:, index])
            for index, head in enumerate(self.categorical_heads)
        ]
        return numerical_prediction, categorical_predictions
