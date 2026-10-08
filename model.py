from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


RAW_FEATURE_NAMES = [
    "Age",
    "Gestational Age",
    "Number of sons ",
    "Number of daughters",
    "Total Number of Children",
    "Gravida",
    "Female Education",
    "Husband Education",
    "Working Status",
    "Physical Health ",
    "Previous Miscarriage",
    "Sufficient Money for Basic Needs",
    "Current Appereance Acceptance",
    "Family System",
    "Male Gender Preference",
    "Relationship with Mother in-law",
]

NUMERIC_FEATURE_NAMES = RAW_FEATURE_NAMES[:5]
CATEGORICAL_FEATURE_NAMES = RAW_FEATURE_NAMES[5:]


class FTTransformer(nn.Module):
    """Feature-tokenized transformer for dense preprocessed tabular inputs."""

    def __init__(
        self,
        model_config: Mapping[str, int | float],
    ) -> None:
        super().__init__()
        self.input_dim = int(model_config["input_dim"])
        self.embed_dim = int(model_config["embed_dim"])
        num_heads = int(model_config["num_heads"])
        depth = int(model_config["depth"])
        dropout = float(model_config.get("dropout", 0.1))

        if self.input_dim < 1 or self.embed_dim < 1 or num_heads < 1 or depth < 1:
            raise ValueError(
                "input_dim, embed_dim, num_heads, and depth must be positive."
            )
        if self.embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.feature_weight = nn.Parameter(
            torch.empty(self.input_dim, self.embed_dim)
        )
        self.feature_bias = nn.Parameter(torch.empty(self.input_dim, self.embed_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.feature_bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch, {self.input_dim}); "
                f"received {tuple(inputs.shape)}."
            )

        tokens = (
            inputs.unsqueeze(-1) * self.feature_weight.unsqueeze(0)
            + self.feature_bias.unsqueeze(0)
        )
        cls_tokens = self.cls_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        encoded = self.transformer(self.input_dropout(tokens))
        return self.classifier(self.norm(encoded[:, 0]))
