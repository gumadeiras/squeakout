from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch
from torch import nn

from .model import SqueakOut, extract_model_state_dict


class SqueakOutAutoencoder(nn.Module):
    """Legacy wrapper kept for downstream checkpoint/import compatibility."""

    def __init__(self) -> None:
        super().__init__()
        self.model = SqueakOut()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        normalized_state_dict = extract_model_state_dict({"state_dict": state_dict})
        legacy_state_dict = OrderedDict(
            (f"model.{key}", value) for key, value in normalized_state_dict.items()
        )
        return super().load_state_dict(legacy_state_dict, strict=strict, assign=assign)


SqueakOut_autoencoder = SqueakOutAutoencoder
