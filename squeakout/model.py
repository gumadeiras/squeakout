from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        layers: list[nn.Module]
        if expand_ratio == 1:
            layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        else:
            layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    def __init__(self, width_mult: float = 1.0) -> None:
        super().__init__()
        input_channel = int(32 * width_mult)
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features: list[nn.Module] = [conv_bn(1, input_channel, 2)]
        for expand_ratio, channels, repeats, stride in inverted_residual_setting:
            output_channel = int(channels * width_mult)
            for repeat_index in range(repeats):
                block_stride = stride if repeat_index == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        block_stride,
                        expand_ratio=expand_ratio,
                    )
                )
                input_channel = output_channel
        features.append(conv_1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*features)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class SqueakOut(nn.Module):
    _ENCODER_BLOCKS = ((0, 2), (2, 4), (4, 7), (7, 14), (14, 19))

    def __init__(self) -> None:
        super().__init__()
        self.backbone = MobileNetV2Backbone()

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Conv2d(16, 3, 1)
        self.conv_score = nn.Conv2d(3, 1, 1)
        self._initialize_weights()

    def _run_feature_block(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for layer_index in range(start, end):
            x = self.backbone.features[layer_index](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []

        for start, end in self._ENCODER_BLOCKS[:-1]:
            x = self._run_feature_block(x, start, end)
            skip_connections.append(x)

        x = self._run_feature_block(x, *self._ENCODER_BLOCKS[-1])
        x1, x2, x3, x4 = skip_connections

        x = self.invres1(torch.cat([x4, self.dconv1(x)], dim=1))
        x = self.invres2(torch.cat([x3, self.dconv2(x)], dim=1))
        x = self.invres3(torch.cat([x2, self.dconv3(x)], dim=1))
        x = self.invres4(torch.cat([x1, self.dconv4(x)], dim=1))
        x = self.conv_last(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv_score(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


_UNUSED_CHECKPOINT_KEYS = {
    "backbone.classifier.1.bias",
    "backbone.classifier.1.weight",
}


def _normalize_checkpoint_key(key: str) -> str:
    normalized = key
    for prefix in ("state_dict.", "model.", "module."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


def extract_model_state_dict(checkpoint: Mapping[str, object]) -> OrderedDict[str, torch.Tensor]:
    raw_state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a valid state_dict mapping")

    normalized_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in raw_state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue

        normalized_key = _normalize_checkpoint_key(key)
        if normalized_key in _UNUSED_CHECKPOINT_KEYS:
            continue
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> SqueakOut:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint).__name__}")

    model = SqueakOut()
    state_dict = extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
