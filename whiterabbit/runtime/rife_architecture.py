# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>
# Portions derived from ComfyUI-Frame-Interpolation (MIT).
# Copyright (c) 2023-2025 Fannovel16 and contributors.
# See LICENSES/MIT-ComfyUI-Frame-Interpolation.txt.

"""WhiteRabbit RIFE inference architectures with scale and internal ensemble."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as functional
from torch import nn

from ..domain.rife import scale_pyramid


class RifeInferenceModel(Protocol):
    """Common runtime interface for supported RIFE generations."""

    pad_align: int

    def __call__(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        timestep: float | torch.Tensor,
        scale_factor: float,
        ensemble: bool,
    ) -> torch.Tensor:
        """Synthesize one or more intermediate frames."""


class EnhancedCoreRife(nn.Module):
    """Run Comfy's current five-block IFNet with WhiteRabbit controls."""

    pad_align = 64

    def __init__(self, network: nn.Module) -> None:
        """Wrap an official Comfy IFNet instance without copying its architecture."""

        super().__init__()
        self.network = network

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        timestep: float | torch.Tensor = 0.5,
        scale_factor: float = 1.0,
        ensemble: bool = False,
    ) -> torch.Tensor:
        """Interpolate with a configurable five-level pyramid and block ensemble."""

        network: Any = self.network
        timestep_tensor = _timestep_tensor(timestep, image_0)
        network._build_warp_grids(image_0.shape[2], image_0.shape[3], image_0.device)
        feature_0 = network.encode(image_0)
        feature_1 = network.encode(image_1)
        flow: torch.Tensor | None = None
        mask: torch.Tensor | None = None
        refinement: torch.Tensor | None = None
        warped_0, warped_1 = image_0, image_1
        scales = scale_pyramid(scale_factor, 5)

        for index, block in enumerate(network.blocks):
            if flow is None:
                flow, mask, refinement = block(
                    torch.cat(
                        (image_0, image_1, feature_0, feature_1, timestep_tensor),
                        1,
                    ),
                    None,
                    scale=scales[index],
                )
                if ensemble:
                    reverse_flow, reverse_mask, reverse_refinement = block(
                        torch.cat(
                            (
                                image_1,
                                image_0,
                                feature_1,
                                feature_0,
                                1 - timestep_tensor,
                            ),
                            1,
                        ),
                        None,
                        scale=scales[index],
                    )
                    flow = (flow + _swap_flow(reverse_flow)) / 2
                    mask = (mask - reverse_mask) / 2
                    refinement = (refinement + reverse_refinement) / 2
            else:
                if mask is None or refinement is None:
                    raise RuntimeError("RIFE refinement state is incomplete.")
                warped_feature_0 = network.warp(feature_0, flow[:, :2])
                warped_feature_1 = network.warp(feature_1, flow[:, 2:4])
                delta, next_mask, next_refinement = block(
                    torch.cat(
                        (
                            warped_0,
                            warped_1,
                            warped_feature_0,
                            warped_feature_1,
                            timestep_tensor,
                            mask,
                            refinement,
                        ),
                        1,
                    ),
                    flow,
                    scale=scales[index],
                )
                if ensemble:
                    reverse_delta, reverse_mask, reverse_refinement = block(
                        torch.cat(
                            (
                                warped_1,
                                warped_0,
                                warped_feature_1,
                                warped_feature_0,
                                1 - timestep_tensor,
                                -mask,
                                refinement,
                            ),
                            1,
                        ),
                        _swap_flow(flow),
                        scale=scales[index],
                    )
                    delta = (delta + _swap_flow(reverse_delta)) / 2
                    next_mask = (next_mask - reverse_mask) / 2
                    next_refinement = (next_refinement + reverse_refinement) / 2
                flow = flow + delta
                mask = next_mask
                refinement = next_refinement
            warped_0 = network.warp(image_0, flow[:, :2])
            warped_1 = network.warp(image_1, flow[:, 2:4])
        if mask is None:
            raise RuntimeError("RIFE produced no blend mask.")
        return torch.lerp(warped_1, warped_0, torch.sigmoid(mask))


class _LegacyResidualConv(nn.Module):
    """Residual convolution used by RIFE 4.7/4.9 checkpoints."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply learned residual scaling and activation."""

        return cast(torch.Tensor, self.relu(inputs + self.conv(inputs) * self.beta))


def _legacy_conv(
    input_channels: int, output_channels: int, stride: int
) -> nn.Sequential:
    """Build the named convolution sequence expected by legacy checkpoints."""

    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride, 1),
        nn.LeakyReLU(0.2, True),
    )


class _LegacyIFBlock(nn.Module):
    """One four-stage RIFE 4.7/4.9 flow-refinement block."""

    def __init__(self, input_channels: int, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            _legacy_conv(input_channels, channels // 2, 2),
            _legacy_conv(channels // 2, channels, 2),
        )
        self.convblock = nn.Sequential(
            *(_LegacyResidualConv(channels) for _ in range(8))
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(channels, 24, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        flow: torch.Tensor | None,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a flow delta and blend mask at one pyramid level."""

        inputs = functional.interpolate(
            inputs,
            scale_factor=1 / scale,
            mode="bilinear",
            align_corners=False,
        )
        if flow is not None:
            resized_flow = (
                functional.interpolate(
                    flow,
                    scale_factor=1 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                / scale
            )
            inputs = torch.cat((inputs, resized_flow), 1)
        features = self.convblock(self.conv0(inputs))
        output = functional.interpolate(
            self.lastconv(features),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )
        return output[:, :4] * scale, output[:, 4:5]


class LegacyRife47(nn.Module):
    """Native inference-only RIFE 4.7 architecture used by 4.7/4.9 weights."""

    pad_align = 64

    def __init__(self) -> None:
        """Create layers with checkpoint-compatible names."""

        super().__init__()
        self.block0 = _LegacyIFBlock(15, 192)
        self.block1 = _LegacyIFBlock(20, 128)
        self.block2 = _LegacyIFBlock(20, 96)
        self.block3 = _LegacyIFBlock(20, 64)
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ConvTranspose2d(16, 4, 4, 2, 1),
        )
        self._warp_grids: dict[
            tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        timestep: float | torch.Tensor = 0.5,
        scale_factor: float = 1.0,
        ensemble: bool = False,
    ) -> torch.Tensor:
        """Interpolate with the legacy four-level pyramid and internal ensemble."""

        original_height, original_width = image_0.shape[2:]
        alignment = required_legacy_alignment(scale_factor)
        image_0 = _pad_to_alignment(image_0, alignment)
        image_1 = _pad_to_alignment(image_1, alignment)
        timing = _timestep_tensor(timestep, image_0)
        feature_0 = self.encode(image_0[:, :3])
        feature_1 = self.encode(image_1[:, :3])
        flow: torch.Tensor | None = None
        mask: torch.Tensor | None = None
        warped_0, warped_1 = image_0, image_1
        blocks = [self.block0, self.block1, self.block2, self.block3]
        scales = scale_pyramid(scale_factor, 4)
        for index, block in enumerate(blocks):
            if flow is None:
                flow, mask = block(
                    torch.cat((image_0, image_1, feature_0, feature_1, timing), 1),
                    None,
                    scales[index],
                )
                if ensemble:
                    reverse_flow, reverse_mask = block(
                        torch.cat(
                            (image_1, image_0, feature_1, feature_0, 1 - timing), 1
                        ),
                        None,
                        scales[index],
                    )
                    flow = (flow + _swap_flow(reverse_flow)) / 2
                    mask = (mask - reverse_mask) / 2
            else:
                if mask is None:
                    raise RuntimeError("RIFE refinement mask is missing.")
                warped_feature_0 = self._warp(feature_0, flow[:, :2])
                warped_feature_1 = self._warp(feature_1, flow[:, 2:4])
                delta, next_mask = block(
                    torch.cat(
                        (
                            warped_0,
                            warped_1,
                            warped_feature_0,
                            warped_feature_1,
                            timing,
                            mask,
                        ),
                        1,
                    ),
                    flow,
                    scales[index],
                )
                if ensemble:
                    reverse_delta, reverse_mask = block(
                        torch.cat(
                            (
                                warped_1,
                                warped_0,
                                warped_feature_1,
                                warped_feature_0,
                                1 - timing,
                                -mask,
                            ),
                            1,
                        ),
                        _swap_flow(flow),
                        scales[index],
                    )
                    delta = (delta + _swap_flow(reverse_delta)) / 2
                    next_mask = (next_mask - reverse_mask) / 2
                flow = flow + delta
                mask = next_mask
            warped_0 = self._warp(image_0, flow[:, :2])
            warped_1 = self._warp(image_1, flow[:, 2:4])
        if mask is None:
            raise RuntimeError("RIFE produced no blend mask.")
        output = warped_0 * torch.sigmoid(mask) + warped_1 * (1 - torch.sigmoid(mask))
        return output[:, :, :original_height, :original_width]

    def _warp(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Backward-warp an image with a cached normalized coordinate grid."""

        height, width = flow.shape[2:]
        key = (height, width, str(flow.device))
        cached = self._warp_grids.get(key)
        if cached is None:
            horizontal = torch.linspace(-1, 1, width, device=flow.device).view(
                1, 1, 1, width
            )
            vertical = torch.linspace(-1, 1, height, device=flow.device).view(
                1, 1, height, 1
            )
            base = torch.cat(
                (
                    horizontal.expand(1, -1, height, -1),
                    vertical.expand(1, -1, -1, width),
                ),
                1,
            )
            divisors = torch.tensor(
                [(width - 1) / 2, (height - 1) / 2], device=flow.device
            ).view(1, 2, 1, 1)
            cached = (base, divisors)
            self._warp_grids = {key: cached}
        base, divisors = cached
        grid = (base.expand(flow.shape[0], -1, -1, -1) + flow / divisors).permute(
            0, 2, 3, 1
        )
        return functional.grid_sample(
            image,
            grid.to(image.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


def create_core_rife(state_dict: dict[str, torch.Tensor]) -> EnhancedCoreRife:
    """Load current RIFE weights with Comfy's official detector and IFNet class."""

    module = import_module("comfy_extras.frame_interpolation_models.ifnet")
    detector: Any = module.detect_rife_config
    head_channels, channels = detector(state_dict)
    network = cast(nn.Module, module.IFNet(head_ch=head_channels, channels=channels))
    network.load_state_dict(state_dict)
    return EnhancedCoreRife(network)


def required_core_alignment(scale_factor: float) -> int:
    """Return padding alignment needed by current IFNet at a quality scale."""

    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero.")
    return round(EnhancedCoreRife.pad_align / min(1.0, scale_factor))


def required_legacy_alignment(scale_factor: float) -> int:
    """Return padding alignment needed by four-block IFNet quality scales."""

    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero.")
    return max(LegacyRife47.pad_align, round(32 / scale_factor))


def remap_core_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize raw current-generation checkpoint prefixes for Comfy IFNet."""

    normalized = {
        key.removeprefix("module.").removeprefix("flownet."): value
        for key, value in state_dict.items()
        if not key.startswith(("teacher.", "caltime."))
    }
    remapped: dict[str, torch.Tensor] = {}
    for key, value in normalized.items():
        replacement = key
        for index in range(5):
            prefix = f"block{index}."
            if key.startswith(prefix):
                replacement = f"blocks.{index}.{key[len(prefix) :]}"
                break
        remapped[replacement] = value
    return remapped


def _timestep_tensor(
    timestep: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a scalar or batch timestep across reference spatial dimensions."""

    if isinstance(timestep, torch.Tensor):
        timing = timestep.to(device=reference.device, dtype=reference.dtype)
        if timing.ndim == 0:
            timing = timing.reshape(1, 1, 1, 1)
        elif timing.ndim == 1:
            timing = timing.reshape(-1, 1, 1, 1)
        return timing.expand(-1, 1, reference.shape[2], reference.shape[3])
    return torch.full(
        (reference.shape[0], 1, reference.shape[2], reference.shape[3]),
        timestep,
        device=reference.device,
        dtype=reference.dtype,
    )


def _swap_flow(flow: torch.Tensor) -> torch.Tensor:
    """Swap forward and backward flow channel pairs."""

    return torch.cat((flow[:, 2:4], flow[:, :2]), 1)


def _pad_to_alignment(images: torch.Tensor, alignment: int) -> torch.Tensor:
    """Reflect-pad NCHW images to the model's spatial alignment."""

    height, width = images.shape[2:]
    padded_height = ((height - 1) // alignment + 1) * alignment
    padded_width = ((width - 1) // alignment + 1) * alignment
    return functional.pad(
        images,
        (0, padded_width - width, 0, padded_height - height),
    )


__all__ = [
    "EnhancedCoreRife",
    "LegacyRife47",
    "RifeInferenceModel",
    "create_core_rife",
    "remap_core_state_dict",
    "required_core_alignment",
    "required_legacy_alignment",
]
