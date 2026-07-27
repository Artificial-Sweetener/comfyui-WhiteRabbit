# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tensor boundary validation shared by WhiteRabbit services."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImageBatchShape:
    """Validated NHWC image-batch dimensions."""

    batch_size: int
    height: int
    width: int
    channels: int


def validate_image_batch(
    images: torch.Tensor,
    *,
    name: str = "images",
    allow_empty: bool = False,
) -> ImageBatchShape:
    """Validate a ComfyUI NHWC IMAGE tensor and return its dimensions."""

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if images.ndim != 4:
        raise ValueError(
            f"{name} must have shape (frames, height, width, channels); "
            f"received rank {images.ndim}."
        )
    batch_size, height, width, channels = (int(value) for value in images.shape)
    if not allow_empty and batch_size == 0:
        raise ValueError(f"{name} must contain at least one frame.")
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must have positive spatial dimensions.")
    if channels not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels; received {channels}.")
    return ImageBatchShape(batch_size, height, width, channels)
