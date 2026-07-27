# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Application service for configurable model-based image upscaling."""

from __future__ import annotations

from typing import Any, cast

import torch

from ..runtime.model_upscaling import ModelUpscaler, UpscaleModel


class ModelUpscaleService:
    """Bridge Comfy model descriptors into the typed model upscaling runtime."""

    def __init__(self, upscaler: ModelUpscaler | None = None) -> None:
        """Create the service with an injectable runtime."""

        self._upscaler = upscaler or ModelUpscaler()

    def upscale(
        self,
        upscale_model: Any,
        image: torch.Tensor,
        maximum_batch_size: int = 0,
        tile_size: int = 0,
        channels_last: bool = False,
        precision: str = "fp32",
    ) -> tuple[torch.Tensor]:
        """Upscale an image batch while retaining the workflow tuple contract."""

        model = cast(UpscaleModel, upscale_model)
        return (
            self._upscaler.upscale(
                model,
                image,
                maximum_batch_size=maximum_batch_size,
                tile_size=tile_size,
                channels_last=channels_last,
                precision=precision,
            ),
        )


__all__ = ["ModelUpscaleService"]
