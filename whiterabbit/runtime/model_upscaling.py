# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Comfy-aware tiled execution for image upscale models."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from importlib import import_module
from typing import Any, Protocol, cast

import torch

from ..domain.image_geometry import chunk_spans
from ..shared.tensor_validation import validate_image_batch

MINIMUM_TILE_SIZE = 128


class UpscaleModel(Protocol):
    """Runtime surface supplied by Comfy upscale model descriptors."""

    model: torch.nn.Module
    scale: float

    def to(self, device: torch.device | str) -> Any:
        """Move the model to a torch device."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Upscale an NCHW image batch."""


class ModelUpscaler:
    """Upscale image batches with configurable chunking, tiling, and precision."""

    def upscale(
        self,
        model: UpscaleModel,
        images: torch.Tensor,
        *,
        maximum_batch_size: int,
        tile_size: int,
        channels_last: bool,
        precision: str,
    ) -> torch.Tensor:
        """Run a Comfy upscale model and return a clamped CPU NHWC batch."""

        shape = validate_image_batch(images, name="image")
        management = import_module("comfy.model_management")
        comfy_utils = import_module("comfy.utils")
        device = cast(torch.device, management.get_torch_device())
        scale = float(getattr(model, "scale", 4.0))

        model.to(device)
        try:
            self._ensure_parameter_device(model.model, device)
            model.model.eval()
            memory_required = int(management.module_size(model.model))
            memory_required += int(
                (512 * 512 * 3) * images.element_size() * max(scale, 1.0) * 384.0
            )
            memory_required += images.nelement() * images.element_size()
            management.free_memory(memory_required, device)

            chunks: list[torch.Tensor] = []
            for start, end in chunk_spans(shape.batch_size, maximum_batch_size):
                source = images[start:end].movedim(-1, -3).to(device, non_blocking=True)
                if channels_last and device.type == "cuda":
                    source = source.to(memory_format=torch.channels_last)
                result = self._tiled_upscale(
                    model,
                    source,
                    scale=scale,
                    initial_tile=(
                        512 if tile_size == 0 else max(tile_size, MINIMUM_TILE_SIZE)
                    ),
                    precision=precision,
                    management=management,
                    comfy_utils=comfy_utils,
                )
                chunks.append(result.movedim(-3, -1).clamp(0, 1).to("cpu"))
            return torch.cat(chunks, dim=0)
        finally:
            model.to("cpu")

    def _tiled_upscale(
        self,
        model: UpscaleModel,
        source: torch.Tensor,
        *,
        scale: float,
        initial_tile: int,
        precision: str,
        management: Any,
        comfy_utils: Any,
    ) -> torch.Tensor:
        """Run tiled upscale with Comfy's OOM-aware tile fallback."""

        tile = initial_tile
        overlap = 32
        while tile >= MINIMUM_TILE_SIZE:
            try:
                steps = source.shape[0] * comfy_utils.get_tiled_scale_steps(
                    source.shape[3],
                    source.shape[2],
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                )
                progress = comfy_utils.ProgressBar(steps)
                with self._precision_context(source.device, precision):
                    with torch.inference_mode():
                        result: Any = comfy_utils.tiled_scale(
                            source,
                            model,
                            tile_x=tile,
                            tile_y=tile,
                            overlap=overlap,
                            upscale_amount=scale,
                            pbar=progress,
                        )
                return cast(torch.Tensor, result)
            except Exception as error:
                management.raise_non_oom(error)
                tile //= 2
        raise RuntimeError("Upscale model exhausted safe tile sizes after GPU OOM.")

    @staticmethod
    def _precision_context(
        device: torch.device,
        precision: str,
    ) -> AbstractContextManager[None]:
        """Return CUDA autocast only for an explicitly requested reduced dtype."""

        if device.type != "cuda" or precision not in {"fp16", "bf16"}:
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @staticmethod
    def _ensure_parameter_device(model: torch.nn.Module, device: torch.device) -> None:
        """Repair descriptors whose wrapper move left parameters on another device."""

        for parameter in model.parameters():
            if parameter.device == device:
                continue
            parameter.data = parameter.data.to(device)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(device)


__all__ = ["ModelUpscaler", "UpscaleModel"]
