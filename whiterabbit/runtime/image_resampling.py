# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tensor execution for image resizing plans."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as functional

from ..domain.image_geometry import ResizeMode, ResizePlan, chunk_spans, parse_pad_color
from ..shared.tensor_validation import validate_image_batch


class ProgressBar(Protocol):
    """Subset of Comfy's progress bar used by batch image operations."""

    def update(self, value: int) -> None:
        """Advance progress by a number of completed items."""


class LanczosResampler:
    """Execute gamma-correct TorchLanc resizes with Comfy device selection."""

    def resize(
        self,
        images: torch.Tensor,
        plan: ResizePlan,
        *,
        maximum_batch_size: int,
        sinc_window: int,
        pad_color: str,
        precision: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Resize an image batch and its optional aligned mask."""

        shape = validate_image_batch(images, name="image")
        source = images.float().clamp(0, 1)
        mask = self._normalize_mask(mask, shape.batch_size, shape.height, shape.width)
        device = self._device()
        progress = self._progress_bar(shape.batch_size)
        output_images: list[torch.Tensor] = []
        output_masks: list[torch.Tensor] = []

        for start, end in chunk_spans(shape.batch_size, maximum_batch_size):
            batch = source[start:end].movedim(-1, 1).to(device, non_blocking=True)
            resized = self._lanczos(
                batch,
                height=plan.resize_height,
                width=plan.resize_width,
                sinc_window=sinc_window,
                precision=precision,
            )
            transformed = self._apply_geometry(resized, plan, pad_color)
            output_images.append(transformed.to("cpu").movedim(1, -1))

            if mask is not None:
                mask_batch = mask[start:end].unsqueeze(1).to(device, non_blocking=True)
                resized_mask = self._nearest(
                    mask_batch, (plan.resize_height, plan.resize_width)
                )
                output_masks.append(
                    self._apply_mask_geometry(resized_mask, plan).squeeze(1).to("cpu")
                )
            progress.update(end - start)

        image_output = torch.cat(output_images, dim=0)
        mask_output = (
            torch.cat(output_masks, dim=0)
            if output_masks
            else torch.zeros(
                (shape.batch_size, plan.output_height, plan.output_width),
                dtype=torch.float32,
            )
        )
        return image_output, plan.output_width, plan.output_height, mask_output

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor | None,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        """Validate an aligned mask and broadcast a single mask across the batch."""

        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor.")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError(
                "mask must have shape (batch, height, width) or (height, width)."
            )
        if tuple(mask.shape[1:]) != (height, width):
            raise ValueError("mask spatial dimensions must match the input image.")
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        elif mask.shape[0] != batch_size:
            raise ValueError("mask batch size must be one or match the image batch.")
        return mask.float().clamp(0, 1)

    def _apply_geometry(
        self,
        resized: torch.Tensor,
        plan: ResizePlan,
        pad_color: str,
    ) -> torch.Tensor:
        """Apply crop or pad geometry to a resized image batch."""

        if plan.mode in {ResizeMode.CROP, ResizeMode.ASPECT_DIVISIBLE_CROP}:
            return resized[
                :,
                :,
                plan.crop.y : plan.crop.y + plan.output_height,
                plan.crop.x : plan.crop.x + plan.output_width,
            ]
        if plan.mode is not ResizeMode.PAD:
            return resized

        channels = resized.shape[1]
        color = torch.tensor(
            parse_pad_color(pad_color, channels),
            device=resized.device,
            dtype=resized.dtype,
        ).view(1, channels, 1, 1)
        canvas = color.expand(
            resized.shape[0],
            -1,
            plan.output_height,
            plan.output_width,
        ).clone()
        top = plan.padding.top
        left = plan.padding.left
        canvas[
            :,
            :,
            top : top + plan.resize_height,
            left : left + plan.resize_width,
        ] = resized
        return canvas

    def _apply_mask_geometry(
        self,
        resized: torch.Tensor,
        plan: ResizePlan,
    ) -> torch.Tensor:
        """Apply the image plan to a nearest-neighbor mask."""

        if plan.mode in {ResizeMode.CROP, ResizeMode.ASPECT_DIVISIBLE_CROP}:
            return resized[
                :,
                :,
                plan.crop.y : plan.crop.y + plan.output_height,
                plan.crop.x : plan.crop.x + plan.output_width,
            ]
        if plan.mode is not ResizeMode.PAD:
            return resized
        canvas = torch.zeros(
            (resized.shape[0], 1, plan.output_height, plan.output_width),
            device=resized.device,
            dtype=resized.dtype,
        )
        top = plan.padding.top
        left = plan.padding.left
        canvas[
            :,
            :,
            top : top + plan.resize_height,
            left : left + plan.resize_width,
        ] = resized
        return canvas

    @staticmethod
    def _nearest(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Use exact nearest sampling when the installed torch supports it."""

        try:
            return functional.interpolate(images, size=size, mode="nearest-exact")
        except RuntimeError:
            return functional.interpolate(images, size=size, mode="nearest")

    @staticmethod
    def _lanczos(
        images: torch.Tensor,
        *,
        height: int,
        width: int,
        sinc_window: int,
        precision: str,
    ) -> torch.Tensor:
        """Call TorchLanc through a narrow dynamic boundary."""

        torchlanc = import_module("torchlanc")
        result: Any = torchlanc.lanczos_resize(
            images,
            height=height,
            width=width,
            a=sinc_window,
            precision=precision,
            clamp=True,
            chunk_size=0,
        )
        return cast(torch.Tensor, result)

    @staticmethod
    def _device() -> torch.device:
        """Use Comfy's selected torch device."""

        management = import_module("comfy.model_management")
        return cast(torch.device, management.get_torch_device())

    @staticmethod
    def _progress_bar(total: int) -> ProgressBar:
        """Construct a Comfy progress bar through a typed protocol."""

        comfy_utils = import_module("comfy.utils")
        return cast(ProgressBar, comfy_utils.ProgressBar(total))


__all__ = ["LanczosResampler"]
