# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""GPU-aware preparation and batch compositing for watermark images."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as functional

from ..domain.image_geometry import chunk_spans
from ..domain.watermark import WatermarkOptions, position_watermark
from ..shared.tensor_validation import validate_image_batch


class ProgressBar(Protocol):
    """Subset of Comfy's progress bar used during batch compositing."""

    def update(self, value: int) -> None:
        """Advance progress by a number of images."""


@dataclass(frozen=True)
class PreparedWatermark:
    """A positioned, clipped premultiplied watermark overlay."""

    premultiplied_rgb: torch.Tensor
    alpha: torch.Tensor
    base_x: int
    base_y: int
    end_x: int
    end_y: int


class WatermarkCompositor:
    """Prepare a watermark once and composite it over a Comfy image batch."""

    def apply(
        self,
        images: torch.Tensor,
        watermark_path: Path,
        options: WatermarkOptions,
    ) -> torch.Tensor:
        """Apply a file-backed watermark and return a CPU BHWC float batch."""

        if images.ndim == 3:
            images = images.unsqueeze(0)
        shape = validate_image_batch(images, name="image")
        device = self._device()
        watermark = self._load_rgba(watermark_path, device)
        prepared = self._prepare(
            watermark,
            base_width=shape.width,
            base_height=shape.height,
            options=options,
        )
        if prepared is None:
            return images.float().clamp(0, 1).to("cpu").contiguous()

        progress = self._progress_bar(shape.batch_size)
        chunks: list[torch.Tensor] = []
        for start, end in chunk_spans(shape.batch_size, options.maximum_batch_size):
            batch = (
                images[start:end]
                .movedim(-1, -3)
                .to(device, dtype=torch.float32, non_blocking=True)
                .clamp(0, 1)
            )
            chunks.append(self._composite(batch, prepared).movedim(-3, -1).to("cpu"))
            progress.update(end - start)
        return torch.cat(chunks, dim=0).to(dtype=torch.float32).clamp(0, 1).contiguous()

    def _prepare(
        self,
        watermark: torch.Tensor,
        *,
        base_width: int,
        base_height: int,
        options: WatermarkOptions,
    ) -> PreparedWatermark | None:
        """Resize, rotate, position, and clip a premultiplied RGBA watermark."""

        source_height, source_width = watermark.shape[1:]
        target_width = max(1, round(base_width * options.scale_percent / 100.0))
        target_height = max(
            1, round(source_height * target_width / max(1, source_width))
        )
        alpha = watermark[3:4]
        premultiplied = torch.cat([watermark[:3] * alpha, alpha]).unsqueeze(0)
        resized = self._lanczos(
            premultiplied,
            target_height,
            target_width,
            options.sinc_window,
            options.precision,
        )[0]
        transparency = max(0, min(100, options.transparency_percent)) / 100.0
        if transparency != 1.0:
            resized = resized * transparency
        rotated = rotate_bicubic_expand(resized.unsqueeze(0), options.rotation_degrees)[
            0
        ]
        premultiplied_rgb = rotated[:3]
        final_alpha = rotated[3:4]
        watermark_height, watermark_width = premultiplied_rgb.shape[1:]
        placement = position_watermark(
            options.position,
            base_width,
            base_height,
            watermark_width,
            watermark_height,
            options.padding_x,
            options.padding_y,
        )
        x, y = self._apply_optical_padding(
            placement.x,
            placement.y,
            options.position,
            final_alpha,
            options.optical_padding,
            options.optical_strength,
        )
        start_x = max(0, x)
        start_y = max(0, y)
        end_x = min(base_width, x + watermark_width)
        end_y = min(base_height, y + watermark_height)
        if end_x <= start_x or end_y <= start_y:
            return None
        watermark_x = start_x - x
        watermark_y = start_y - y
        width = end_x - start_x
        height = end_y - start_y
        return PreparedWatermark(
            premultiplied_rgb[
                :, watermark_y : watermark_y + height, watermark_x : watermark_x + width
            ].contiguous(),
            final_alpha[
                :, watermark_y : watermark_y + height, watermark_x : watermark_x + width
            ].contiguous(),
            start_x,
            start_y,
            end_x,
            end_y,
        )

    @staticmethod
    def _composite(
        images: torch.Tensor,
        watermark: PreparedWatermark,
    ) -> torch.Tensor:
        """Premultiplied-alpha composite an overlay into an NCHW batch."""

        alpha = watermark.alpha.unsqueeze(0).expand(images.shape[0], -1, -1, -1)
        color = watermark.premultiplied_rgb.unsqueeze(0).expand(
            images.shape[0], -1, -1, -1
        )
        region = (
            slice(None),
            slice(None),
            slice(watermark.base_y, watermark.end_y),
            slice(watermark.base_x, watermark.end_x),
        )
        channels = images.shape[1]
        if channels == 1:
            rgb = images.repeat(1, 3, 1, 1)
            rgb_region = rgb[region]
            rgb[region] = rgb_region * (1 - alpha) + color
            return (
                0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]
            ).clamp(0, 1)
        rgb_region = images[
            :,
            :3,
            watermark.base_y : watermark.end_y,
            watermark.base_x : watermark.end_x,
        ]
        images[
            :,
            :3,
            watermark.base_y : watermark.end_y,
            watermark.base_x : watermark.end_x,
        ] = rgb_region * (1 - alpha) + color
        return images

    @staticmethod
    def _apply_optical_padding(
        x: int,
        y: int,
        position: str,
        alpha: torch.Tensor,
        enabled: bool,
        strength_percent: int,
    ) -> tuple[int, int]:
        """Shift corner placement using the alpha-weighted visual center."""

        if not enabled or position == "center":
            return x, y
        values = alpha[0]
        denominator = values.sum()
        if denominator.item() <= 1e-8:
            return x, y
        height, width = values.shape
        vertical = torch.linspace(0, height - 1, height, device=values.device)
        horizontal = torch.linspace(0, width - 1, width, device=values.device)
        center_y = (values.sum(dim=1) * vertical).sum() / denominator
        center_x = (values.sum(dim=0) * horizontal).sum() / denominator
        strength = max(0, min(100, strength_percent)) / 100.0
        delta_x = round(((width - 1) * 0.5 - center_x.item()) * strength)
        delta_y = round(((height - 1) * 0.5 - center_y.item()) * strength)
        if "right" in position:
            x += delta_x
        if "left" in position:
            x -= delta_x
        if "bottom" in position:
            y += delta_y
        if "top" in position:
            y -= delta_y
        return x, y

    @staticmethod
    def _load_rgba(path: Path, device: torch.device) -> torch.Tensor:
        """Load a file as normalized NCHW RGBA while preserving transparency."""

        numpy = import_module("numpy")
        image_module = import_module("PIL.Image")
        try:
            with image_module.open(path) as image:
                array: Any = (
                    numpy.asarray(image.convert("RGBA"), dtype=numpy.float32) / 255.0
                )
        except Exception as error:
            raise ValueError(
                f"Failed to load watermark image from '{path}': {error}"
            ) from error
        tensor = torch.from_numpy(cast(Any, array)).to(
            device=device, dtype=torch.float32
        )
        return tensor.permute(2, 0, 1).contiguous()

    @staticmethod
    def _lanczos(
        images: torch.Tensor,
        height: int,
        width: int,
        sinc_window: int,
        precision: str,
    ) -> torch.Tensor:
        """Resize through TorchLanc's gamma-correct kernel."""

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


def rotate_bicubic_expand(images: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate NCHW images around their center with an expanded transparent canvas."""

    normalized = degrees % 360.0
    if normalized == 0:
        return images
    batch, _, height, width = images.shape
    radians = math.radians(normalized)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    output_width = max(1, math.ceil(abs(width * cosine) + abs(height * sine) - 0.0001))
    output_height = max(1, math.ceil(abs(height * cosine) + abs(width * sine) - 0.0001))
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    output_center_x = (output_width - 1) * 0.5
    output_center_y = (output_height - 1) * 0.5
    vertical = torch.linspace(
        0, output_height - 1, output_height, device=images.device, dtype=images.dtype
    )
    horizontal = torch.linspace(
        0, output_width - 1, output_width, device=images.device, dtype=images.dtype
    )
    grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
    relative_x = grid_x - output_center_x
    relative_y = grid_y - output_center_y
    input_x = cosine * relative_x + sine * relative_y + center_x
    input_y = -sine * relative_x + cosine * relative_y + center_y
    normalized_x = (input_x + 0.5) / width * 2 - 1
    normalized_y = (input_y + 0.5) / height * 2 - 1
    grid = torch.stack((normalized_x, normalized_y), dim=-1).unsqueeze(0)
    grid = grid.repeat(batch, 1, 1, 1)
    try:
        return functional.grid_sample(
            images, grid, mode="bicubic", padding_mode="zeros", align_corners=False
        )
    except RuntimeError:
        return functional.grid_sample(
            images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )


__all__ = ["PreparedWatermark", "WatermarkCompositor", "rotate_bicubic_expand"]
