# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tensor runtime for Pixel Hold stabilization."""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any, cast

import torch
import torch.nn.functional as functional

from ..domain.pixel_hold import PixelHoldOptions
from ..shared.tensor_validation import validate_image_batch


def srgb_to_linear(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized sRGB values to linear light."""

    return torch.where(
        images <= 0.04045,
        images / 12.92,
        ((images + 0.055) / 1.055).clamp(min=0) ** 2.4,
    )


def linear_to_srgb(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized linear-light values to sRGB."""

    return torch.where(
        images <= 0.0031308,
        12.92 * images,
        1.055 * images.clamp(min=0) ** (1 / 2.4) - 0.055,
    )


def luma(images: torch.Tensor) -> torch.Tensor:
    """Return Rec. 709 luma for an NHWC RGB batch."""

    return (
        0.2126 * images[..., 0:1]
        + 0.7152 * images[..., 1:2]
        + 0.0722 * images[..., 2:3]
    )


def sobel_magnitude(values: torch.Tensor) -> torch.Tensor:
    """Return Sobel gradient magnitude for a one-channel NHWC batch."""

    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=values.device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=values.device,
    ).view(1, 1, 3, 3)
    nchw = functional.pad(values.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="reflect")
    horizontal = functional.conv2d(nchw, kernel_x)
    vertical = functional.conv2d(nchw, kernel_y)
    return torch.sqrt(horizontal.square() + vertical.square()).permute(0, 2, 3, 1)


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a separable Gaussian blur to an NHWC batch."""

    if sigma <= 0:
        return images
    _, height, width, channels = images.shape
    radius = min(math.ceil(3.0 * sigma), max(0, min(height, width) // 2 - 1))
    if radius <= 0:
        return images
    positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-positions.square() / (2 * sigma * sigma))
    kernel = (kernel / kernel.sum()).to(images.device)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    nchw = images.permute(0, 3, 1, 2).contiguous()
    nchw = functional.conv2d(
        functional.pad(nchw, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=channels,
    )
    nchw = functional.conv2d(
        functional.pad(nchw, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=channels,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate an NHWC mask with replicate edge handling."""

    if radius <= 0:
        return mask
    nchw = functional.pad(
        mask.permute(0, 3, 1, 2),
        (radius, radius, radius, radius),
        mode="replicate",
    )
    return functional.max_pool2d(nchw, kernel_size=2 * radius + 1, stride=1).permute(
        0, 2, 3, 1
    )


class PixelHoldRuntime:
    """Stabilize low-change image regions against a reference frame."""

    @torch.no_grad()
    def apply(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor | None,
        options: PixelHoldOptions,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Pixel Hold and return stabilized images plus a mask preview."""

        shape = validate_image_batch(frames, name="frames")
        if shape.channels != 3:
            raise ValueError("Pixel Hold requires RGB IMAGE batches.")
        reference_batch = self._build_reference(frames, reference, options)
        source = srgb_to_linear(frames) if options.linearize else frames
        reference_linear = (
            srgb_to_linear(reference_batch) if options.linearize else reference_batch
        )
        device = self._processing_device(shape.height, shape.width, options)
        reference_linear = reference_linear.to(device)
        reference_luma = luma(reference_linear)
        reference_gradient = sobel_magnitude(reference_luma)
        reference_low = (
            gaussian_blur(reference_linear.to("cpu"), 13.0)
            if options.apply_mode == "lowfreq"
            else None
        )

        outputs: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        clear_count = 0
        for index in range(shape.batch_size):
            frame = source[index : index + 1].to(device)
            frame_luma = luma(frame)
            luma_delta = (frame_luma - reference_luma[index : index + 1]).abs()
            gradient_delta = (
                sobel_magnitude(frame_luma) - reference_gradient[index : index + 1]
            ).abs()
            threshold = self._luma_threshold(luma_delta, options)
            mask = self._build_mask(
                luma_delta,
                gradient_delta,
                shape.height,
                shape.width,
                threshold,
                options,
            )
            mask = self._protect_edges(
                mask,
                frame_luma,
                reference_luma[index : index + 1],
                options,
            )
            composed = self._compose(
                frame,
                reference_linear[index : index + 1],
                reference_low[index : index + 1] if reference_low is not None else None,
                mask,
                options,
            )
            output = linear_to_srgb(composed) if options.linearize else composed
            outputs.append(output.clamp(0, 1))
            masks.append(mask.to("cpu").repeat(1, 1, 1, 3).clamp(0, 1))

            if device.type == "cuda" and options.gpu_clear_interval > 0:
                clear_count += 1
                if clear_count >= options.gpu_clear_interval:
                    torch.cuda.empty_cache()
                    clear_count = 0
        return torch.cat(outputs), torch.cat(masks)

    def _build_reference(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor | None,
        options: PixelHoldOptions,
    ) -> torch.Tensor:
        """Resolve an external or batch-index reference into one frame per input."""

        frame_count, height, width, _ = frames.shape
        if options.reference_source == "external" and reference is not None:
            reference_shape = validate_image_batch(reference, name="reference")
            if reference_shape.channels != 3:
                raise ValueError("Pixel Hold requires an RGB reference image.")
            resolved = reference[:1]
            if resolved.shape[1:3] != (height, width):
                resolved = self._resize_reference(resolved.to("cpu"), height, width)
        else:
            index = max(0, min(options.reference_index, frame_count - 1))
            resolved = frames[index : index + 1]
        return resolved.repeat(frame_count, 1, 1, 1)

    def _build_mask(
        self,
        luma_delta: torch.Tensor,
        gradient_delta: torch.Tensor,
        height: int,
        width: int,
        luma_threshold: float,
        options: PixelHoldOptions,
    ) -> torch.Tensor:
        """Build a pixel or tile stability mask and apply morphology."""

        if options.mode == "tile":
            luma_score = self._tile_score(
                luma_delta, options.tile_size, options.score_mode
            )
            gradient_score = self._tile_score(
                gradient_delta, options.tile_size, options.score_mode
            )
            mask = (luma_score < luma_threshold).float() * (
                gradient_score < options.gradient_threshold
            ).float()
            mask = functional.interpolate(
                mask.permute(0, 3, 1, 2), size=(height, width), mode="nearest"
            ).permute(0, 2, 3, 1)
        else:
            mask = (luma_delta < luma_threshold).float() * (
                gradient_delta < options.gradient_threshold
            ).float()
        mask = dilate(mask, options.dilation)
        if options.feather_sigma > 0:
            mask = gaussian_blur(mask.to("cpu"), options.feather_sigma).to(
                luma_delta.device
            )
        return mask.clamp(0, 1)

    def _protect_edges(
        self,
        mask: torch.Tensor,
        frame_luma: torch.Tensor,
        reference_luma: torch.Tensor,
        options: PixelHoldOptions,
    ) -> torch.Tensor:
        """Remove a soft band around high-motion edges from the hold mask."""

        if not options.edge_band:
            return mask
        delta = (frame_luma - reference_luma).abs()
        high = (delta > options.edge_high_threshold).float()
        low = (delta < options.edge_low_threshold).float()
        band = dilate(high, options.band_radius) * low
        if options.feather_sigma > 0:
            band = gaussian_blur(band.to("cpu"), options.feather_sigma).to(mask.device)
        return (mask * (1 - band.clamp(0, 1))).clamp(0, 1)

    @staticmethod
    def _compose(
        frame: torch.Tensor,
        reference: torch.Tensor,
        reference_low: torch.Tensor | None,
        mask: torch.Tensor,
        options: PixelHoldOptions,
    ) -> torch.Tensor:
        """Composite full-frequency or low-frequency reference content."""

        if options.apply_mode == "all":
            return (mask * reference + (1 - mask) * frame).to("cpu")
        if reference_low is None:
            raise RuntimeError(
                "Low-frequency composition requires a blurred reference."
            )
        frame_cpu = frame.to("cpu")
        frame_low = gaussian_blur(frame_cpu, 13.0)
        frame_high = frame_cpu - frame_low
        mask_cpu = mask.to("cpu")
        low_mix = mask_cpu * reference_low + (1 - mask_cpu) * frame_low
        return (frame_high + low_mix).clamp(0, 1)

    @staticmethod
    def _tile_score(
        values: torch.Tensor,
        tile_size: int,
        score_mode: str,
    ) -> torch.Tensor:
        """Aggregate a one-channel delta image into tile scores."""

        nchw = values.permute(0, 3, 1, 2)
        tile_size = max(1, min(tile_size, nchw.shape[-2], nchw.shape[-1]))
        if score_mode != "mad_tile":
            return functional.avg_pool2d(
                nchw, kernel_size=tile_size, stride=tile_size
            ).permute(0, 2, 3, 1)
        batch, channels, height, width = nchw.shape
        tile_rows = height // tile_size
        tile_columns = width // tile_size
        cropped = nchw[:, :, : tile_rows * tile_size, : tile_columns * tile_size]
        patches = functional.unfold(cropped, kernel_size=tile_size, stride=tile_size)
        patches = patches.transpose(1, 2).reshape(-1, tile_size * tile_size)
        median = patches.median(dim=1, keepdim=True).values
        return (
            (patches - median)
            .abs()
            .median(dim=1)
            .values.view(batch, tile_rows, tile_columns, channels)
        )

    @staticmethod
    def _luma_threshold(
        luma_delta: torch.Tensor,
        options: PixelHoldOptions,
    ) -> float:
        """Resolve manual or robust per-frame luma sensitivity."""

        if not options.automatic_luma:
            return options.luma_threshold
        median = torch.median(luma_delta.reshape(-1))
        sigma = 1.4826 * median.item()
        return max(0.0, min(4.0 / 255.0, options.automatic_strength * sigma))

    @staticmethod
    def _processing_device(
        height: int,
        width: int,
        options: PixelHoldOptions,
    ) -> torch.device:
        """Resolve explicit CPU/GPU and the legacy large-frame auto threshold."""

        wants_gpu = options.processing_device == "gpu" or (
            options.processing_device == "auto"
            and torch.cuda.is_available()
            and height * width >= 6_000_000
        )
        return torch.device("cuda" if wants_gpu else "cpu")

    @staticmethod
    def _resize_reference(
        reference: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Retain the characterized 8-bit Pillow Lanczos reference resize."""

        numpy = import_module("numpy")
        image_module = import_module("PIL.Image")
        array: Any = (reference[0].numpy() * 255.0).clip(0, 255).astype(numpy.uint8)
        image: Any = image_module.fromarray(array, mode="RGB")
        resized: Any = image.resize(
            (width, height), resample=image_module.Resampling.LANCZOS
        )
        output: Any = numpy.asarray(resized).astype(numpy.float32) / 255.0
        return torch.from_numpy(cast(Any, output)).unsqueeze(0)


__all__ = [
    "PixelHoldRuntime",
    "dilate",
    "gaussian_blur",
    "linear_to_srgb",
    "luma",
    "sobel_magnitude",
    "srgb_to_linear",
]
