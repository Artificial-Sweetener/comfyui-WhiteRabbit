# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Exact-rational FPS conversion with optional RIFE output stabilization."""

from __future__ import annotations

from fractions import Fraction
from importlib import import_module
from typing import Protocol, cast

import torch

from ..domain.rife import FpsResampleOptions
from ..shared.tensor_validation import validate_image_batch
from .pixel_hold import (
    dilate,
    gaussian_blur,
    linear_to_srgb,
    srgb_to_linear,
)
from .rife_interpolation import RifeInterpolationEngine


class ProgressBar(Protocol):
    """Subset of Comfy's progress bar used for FPS conversion."""

    def update(self, value: int) -> None:
        """Advance progress by output frames."""


class RifeFpsResampler:
    """Convert frame rate without cumulative floating-point timeline drift."""

    def __init__(self, interpolation: RifeInterpolationEngine | None = None) -> None:
        """Create the resampler with an injectable interpolation engine."""

        self._interpolation = interpolation or RifeInterpolationEngine()

    def resample(
        self,
        frames: torch.Tensor,
        options: FpsResampleOptions,
    ) -> torch.Tensor:
        """Resample a clip and apply each enabled stabilization feature."""

        shape = validate_image_batch(frames)
        if options.input_fps <= 0 or options.output_fps <= 0:
            raise ValueError("fps_in and fps_out must be > 0")
        input_rate = Fraction(str(float(options.input_fps)))
        output_rate = Fraction(str(float(options.output_fps)))
        if input_rate == output_rate:
            return frames
        ratio = input_rate / output_rate
        if ratio.denominator == 1 and ratio.numerator > 1:
            return frames[:: ratio.numerator].contiguous()
        if shape.batch_size <= 1:
            return frames[:1]

        output_count = (
            int(Fraction(shape.batch_size - 1) * output_rate / input_rate) + 1
        )
        output = torch.empty(
            (output_count, shape.height, shape.width, shape.channels),
            dtype=torch.float32,
            device="cpu",
        )
        numerator = input_rate.numerator * output_rate.denominator
        denominator = input_rate.denominator * output_rate.numerator
        accumulator = 0
        progress = self._progress_bar(output_count)
        frames_since_clear = 0
        for output_index in range(output_count):
            source_index = accumulator // denominator
            remainder = accumulator % denominator
            if source_index >= shape.batch_size - 1:
                result = frames[-1:]
            elif remainder == 0:
                result = frames[source_index : source_index + 1]
            else:
                timing = remainder / denominator
                result = self._interpolation.synthesize(
                    options.model_name,
                    frames[source_index : source_index + 1],
                    frames[source_index + 1 : source_index + 2],
                    timing,
                    options.scale_factor,
                    options.ensemble,
                )
                result = self._stabilize(
                    result,
                    frames[source_index : source_index + 1],
                    frames[source_index + 1 : source_index + 2],
                    timing,
                    options,
                )
            output[output_index : output_index + 1].copy_(result)
            accumulator += numerator
            progress.update(1)
            frames_since_clear += 1
            if (
                options.clear_cache_interval > 0
                and frames_since_clear >= options.clear_cache_interval
            ):
                import_module("comfy.model_management").soft_empty_cache()
                frames_since_clear = 0
        return output.clamp(0, 1)

    @staticmethod
    def _stabilize(
        result: torch.Tensor,
        frame_0: torch.Tensor,
        frame_1: torch.Tensor,
        timing: float,
        options: FpsResampleOptions,
    ) -> torch.Tensor:
        """Apply low-frequency, exposure, and edge-band stabilizers."""

        if not any(
            (
                options.linearize,
                options.low_frequency_guardrail,
                options.source_pair_match,
                options.edge_band_lock,
            )
        ):
            return result
        to_linear = srgb_to_linear if options.linearize else _identity
        to_output = linear_to_srgb if options.linearize else _identity
        result_linear = to_linear(result)
        frame_0_linear = to_linear(frame_0)
        frame_1_linear = to_linear(frame_1)
        if options.low_frequency_guardrail and options.low_frequency_sigma > 0:
            high_frequency = result_linear - gaussian_blur(
                result_linear, options.low_frequency_sigma
            )
            target_low = (1 - timing) * gaussian_blur(
                frame_0_linear, options.low_frequency_sigma
            ) + timing * gaussian_blur(frame_1_linear, options.low_frequency_sigma)
            result_linear = high_frequency + target_low
        if options.source_pair_match:
            target = (1 - timing) * frame_0_linear + timing * frame_1_linear
            scale, offset = _exposure_match(
                result_linear,
                target,
                options.match_scale_cap,
                options.match_offset_cap,
            )
            result_linear = (scale * result_linear + offset).clamp(0, 1)
        if options.edge_band_lock:
            delta = (
                (_linear_luma(frame_0_linear) - _linear_luma(frame_1_linear))
                .abs()
                .unsqueeze(-1)
            )
            high = (delta > options.edge_high_threshold).float()
            low = (delta < options.edge_low_threshold).float()
            band = dilate(high, options.edge_band_radius) * low
            if options.edge_band_sigma > 0:
                band = gaussian_blur(band, options.edge_band_sigma).clamp(0, 1)
            nearest = frame_0_linear if timing < 0.5 else frame_1_linear
            result_linear = band * nearest + (1 - band) * result_linear
        return to_output(result_linear).clamp(0, 1)

    @staticmethod
    def _progress_bar(total: int) -> ProgressBar:
        """Create a Comfy progress bar through a typed protocol."""

        comfy_utils = import_module("comfy.utils")
        return cast(ProgressBar, comfy_utils.ProgressBar(total))


def _identity(images: torch.Tensor) -> torch.Tensor:
    """Return images unchanged for disabled linear-light conversion."""

    return images


def _linear_luma(images: torch.Tensor) -> torch.Tensor:
    """Return Rec. 709 luma from NHWC linear RGB images."""

    return 0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2]


def _exposure_match(
    result: torch.Tensor,
    target: torch.Tensor,
    scale_cap: float,
    offset_cap: float,
) -> tuple[float, float]:
    """Solve capped robust luma scale and offset from 5/50/95 percentiles."""

    result_luma = _linear_luma(result).flatten()
    target_luma = _linear_luma(target).flatten()
    quantiles = torch.tensor([0.05, 0.5, 0.95], device=result.device)
    result_low, result_middle, result_high = torch.quantile(result_luma, quantiles)
    target_low, target_middle, target_high = torch.quantile(target_luma, quantiles)
    scale = ((target_high - target_low) / (result_high - result_low + 1e-6)).item()
    scale = max(1 - scale_cap, min(1 + scale_cap, scale))
    offset = (target_middle - scale * result_middle).item()
    return scale, max(-offset_cap, min(offset_cap, offset))


__all__ = ["RifeFpsResampler"]
