# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Pure geometry for image resize, crop, and pad operations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ResizeMode(str, Enum):
    """Supported workflow-facing resize modes."""

    KEEP_ASPECT = "Keep AR"
    STRETCH = "Stretch"
    CROP = "Crop (Cover + Crop)"
    PAD = "Pad (Fit + Pad)"
    ASPECT_DIVISIBLE_CROP = "AR Scale + Divisible Crop"


@dataclass(frozen=True)
class CropOffset:
    """Top-left offset of a crop within a resized image."""

    x: int
    y: int


@dataclass(frozen=True)
class Padding:
    """Padding applied to the four sides of an image."""

    left: int
    right: int
    top: int
    bottom: int


@dataclass(frozen=True)
class ResizePlan:
    """Resolved resize dimensions and post-resize geometry."""

    mode: ResizeMode
    resize_width: int
    resize_height: int
    output_width: int
    output_height: int
    crop: CropOffset = CropOffset(0, 0)
    padding: Padding = Padding(0, 0, 0, 0)


def chunk_spans(item_count: int, maximum_batch_size: int) -> list[tuple[int, int]]:
    """Split an item count into stable half-open batch spans."""

    if maximum_batch_size <= 0 or maximum_batch_size >= item_count:
        return [(0, item_count)]
    return [
        (start, min(item_count, start + maximum_batch_size))
        for start in range(0, item_count, maximum_batch_size)
    ]


def floor_multiple(value: int, divisor: int) -> int:
    """Round down to a positive multiple while preserving legacy clamping."""

    if divisor <= 1:
        return max(1, value)
    return value - (value % divisor)


def ceil_multiple(value: int, divisor: int) -> int:
    """Round up to a positive multiple while preserving legacy clamping."""

    if divisor <= 1:
        return max(1, value)
    remainder = value % divisor
    return value if remainder == 0 else value + divisor - remainder


def fit_keep_aspect(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    """Fit source geometry inside a target box without changing its aspect ratio."""

    if target_width <= 0 and target_height <= 0:
        return source_width, source_height
    if target_width <= 0:
        ratio = target_height / source_height
    elif target_height <= 0:
        ratio = target_width / source_width
    else:
        ratio = min(target_width / source_width, target_height / source_height)
    return (
        max(1, round(source_width * ratio)),
        max(1, round(source_height * ratio)),
    )


def fit_keep_aspect_divisible(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    divisor: int,
) -> tuple[int, int]:
    """Fit with exact source aspect ratio when a divisible representation exists."""

    if divisor <= 1:
        return fit_keep_aspect(source_width, source_height, target_width, target_height)
    fit_width, fit_height = fit_keep_aspect(
        source_width, source_height, target_width, target_height
    )
    common_divisor = math.gcd(source_width, source_height)
    base_width = divisor * (source_width // common_divisor)
    base_height = divisor * (source_height // common_divisor)
    multiplier = min(fit_width // base_width, fit_height // base_height)
    if multiplier >= 1:
        return base_width * multiplier, base_height * multiplier
    return (
        max(divisor, floor_multiple(fit_width, divisor)),
        max(divisor, floor_multiple(fit_height, divisor)),
    )


def cover_keep_aspect(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    """Scale source geometry to cover a target box."""

    ratio = max(target_width / source_width, target_height / source_height)
    return (
        max(1, math.ceil(source_width * ratio)),
        max(1, math.ceil(source_height * ratio)),
    )


def crop_offset(
    position: str,
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
) -> CropOffset:
    """Resolve crop offsets for a nine-point anchor."""

    horizontal_excess = max(0, input_width - output_width)
    vertical_excess = max(0, input_height - output_height)
    left_positions = {"top-left", "left", "bottom-left"}
    right_positions = {"top-right", "right", "bottom-right"}
    top_positions = {"top-left", "top", "top-right"}
    bottom_positions = {"bottom-left", "bottom", "bottom-right"}
    x = 0 if position in left_positions else horizontal_excess
    y = 0 if position in top_positions else vertical_excess
    if position not in left_positions | right_positions:
        x = horizontal_excess // 2
    if position not in top_positions | bottom_positions:
        y = vertical_excess // 2
    return CropOffset(x, y)


def padding_for_anchor(position: str, width: int, height: int) -> Padding:
    """Resolve padding distribution for a nine-point anchor."""

    left = width // 2
    right = width - left
    top = height // 2
    bottom = height - top
    if position in {"top-left", "top", "top-right"}:
        top, bottom = 0, height
    if position in {"bottom-left", "bottom", "bottom-right"}:
        top, bottom = height, 0
    if position in {"top-left", "left", "bottom-left"}:
        left, right = 0, width
    if position in {"top-right", "right", "bottom-right"}:
        left, right = width, 0
    return Padding(left, right, top, bottom)


def parse_pad_color(value: str, channel_count: int) -> tuple[float, ...]:
    """Parse an RGB byte string into normalized channels."""

    try:
        parsed = [int(part.strip()) for part in value.strip().split(",")]
    except ValueError:
        parsed = [0, 0, 0]
    red, green, blue = [max(0, min(255, item)) for item in (parsed + [0, 0, 0])[:3]]
    rgb = (red / 255.0, green / 255.0, blue / 255.0)
    if channel_count == 1:
        return rgb[:1]
    if channel_count == 4:
        return (*rgb, 1.0)
    return rgb


def build_resize_plan(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    resize_mode: str,
    divisible_by: int,
    position: str,
) -> ResizePlan:
    """Resolve workflow resize controls into one immutable execution plan."""

    try:
        mode = ResizeMode(resize_mode)
    except ValueError as error:
        choices = ", ".join(repr(item.value) for item in ResizeMode)
        raise ValueError(f"Unknown resize_mode. Use one of: {choices}.") from error
    divisor = divisible_by if divisible_by > 1 else 1

    if mode is ResizeMode.STRETCH:
        width = floor_multiple(target_width, divisor)
        height = floor_multiple(target_height, divisor)
        return ResizePlan(mode, width, height, width, height)
    if mode is ResizeMode.KEEP_ASPECT:
        width, height = fit_keep_aspect_divisible(
            source_width,
            source_height,
            target_width,
            target_height,
            divisor,
        )
        return ResizePlan(mode, width, height, width, height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"{mode.value} requires non-zero width and height.")

    output_width = floor_multiple(target_width, divisor)
    output_height = floor_multiple(target_height, divisor)
    if mode is ResizeMode.CROP:
        width, height = cover_keep_aspect(
            source_width, source_height, output_width, output_height
        )
        return ResizePlan(
            mode,
            width,
            height,
            output_width,
            output_height,
            crop=crop_offset(position, width, height, output_width, output_height),
        )
    if mode is ResizeMode.PAD:
        width, height = fit_keep_aspect(
            source_width, source_height, output_width, output_height
        )
        padding = padding_for_anchor(
            position, output_width - width, output_height - height
        )
        return ResizePlan(
            mode, width, height, output_width, output_height, padding=padding
        )

    width, height, output_width, output_height = _aspect_divisible_crop(
        source_width,
        source_height,
        target_width,
        target_height,
        divisor,
    )
    return ResizePlan(
        mode,
        width,
        height,
        output_width,
        output_height,
        crop=crop_offset(position, width, height, output_width, output_height),
    )


def _aspect_divisible_crop(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    divisor: int,
) -> tuple[int, int, int, int]:
    """Scale the source long side and crop only its short side to divisibility."""

    divisible_width = floor_multiple(target_width, divisor)
    divisible_height = floor_multiple(target_height, divisor)
    if source_height >= source_width:
        resize_height = divisible_height
        resize_width = max(1, round(source_width * resize_height / source_height))
        if resize_width < divisor:
            raise ValueError(
                f"AR Scale + Divisible Crop: scaled width {resize_width}px "
                f"< divisible_by {divisor}."
            )
        output_width = min(divisible_width, floor_multiple(resize_width, divisor))
        return resize_width, resize_height, output_width, resize_height
    resize_width = divisible_width
    resize_height = max(1, round(source_height * resize_width / source_width))
    if resize_height < divisor:
        raise ValueError(
            f"AR Scale + Divisible Crop: scaled height {resize_height}px "
            f"< divisible_by {divisor}."
        )
    output_height = min(divisible_height, floor_multiple(resize_height, divisor))
    return resize_width, resize_height, resize_width, output_height


__all__ = [
    "CropOffset",
    "Padding",
    "ResizeMode",
    "ResizePlan",
    "build_resize_plan",
    "ceil_multiple",
    "chunk_spans",
    "crop_offset",
    "fit_keep_aspect",
    "fit_keep_aspect_divisible",
    "floor_multiple",
    "padding_for_anchor",
    "parse_pad_color",
]
