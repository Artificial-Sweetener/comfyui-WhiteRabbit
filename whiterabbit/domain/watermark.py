# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Pure configuration and placement geometry for image watermarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WatermarkPosition:
    """Top-left watermark coordinates in base-image pixels."""

    x: int
    y: int


@dataclass(frozen=True)
class WatermarkOptions:
    """Controls for one prepared watermark applied across an image batch."""

    position: str
    scale_percent: int
    transparency_percent: int
    rotation_degrees: int
    padding_x: int
    padding_y: int
    optical_padding: bool
    optical_strength: int
    maximum_batch_size: int
    sinc_window: int
    precision: str


def position_watermark(
    position: str,
    base_width: int,
    base_height: int,
    watermark_width: int,
    watermark_height: int,
    padding_x: int,
    padding_y: int,
) -> WatermarkPosition:
    """Place a watermark using the characterized corner and center anchors."""

    normalized = (position or "bottom-right").strip().lower()
    if normalized == "center":
        return WatermarkPosition(
            (base_width - watermark_width) // 2,
            (base_height - watermark_height) // 2,
        )
    if "left" in normalized:
        x = padding_x
    elif "right" in normalized:
        x = base_width - watermark_width - padding_x
    else:
        x = (base_width - watermark_width) // 2
    if "top" in normalized:
        y = padding_y
    elif "bottom" in normalized:
        y = base_height - watermark_height - padding_y
    else:
        y = (base_height - watermark_height) // 2
    return WatermarkPosition(x, y)


__all__ = ["WatermarkOptions", "WatermarkPosition", "position_watermark"]
