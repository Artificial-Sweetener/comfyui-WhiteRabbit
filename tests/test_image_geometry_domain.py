# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for pure image resize geometry."""

from __future__ import annotations

import pytest

from whiterabbit.domain.image_geometry import (
    CropOffset,
    Padding,
    ResizeMode,
    build_resize_plan,
    cover_keep_aspect,
    crop_offset,
    fit_keep_aspect,
    padding_for_anchor,
    parse_pad_color,
)


def test_geometry_matches_the_characterized_integer_math() -> None:
    """Extracted domain geometry preserves fitting, covering, and anchors."""

    assert fit_keep_aspect(1920, 1080, 1024, 1024) == (1024, 576)
    assert cover_keep_aspect(1920, 1080, 1024, 1024) == (1821, 1024)
    assert padding_for_anchor("center", 5, 3) == Padding(2, 3, 1, 2)
    assert padding_for_anchor("bottom-right", 5, 3) == Padding(5, 0, 3, 0)
    assert crop_offset("center", 1821, 1024, 1024, 1024) == CropOffset(398, 0)
    assert crop_offset("right", 1821, 1024, 1024, 1024) == CropOffset(797, 0)


def test_pad_color_matches_characterized_clamping_and_channels() -> None:
    """Pad colors remain normalized, clamped, and channel-aware."""

    assert parse_pad_color("300, -1, 128", 3) == (1.0, 0.0, 128.0 / 255.0)
    assert parse_pad_color("invalid", 3) == (0.0, 0.0, 0.0)
    assert parse_pad_color("255, 0, 128", 4) == (1.0, 0.0, 128.0 / 255.0, 1.0)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ResizeMode.KEEP_ASPECT.value, (1024, 576, 1024, 576)),
        (ResizeMode.STRETCH.value, (1024, 1024, 1024, 1024)),
        (ResizeMode.CROP.value, (1821, 1024, 1024, 1024)),
        (ResizeMode.PAD.value, (1024, 576, 1024, 1024)),
    ],
)
def test_resize_plan_resolves_each_geometry_mode(
    mode: str,
    expected: tuple[int, int, int, int],
) -> None:
    """The plan separates resample dimensions from final output geometry."""

    plan = build_resize_plan(1920, 1080, 1024, 1024, mode, 1, "center")
    assert (
        plan.resize_width,
        plan.resize_height,
        plan.output_width,
        plan.output_height,
    ) == expected
