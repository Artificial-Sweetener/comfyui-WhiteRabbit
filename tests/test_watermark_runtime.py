# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for watermark geometry, compositing, and v3 registration."""

from __future__ import annotations

import pytest
import torch

from whiterabbit.domain.watermark import WatermarkPosition, position_watermark
from whiterabbit.nodes_v3.watermark import BatchWatermarkSingleV3
from whiterabbit.runtime.watermark_composite import (
    PreparedWatermark,
    WatermarkCompositor,
    rotate_bicubic_expand,
)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("top-left", WatermarkPosition(10, 20)),
        ("top-right", WatermarkPosition(70, 20)),
        ("bottom-left", WatermarkPosition(10, 60)),
        ("bottom-right", WatermarkPosition(70, 60)),
        ("center", WatermarkPosition(40, 40)),
    ],
)
def test_positioning_matches_the_characterized_contract(
    position: str,
    expected: WatermarkPosition,
) -> None:
    """Corner padding and center placement preserve exact integer behavior."""

    assert position_watermark(position, 100, 100, 20, 20, 10, 20) == expected


def test_premultiplied_overlay_preserves_base_alpha() -> None:
    """RGBA input receives RGB compositing without rewriting its alpha channel."""

    images = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    images[:, 3] = 0.25
    prepared = PreparedWatermark(
        premultiplied_rgb=torch.full((3, 2, 2), 0.5),
        alpha=torch.full((1, 2, 2), 0.5),
        base_x=1,
        base_y=1,
        end_x=3,
        end_y=3,
    )
    output = WatermarkCompositor._composite(images, prepared)
    torch.testing.assert_close(output[:, :3, 1:3, 1:3], torch.full((1, 3, 2, 2), 0.5))
    torch.testing.assert_close(output[:, 3], torch.full((1, 4, 4), 0.25))


def test_rotation_expands_the_watermark_canvas() -> None:
    """A quarter turn swaps non-square watermark dimensions without clipping."""

    watermark = torch.ones((1, 4, 2, 6), dtype=torch.float32)
    rotated = rotate_bicubic_expand(watermark, 90)
    assert rotated.shape == (1, 4, 6, 2)


def test_watermark_v3_schema_preserves_public_input_order() -> None:
    """The v3 node retains the serialized watermark widget contract."""

    schema = BatchWatermarkSingleV3.define_schema()
    assert schema.node_id == "BatchWatermarkSingle"
    assert [item.id for item in schema.inputs] == [
        "image",
        "watermark",
        "position",
        "scale",
        "transparency",
        "rotation",
        "padding_x",
        "padding_y",
        "optical_padding",
        "optical_strength",
        "max_batch_size",
        "sinc_window",
        "precision",
    ]
