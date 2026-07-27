# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for the typed Pixel Hold runtime and v3 node."""

from __future__ import annotations

import pytest
import torch

from whiterabbit.domain.pixel_hold import PixelHoldOptions
from whiterabbit.nodes_v3.pixel_hold import PixelHoldV3
from whiterabbit.runtime.pixel_hold import PixelHoldRuntime


def test_runtime_matches_characterized_full_frame_hold() -> None:
    """Low-change pixels lock to the selected batch reference."""

    frames = torch.stack(
        [
            torch.zeros((32, 32, 3), dtype=torch.float32),
            torch.full((32, 32, 3), 0.005, dtype=torch.float32),
        ]
    )
    options = PixelHoldOptions(
        reference_source="batch_index",
        linearize=False,
        automatic_luma=False,
        luma_threshold=0.01,
        gradient_threshold=0.1,
        mode="pixel",
        edge_band=False,
        dilation=0,
        feather_sigma=0,
        processing_device="cpu",
    )
    output, mask = PixelHoldRuntime().apply(frames, None, options)
    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(mask, torch.ones_like(mask))


def test_tile_mode_handles_frames_smaller_than_the_selected_tile() -> None:
    """Small frames use one bounded tile instead of failing in average pooling."""

    frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    options = PixelHoldOptions(
        reference_source="batch_index",
        linearize=False,
        automatic_luma=False,
        mode="tile",
        tile_size=32,
        edge_band=False,
        dilation=0,
        feather_sigma=0,
        processing_device="cpu",
    )

    output, mask = PixelHoldRuntime().apply(frames, None, options)

    assert output.shape == mask.shape == frames.shape


def test_external_reference_requires_rgb() -> None:
    """An incompatible reference fails at the boundary with an actionable error."""

    frames = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
    reference = torch.zeros((1, 8, 8, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="RGB reference"):
        PixelHoldRuntime().apply(frames, reference, PixelHoldOptions())


def test_pixel_hold_v3_schema_preserves_order_and_outputs() -> None:
    """The v3 schema retains every public control in serialized order."""

    schema = PixelHoldV3.define_schema()
    assert schema.node_id == "PixelHold"
    assert [item.id for item in schema.inputs] == [
        "frames",
        "ref_source",
        "ref_index",
        "reference",
        "linearize",
        "auto_luma",
        "auto_k",
        "tau_luma",
        "tau_grad",
        "mode",
        "tile_size",
        "score_mode",
        "edge_band",
        "band_radius",
        "tau_edge_low",
        "tau_edge_high",
        "apply",
        "dilate",
        "feather_sigma",
        "process_on",
        "gpu_clear_every",
    ]
    assert [item.id for item in schema.outputs] == ["images", "mask_preview"]
