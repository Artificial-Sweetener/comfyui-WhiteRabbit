# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for typed Comfy v3 loop nodes."""

from __future__ import annotations

import torch

from whiterabbit.nodes_v3.looping import (
    AssembleLoopFramesV3,
    PrepareLoopFramesV3,
    RollFramesV3,
    TrimBatchEndsV3,
    UnrollFramesV3,
)


def test_loop_v3_node_ids_and_input_order_preserve_workflow_contracts() -> None:
    """Migrated schemas retain serialized identifiers and widget order."""

    expected = {
        PrepareLoopFramesV3: ("PrepareLoopFrames", ["images"]),
        AssembleLoopFramesV3: (
            "AssembleLoopFrames",
            ["original_images", "interpolated_frames"],
        ),
        RollFramesV3: ("RollFrames", ["images", "offset"]),
        UnrollFramesV3: ("UnrollFrames", ["images", "base_offset", "m"]),
        TrimBatchEndsV3: (
            "TrimBatchEnds",
            ["clip_frames", "trim_start_frames", "trim_end_frames"],
        ),
    }

    for node_class, (node_id, input_ids) in expected.items():
        schema = node_class.define_schema()
        assert schema.node_id == node_id
        assert schema.category == "video utils"
        assert [input_item.id for input_item in schema.inputs] == input_ids


def test_loop_v3_nodes_delegate_to_typed_services() -> None:
    """V3 execution retains tuple outputs and characterized tensor behavior."""

    frames = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
    seam, original = PrepareLoopFramesV3.execute(frames)
    assert seam.flatten().tolist() == [3.0, 0.0]
    assert original is frames
    rolled, offset = RollFramesV3.execute(frames, 1)
    assert rolled.flatten().tolist() == [1.0, 2.0, 3.0, 0.0]
    assert offset == 1
