# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Contract tests for typed Comfy v3 scaling nodes."""

from __future__ import annotations

import pytest
import torch

from whiterabbit.nodes_v3.scaling import (
    BatchResizeWithLanczosV3,
    UpscaleWithModelAdvancedV3,
)
from whiterabbit.runtime.image_resampling import LanczosResampler


def test_scaling_v3_node_ids_and_input_order_preserve_workflows() -> None:
    """Scaling schemas retain serialized IDs and ordered widget names."""

    expected = {
        UpscaleWithModelAdvancedV3: (
            "UpscaleWithModelAdvanced",
            [
                "upscale_model",
                "image",
                "max_batch_size",
                "tile_size",
                "channels_last",
                "precision",
            ],
        ),
        BatchResizeWithLanczosV3: (
            "BatchResizeWithLanczos",
            [
                "image",
                "width",
                "height",
                "resize_mode",
                "divisible_by",
                "max_batch_size",
                "sinc_window",
                "pad_color",
                "crop_position",
                "precision",
                "mask",
            ],
        ),
    }
    for node_class, (node_id, input_ids) in expected.items():
        schema = node_class.define_schema()
        assert schema.node_id == node_id
        assert [input_item.id for input_item in schema.inputs] == input_ids


def test_resize_mask_validation_supports_single_mask_broadcast() -> None:
    """Mask inputs are aligned explicitly and one mask can cover a whole batch."""

    mask = torch.ones((8, 12), dtype=torch.float32)
    normalized = LanczosResampler._normalize_mask(mask, 3, 8, 12)
    assert normalized is not None
    assert normalized.shape == (3, 8, 12)
    with pytest.raises(ValueError, match="spatial dimensions"):
        LanczosResampler._normalize_mask(torch.ones((1, 7, 12)), 3, 8, 12)
    with pytest.raises(ValueError, match="batch size"):
        LanczosResampler._normalize_mask(torch.ones((2, 8, 12)), 3, 8, 12)
