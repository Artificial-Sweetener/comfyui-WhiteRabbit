# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Comfy v3 node registration for WhiteRabbit."""

from __future__ import annotations

from .loop_autocrop import AutocropToLoopV3
from .looping import (
    AssembleLoopFramesV3,
    PrepareLoopFramesV3,
    RollFramesV3,
    TrimBatchEndsV3,
    UnrollFramesV3,
)
from .pixel_hold import PixelHoldV3
from .rife import (
    RifeFpsResampleV3,
    RifeSeamTimingAnalyzerV3,
    RifeVfiAdvancedV3,
    RifeVfiOptV3,
)
from .scaling import BatchResizeWithLanczosV3, UpscaleWithModelAdvancedV3
from .watermark import BatchWatermarkSingleV3


def get_nodes() -> list[type[object]]:
    """Return every WhiteRabbit v3 node in stable historical order."""

    return [
        PrepareLoopFramesV3,
        AssembleLoopFramesV3,
        RollFramesV3,
        UnrollFramesV3,
        AutocropToLoopV3,
        TrimBatchEndsV3,
        RifeVfiOptV3,
        RifeVfiAdvancedV3,
        RifeSeamTimingAnalyzerV3,
        RifeFpsResampleV3,
        PixelHoldV3,
        UpscaleWithModelAdvancedV3,
        BatchResizeWithLanczosV3,
        BatchWatermarkSingleV3,
    ]


__all__ = ["get_nodes"]
