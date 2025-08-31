# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

from .interpolation import (
    RIFE_FPS_Resample,
    RIFE_SeamTimingAnalyzer,
    RIFE_VFI_Advanced,
    RIFE_VFI_Opt,
)
from .noise_control import PixelHold
from .post_process import BatchWatermarkSingle
from .scaling import BatchResizeWithLanczos, UpscaleWithModelAdvanced
from .video_loop import (
    AssembleLoopFrames,
    AutocropToLoop,
    PrepareLoopFrames,
    RollFrames,
    TrimBatchEnds,
    UnrollFrames,
)

NODE_CLASS_MAPPINGS = {
    "PrepareLoopFrames": PrepareLoopFrames,
    "AssembleLoopFrames": AssembleLoopFrames,
    "RollFrames": RollFrames,
    "UnrollFrames": UnrollFrames,
    "AutocropToLoop": AutocropToLoop,
    "TrimBatchEnds": TrimBatchEnds,
    "RIFE_VFI_Opt": RIFE_VFI_Opt,
    "RIFE_VFI_Advanced": RIFE_VFI_Advanced,
    "RIFE_SeamTimingAnalyzer": RIFE_SeamTimingAnalyzer,
    "RIFE_FPS_Resample": RIFE_FPS_Resample,
    "PixelHold": PixelHold,
    "UpscaleWithModelAdvanced": UpscaleWithModelAdvanced,
    "BatchResizeWithLanczos": BatchResizeWithLanczos,
    "BatchWatermarkSingle": BatchWatermarkSingle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrepareLoopFrames": "🐇 Prepare Loop Frames",
    "AssembleLoopFrames": "🐇 Assemble Loop Frames",
    "RollFrames": "🐇 Roll Frames",
    "UnrollFrames": "🐇 Unroll Frames",
    "AutocropToLoop": "🐇 Autocrop to Loop",
    "TrimBatchEnds": "🐇 Trim Batch Ends",
    "RIFE_VFI_Opt": "🐇 RIFE VFI Interpolate by Multiple",
    "RIFE_VFI_Advanced": "🐇 RIFE VFI Custom Timing",
    "RIFE_SeamTimingAnalyzer": "🐇 RIFE Seam Timing Analyzer",
    "RIFE_FPS_Resample": "🐇 RIFE VFI FPS Resample",
    "PixelHold": "🐇 Pixel Hold",
    "UpscaleWithModelAdvanced": "🐇 Upscale w/ Model (Advanced)",
    "BatchResizeWithLanczos": "🐇 Batch Resize w/ Lanczos",
    "BatchWatermarkSingle": "🐇 Watermark",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
