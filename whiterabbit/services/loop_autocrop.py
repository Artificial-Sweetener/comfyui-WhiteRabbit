# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Application service for automatic video-loop end cropping."""

from __future__ import annotations

import torch

from ..domain.loop_autocrop import LoopAutocropOptions
from ..runtime.loop_autocrop import LoopAutocropRuntime


class LoopAutocropService:
    """Translate workflow controls into the typed loop scoring runtime."""

    def __init__(self, runtime: LoopAutocropRuntime | None = None) -> None:
        """Create the service with an injectable scoring runtime."""

        self._runtime = runtime or LoopAutocropRuntime()

    def find_and_crop(
        self,
        clip_frames: torch.Tensor,
        max_end_crop_frames: int,
        include_first_step: bool,
        include_last_step: bool,
        include_global_median_step: bool,
        seam_window_frames: int,
        distance_metric: str,
        score_in_8bit: bool,
        use_ssim_similarity: bool,
        use_exposure_guard: bool,
        use_flow_guard: bool,
        weight_step_size: float,
        weight_similarity: float,
        weight_exposure: float,
        weight_flow: float,
        ssim_downsample_scales: str,
        accelerate_with_gpu: bool,
        use_mixed_precision: bool,
    ) -> tuple[torch.Tensor, int, int, float, str]:
        """Select and return the most natural crop from the end of a clip."""

        options = LoopAutocropOptions(
            maximum_end_crop=max_end_crop_frames,
            include_first_step=include_first_step,
            include_last_step=include_last_step,
            include_global_median_step=include_global_median_step,
            seam_window_frames=seam_window_frames,
            distance_metric=distance_metric,
            score_in_8bit=score_in_8bit,
            use_ssim_similarity=use_ssim_similarity,
            use_exposure_guard=use_exposure_guard,
            use_flow_guard=use_flow_guard,
            weight_step_size=weight_step_size,
            weight_similarity=weight_similarity,
            weight_exposure=weight_exposure,
            weight_flow=weight_flow,
            ssim_downsample_scales=ssim_downsample_scales,
            accelerate_with_gpu=accelerate_with_gpu,
            use_mixed_precision=use_mixed_precision,
        )
        return self._runtime.find(clip_frames, options)


__all__ = ["LoopAutocropService"]
