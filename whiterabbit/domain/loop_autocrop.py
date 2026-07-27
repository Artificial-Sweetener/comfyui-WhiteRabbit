# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Configuration and result models for automatic loop cropping."""

from __future__ import annotations

from dataclasses import dataclass

DIAGNOSTICS_HEADER = (
    "end_crop,score,D_seam,D_target,S_seam,S_target,E_seam,E_target,F_seam,F_target"
)


@dataclass(frozen=True)
class LoopAutocropOptions:
    """Scoring controls for loop end-crop selection."""

    maximum_end_crop: int
    include_first_step: bool
    include_last_step: bool
    include_global_median_step: bool
    seam_window_frames: int
    distance_metric: str
    score_in_8bit: bool
    use_ssim_similarity: bool
    use_exposure_guard: bool
    use_flow_guard: bool
    weight_step_size: float
    weight_similarity: float
    weight_exposure: float
    weight_flow: float
    ssim_downsample_scales: str
    accelerate_with_gpu: bool
    use_mixed_precision: bool


@dataclass(frozen=True)
class CandidateMetrics:
    """Measured and target metrics for one end-crop candidate."""

    end_crop: int
    score: float
    seam_distance: float
    target_distance: float
    seam_similarity: float
    target_similarity: float
    seam_exposure: float
    target_exposure: float
    seam_flow: float
    target_flow: float

    def to_csv(self) -> str:
        """Serialize the diagnostics row in the established workflow format."""

        return (
            f"{self.end_crop},{self.score:.6f},{self.seam_distance:.6f},"
            f"{self.target_distance:.6f},{self.seam_similarity:.6f},"
            f"{self.target_similarity:.6f},{self.seam_exposure:.6f},"
            f"{self.target_exposure:.6f},{self.seam_flow:.6f},"
            f"{self.target_flow:.6f}"
        )


__all__ = [
    "CandidateMetrics",
    "DIAGNOSTICS_HEADER",
    "LoopAutocropOptions",
]
