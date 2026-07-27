# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Comfy v3 node for automatic loop end cropping."""

from __future__ import annotations

from typing import Any

import torch

from ..services.loop_autocrop import LoopAutocropService
from ._api import ComfyNodeBase, io

_SERVICE = LoopAutocropService()


class AutocropToLoopV3(ComfyNodeBase):
    """Expose multi-metric automatic loop cropping to ComfyUI."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable Autocrop to Loop workflow schema."""

        return io.Schema(
            node_id="AutocropToLoop",
            display_name="🐇 Autocrop to Loop",
            category="video utils",
            description=(
                "Auto-crops the clip to create a smoother loop: tests crops from "
                "the end and scores the seam so it feels like a normal step."
            ),
            inputs=_autocrop_inputs(),
            outputs=[
                io.Image.Output("cropped_clip"),
                io.Int.Output("end_crop_frames"),
                io.Int.Output("cropped_length"),
                io.Float.Output("score"),
                io.String.Output("diagnostics_csv"),
            ],
        )

    @classmethod
    def execute(
        cls,
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
        """Score and crop the clip through the typed application service."""

        return _SERVICE.find_and_crop(
            clip_frames,
            max_end_crop_frames,
            include_first_step,
            include_last_step,
            include_global_median_step,
            seam_window_frames,
            distance_metric,
            score_in_8bit,
            use_ssim_similarity,
            use_exposure_guard,
            use_flow_guard,
            weight_step_size,
            weight_similarity,
            weight_exposure,
            weight_flow,
            ssim_downsample_scales,
            accelerate_with_gpu,
            use_mixed_precision,
        )


def _autocrop_inputs() -> list[Any]:
    """Build ordered Autocrop controls separately from execution logic."""

    return [
        io.Image.Input(
            "clip_frames",
            tooltip=(
                "Your full clip (NHWC, 0–1). Tries every crop from "
                "0..max_end_crop_frames and returns the best loop."
            ),
        ),
        io.Int.Input(
            "max_end_crop_frames",
            default=12,
            min=0,
            max=10000,
            tooltip=(
                "Largest crop to test at the END. Higher = more candidates "
                "(slower), but potentially better."
            ),
        ),
        io.Boolean.Input(
            "include_first_step",
            default=True,
            tooltip=(
                "Use the first neighbor pair (frame 0→1) as a target step "
                "size/similarity."
            ),
        ),
        io.Boolean.Input(
            "include_last_step",
            default=True,
            tooltip="Use the last neighbor pair inside the KEPT region as a target.",
        ),
        io.Boolean.Input(
            "include_global_median_step",
            default=False,
            tooltip=(
                "Also use the median step across the KEPT region (needs ≥3 frames). "
                "Helps ignore outliers."
            ),
        ),
        io.Int.Input(
            "seam_window_frames",
            default=2,
            min=1,
            max=6,
            tooltip=(
                "Average over multiple aligned pairs across the seam. Larger = "
                "more robust."
            ),
        ),
        io.Combo.Input(
            "distance_metric",
            options=["L1", "MSE"],
            default="L1",
            tooltip=(
                "How to measure step size for matching. L1 is usually more "
                "forgiving; MSE penalizes big errors more."
            ),
        ),
        io.Boolean.Input(
            "score_in_8bit",
            default=False,
            tooltip=(
                "Score with an 8-bit view (simulate export). Output video still "
                "stays float."
            ),
        ),
        io.Boolean.Input(
            "use_ssim_similarity",
            default=True,
            tooltip=(
                "Include SSIM so the seam ‘looks’ like a normal neighbor—avoid "
                "freeze or jump."
            ),
        ),
        io.Boolean.Input(
            "use_exposure_guard",
            default=True,
            tooltip="Promote smooth brightness across the seam (reduces flicker pops).",
        ),
        io.Boolean.Input(
            "use_flow_guard",
            default=False,
            tooltip=(
                "Encourage consistent motion across the seam (needs OpenCV; slower)."
            ),
        ),
        io.Float.Input(
            "weight_step_size",
            default=0.55,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip="Importance of matching step size. Higher = less freeze/jump risk.",
        ),
        io.Float.Input(
            "weight_similarity",
            default=0.30,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip=(
                "Importance of visual similarity (SSIM). Helps avoid a "
                "frozen-looking seam."
            ),
        ),
        io.Float.Input(
            "weight_exposure",
            default=0.10,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip="Importance of even brightness across the seam.",
        ),
        io.Float.Input(
            "weight_flow",
            default=0.05,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip="Importance of motion continuity across the seam.",
        ),
        io.String.Input(
            "ssim_downsample_scales",
            default="1,2",
            tooltip=(
                "SSIM scales to average, as a comma list. Example: 1,2 = full-res "
                "and half-res."
            ),
        ),
        io.Boolean.Input(
            "accelerate_with_gpu",
            default=True,
            tooltip=(
                "If ON and CUDA is available, run scoring on GPU for a big speedup "
                "(same results)."
            ),
        ),
        io.Boolean.Input(
            "use_mixed_precision",
            default=True,
            tooltip=(
                "If ON (with GPU), use mixed precision for SSIM/conv math (faster "
                "on larger clips)."
            ),
        ),
    ]


AUTOCROP_NODES = [AutocropToLoopV3]

__all__ = ["AUTOCROP_NODES", "AutocropToLoopV3"]
