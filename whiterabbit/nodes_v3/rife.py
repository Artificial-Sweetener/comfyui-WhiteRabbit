# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Comfy v3 nodes for WhiteRabbit's enhanced RIFE workflows."""

from __future__ import annotations

from typing import Any

import torch

from ..domain.rife import (
    RIFE_MODEL_NAMES,
    RIFE_SCALE_FACTOR_MAXIMUM,
    RIFE_SCALE_FACTOR_MINIMUM,
    RIFE_SCALE_FACTOR_STEP,
    RIFE_TIMING_MODES,
)
from ..services.rife import RifeService
from ._api import ComfyNodeBase, io

_SERVICE = RifeService()
_INTERPOLATION_STATES: Any = None if io is None else io.Custom("INTERPOLATION_STATES")


def _model_input(probe_only: bool = False) -> Any:
    """Build the shared trusted model selector."""

    purpose = (
        "used only to test candidate timings" if probe_only else "for interpolation"
    )
    return io.Combo.Input(
        "ckpt_name",
        options=RIFE_MODEL_NAMES,
        default="rife47.pth",
        tooltip=(
            "Choose RIFE 4.7, 4.9, 4.25, or 4.26 "
            f"({purpose}). Missing catalog models download automatically into "
            "ComfyUI's frame_interpolation model folder."
        ),
    )


def _scale_input(probe_only: bool = False) -> Any:
    """Build the shared quality/speed pyramid selector."""

    tooltip = (
        "Quality vs speed for the probe renders. 1.0 recommended."
        if probe_only
        else (
            "Quality vs speed. 1.0 recommended. Lower = faster/softer; higher = "
            "sharper/slower."
        )
    )
    return io.Float.Input(
        "scale_factor",
        default=1.0,
        min=RIFE_SCALE_FACTOR_MINIMUM,
        max=RIFE_SCALE_FACTOR_MAXIMUM,
        step=RIFE_SCALE_FACTOR_STEP,
        tooltip=tooltip,
    )


def _ensemble_input() -> Any:
    """Build the shared internal bidirectional ensemble toggle."""

    return io.Boolean.Input(
        "ensemble",
        default=True,
        tooltip="Blend forward & backward predictions to reduce artifacts (slower).",
    )


class RifeVfiOptV3(ComfyNodeBase):
    """Expose uniform RIFE interpolation by output multiple."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable interpolate-by-multiple schema."""

        return io.Schema(
            node_id="RIFE_VFI_Opt",
            display_name="🐇 RIFE VFI Interpolate by Multiple",
            category="video utils",
            description=(
                "Interpolate a clip by a chosen multiple using RIFE 4.7, 4.9, "
                "4.25, or 4.26 — inserts evenly spaced in-between frames between "
                "every pair (e.g., ×2 adds 1 frame per pair)."
            ),
            inputs=[
                _model_input(),
                io.Image.Input(
                    "frames", tooltip="Your input clip: one image per frame."
                ),
                io.Int.Input(
                    "multiplier",
                    default=2,
                    min=1,
                    tooltip=(
                        "Adds extra frames to smooth motion: 2 adds 1 new frame "
                        "per pair; 4 adds 3."
                    ),
                ),
                _scale_input(),
                _ensemble_input(),
                io.Int.Input(
                    "clear_cache_after_n_frames",
                    default=10,
                    min=0,
                    max=1000,
                    tooltip=(
                        "Free up GPU memory every N generated frames (advanced). "
                        "Set 0 to never."
                    ),
                ),
                _INTERPOLATION_STATES.Input(
                    "optional_interpolation_states",
                    optional=True,
                    tooltip=(
                        "Don’t create in-between frames for selected frame pairs "
                        "(e.g., scene cuts). Timing stays the same."
                    ),
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        ckpt_name: str,
        frames: torch.Tensor,
        multiplier: int = 2,
        scale_factor: float = 1.0,
        ensemble: bool = True,
        clear_cache_after_n_frames: int = 10,
        optional_interpolation_states: Any = None,
    ) -> tuple[torch.Tensor]:
        """Interpolate uniformly through the typed RIFE service."""

        return _SERVICE.interpolate(
            ckpt_name,
            frames,
            multiplier,
            scale_factor,
            ensemble,
            clear_cache_after_n_frames,
            optional_interpolation_states,
        )


class RifeVfiAdvancedV3(ComfyNodeBase):
    """Expose custom RIFE in-between timing schedules."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable custom timing schema."""

        return io.Schema(
            node_id="RIFE_VFI_Advanced",
            display_name="🐇 RIFE VFI Custom Timing",
            category="video utils",
            description=(
                "Custom timing for RIFE 4.7, 4.9, 4.25, or 4.26 — still "
                "“interpolate by multiple,” but you control where the in-betweens "
                "land (ease in/out, clamps, or your own t-list)."
            ),
            inputs=[
                _model_input(),
                io.Image.Input(
                    "frames", tooltip="Your input clip: one image per frame."
                ),
                io.Int.Input(
                    "multiplier",
                    default=2,
                    min=0,
                    tooltip=(
                        "How many new frames to create between each pair. 0 = "
                        "passthrough (no new frames)."
                    ),
                ),
                io.Combo.Input(
                    "t_mode",
                    options=RIFE_TIMING_MODES,
                    default="linear",
                    tooltip=(
                        "How to spread the new frames over time: straight line, "
                        "ease in/out, limit the range, or provide your own list."
                    ),
                ),
                io.Float.Input(
                    "t_gamma",
                    default=1.0,
                    min=0.05,
                    max=10.0,
                    step=0.05,
                    tooltip="Easing strength for gamma modes. Higher = more easing.",
                ),
                io.Float.Input(
                    "t_min",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "Earliest allowed position between the two frames (0 = "
                        "exactly the first frame). Use with bounded_linear."
                    ),
                ),
                io.Float.Input(
                    "t_max",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "Latest allowed position between the two frames (1 = "
                        "exactly the next frame). Use with bounded_linear."
                    ),
                ),
                _scale_input(),
                _ensemble_input(),
                io.Int.Input(
                    "clear_cache_after_n_frames",
                    default=10,
                    min=0,
                    max=1000,
                    tooltip=(
                        "Free up GPU memory every N generated frames (advanced). "
                        "Set 0 to never."
                    ),
                ),
                io.String.Input(
                    "custom_t_list_csv",
                    default="",
                    optional=True,
                    tooltip=(
                        "Exact positions between the two frames (0–1), "
                        "comma-separated, e.g. 0.18,0.41,0.66. Overrides the schedule."
                    ),
                ),
                _INTERPOLATION_STATES.Input(
                    "optional_interpolation_states",
                    optional=True,
                    tooltip=(
                        "Don’t create in-between frames for selected frame pairs "
                        "(e.g., scene cuts). Timing stays the same."
                    ),
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        ckpt_name: str,
        frames: torch.Tensor,
        multiplier: int = 2,
        t_mode: str = "linear",
        t_gamma: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        scale_factor: float = 1.0,
        ensemble: bool = True,
        clear_cache_after_n_frames: int = 10,
        custom_t_list_csv: str = "",
        optional_interpolation_states: Any = None,
    ) -> tuple[torch.Tensor]:
        """Interpolate with custom timing through the typed RIFE service."""

        return _SERVICE.interpolate_advanced(
            ckpt_name,
            frames,
            multiplier,
            t_mode,
            t_gamma,
            t_min,
            t_max,
            scale_factor,
            ensemble,
            clear_cache_after_n_frames,
            custom_t_list_csv,
            optional_interpolation_states,
        )


class RifeFpsResampleV3(ComfyNodeBase):
    """Expose exact-rational RIFE FPS conversion and stabilizers."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable FPS resample schema."""

        return io.Schema(
            node_id="RIFE_FPS_Resample",
            display_name="🐇 RIFE VFI FPS Resample",
            category="video utils",
            description=(
                "Convert a clip from one FPS to another using RIFE 4.7, 4.9, 4.25, "
                "or 4.26. Non-integer changes synthesize in-betweens; exact integer "
                "downscales just decimate. Includes optional stabilizers to reduce "
                "flicker and protect edges."
            ),
            inputs=_fps_inputs(),
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        ckpt_name: str,
        frames: torch.Tensor,
        fps_in: float,
        fps_out: float,
        scale_factor: float = 1.0,
        ensemble: bool = True,
        linearize: bool = False,
        lf_guardrail: bool = False,
        lf_sigma: float = 13.0,
        source_pair_match: bool = False,
        match_a_cap: float = 0.02,
        match_b_cap: float = 2.0 / 255.0,
        edge_band_lock: bool = False,
        tau_low: float = 1.5 / 255.0,
        tau_high: float = 6.0 / 255.0,
        band_radius: int = 4,
        band_soft_sigma: float = 2.0,
        clear_cache_after_n_frames: int = 10,
    ) -> tuple[torch.Tensor]:
        """Convert FPS through the typed RIFE service."""

        return _SERVICE.resample_fps(
            ckpt_name,
            fps_in,
            fps_out,
            frames,
            scale_factor,
            ensemble,
            linearize,
            lf_guardrail,
            lf_sigma,
            source_pair_match,
            match_a_cap,
            match_b_cap,
            edge_band_lock,
            tau_low,
            tau_high,
            band_radius,
            band_soft_sigma,
            clear_cache_after_n_frames,
        )


def _fps_inputs() -> list[Any]:
    """Build the ordered FPS controls separately from node execution."""

    return [
        _model_input(),
        io.Image.Input("frames", tooltip="Your input clip: one image per frame."),
        io.Float.Input(
            "fps_in",
            default=24.0,
            min=1e-6,
            max=1000.0,
            step=0.01,
            tooltip="Current frame rate of your clip (frames per second).",
        ),
        io.Float.Input(
            "fps_out",
            default=60.0,
            min=1e-6,
            max=2000.0,
            step=0.01,
            tooltip="Target frame rate you want (frames per second).",
        ),
        _scale_input(),
        _ensemble_input(),
        io.Boolean.Input(
            "linearize",
            default=False,
            tooltip=(
                "Work in linear light for more accurate brightness and gradients "
                "(slower)."
            ),
        ),
        io.Boolean.Input(
            "lf_guardrail",
            default=False,
            tooltip=(
                "Keep overall brightness and gradients close to the originals to "
                "reduce flicker."
            ),
        ),
        io.Float.Input(
            "lf_sigma",
            default=13.0,
            min=0.0,
            max=64.0,
            step=0.5,
            tooltip=(
                "How strong the low-frequency smoothing is. Higher = smoother changes."
            ),
        ),
        io.Boolean.Input(
            "source_pair_match",
            default=False,
            tooltip="Match exposure and contrast to the source pair to reduce flicker.",
        ),
        io.Float.Input(
            "match_a_cap",
            default=0.02,
            min=0.0,
            max=0.2,
            step=0.001,
            tooltip="Maximum change allowed for exposure scale.",
        ),
        io.Float.Input(
            "match_b_cap",
            default=2.0 / 255.0,
            min=0.0,
            max=0.1,
            step=0.0005,
            tooltip="Maximum change allowed for brightness offset.",
        ),
        io.Boolean.Input(
            "edge_band_lock",
            default=False,
            tooltip=(
                "Protect sharp edges: near edges, mix in more of the nearest real "
                "frame to avoid smearing."
            ),
        ),
        io.Float.Input(
            "tau_low",
            default=1.5 / 255.0,
            min=0.0,
            max=0.25,
            step=0.0005,
            tooltip="Edge sensitivity: lower threshold (smaller finds more edges).",
        ),
        io.Float.Input(
            "tau_high",
            default=6.0 / 255.0,
            min=0.0,
            max=0.5,
            step=0.0005,
            tooltip=(
                "Edge sensitivity: higher threshold (larger finds only strong edges)."
            ),
        ),
        io.Int.Input(
            "band_radius",
            default=4,
            min=0,
            max=64,
            tooltip="Width of the edge protection band (pixels).",
        ),
        io.Float.Input(
            "band_soft_sigma",
            default=2.0,
            min=0.0,
            max=16.0,
            step=0.5,
            tooltip="Soften the edge band. Higher = smoother.",
        ),
        io.Int.Input(
            "clear_cache_after_n_frames",
            default=10,
            min=0,
            max=1000,
            tooltip=(
                "Free up GPU memory every N output frames (advanced). Set 0 to never."
            ),
        ),
    ]


class RifeSeamTimingAnalyzerV3(ComfyNodeBase):
    """Expose motion-calibrated RIFE loop seam timing."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable seam timing analyzer schema."""

        return io.Schema(
            node_id="RIFE_SeamTimingAnalyzer",
            display_name="🐇 RIFE Seam Timing Analyzer",
            category="video utils",
            description=(
                "Finds a smooth loop timing: measures motion in your clip and "
                "solves a set of t-values across the wrap [last→first] so the seam "
                "blends naturally."
            ),
            inputs=_seam_inputs(),
            outputs=[
                io.String.Output("t_list_csv"),
                io.Int.Output("multiplier"),
            ],
        )

    @classmethod
    def execute(
        cls,
        ckpt_name: str,
        scale_factor: float,
        ensemble: bool,
        full_clip: torch.Tensor,
        multiplier: int,
        use_first_two: bool,
        use_last_two: bool,
        use_global_median: bool,
        calibrate_metric: str,
        calibrate_iters: int,
        t_min: float,
        t_max: float,
        auto_tmax: bool = False,
        t_cap: float = 0.995,
    ) -> tuple[str, int]:
        """Calibrate seam timing through the typed RIFE service."""

        return _SERVICE.analyze_seam(
            ckpt_name,
            scale_factor,
            ensemble,
            full_clip,
            multiplier,
            use_first_two,
            use_last_two,
            use_global_median,
            calibrate_metric,
            calibrate_iters,
            t_min,
            t_max,
            auto_tmax,
            t_cap,
        )


def _seam_inputs() -> list[Any]:
    """Build ordered seam analysis controls."""

    return [
        _model_input(probe_only=True),
        _scale_input(probe_only=True),
        _ensemble_input(),
        io.Image.Input(
            "full_clip",
            tooltip=(
                "Your input clip (≥2 frames). Real motion here decides the loop "
                "seam timing."
            ),
        ),
        io.Int.Input(
            "multiplier",
            default=4,
            min=0,
            tooltip=(
                "How many new frames you plan to create at the loop seam "
                "[last→first]. Set 0 to skip."
            ),
        ),
        io.Boolean.Input(
            "use_first_two",
            default=True,
            tooltip="Match the motion between the first two frames in your clip.",
        ),
        io.Boolean.Input(
            "use_last_two",
            default=True,
            tooltip="Match the motion between the last two frames in your clip.",
        ),
        io.Boolean.Input(
            "use_global_median",
            default=False,
            tooltip=(
                "Use the median motion across the whole clip (needs ≥3 frames). "
                "Helps ignore outliers."
            ),
        ),
        io.Combo.Input(
            "calibrate_metric",
            options=["MSE", "L1"],
            default="MSE",
            tooltip=(
                "How we compare frames while solving: MSE (more sensitive) or L1 "
                "(more forgiving)."
            ),
        ),
        io.Int.Input(
            "calibrate_iters",
            default=12,
            min=4,
            max=24,
            tooltip="Search depth per solve. Higher = slower, but a tighter match.",
        ),
        io.Float.Input(
            "t_min",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.001,
            tooltip=(
                "Earliest allowed blend point at the seam (0 = exactly the last frame)."
            ),
        ),
        io.Float.Input(
            "t_max",
            default=0.96,
            min=0.0,
            max=1.0,
            step=0.001,
            tooltip=(
                "Latest allowed blend point at the seam (keep below 1.0 to avoid "
                "sticking to the first frame)."
            ),
        ),
        io.Boolean.Input(
            "auto_tmax",
            default=False,
            optional=True,
            tooltip=(
                "Automatically push the upper limit closer to the next frame to "
                "hit the target motion step."
            ),
        ),
        io.Float.Input(
            "t_cap",
            default=0.995,
            min=0.5,
            max=0.9999,
            step=0.0001,
            optional=True,
            tooltip=(
                "Safety cap used with the auto upper limit (keeps it just shy of 1.0)."
            ),
        ),
    ]


RIFE_NODES = [
    RifeVfiOptV3,
    RifeVfiAdvancedV3,
    RifeSeamTimingAnalyzerV3,
    RifeFpsResampleV3,
]

__all__ = [
    "RIFE_NODES",
    "RifeFpsResampleV3",
    "RifeSeamTimingAnalyzerV3",
    "RifeVfiAdvancedV3",
    "RifeVfiOptV3",
]
