# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Application services for all WhiteRabbit RIFE workflows."""

from __future__ import annotations

from typing import Any, cast

import torch

from ..domain.rife import (
    FpsResampleOptions,
    map_timing,
    parse_custom_timings,
)
from ..runtime.rife_fps import RifeFpsResampler
from ..runtime.rife_interpolation import (
    InterpolationStates,
    RifeInterpolationEngine,
)
from ..runtime.rife_seam import RifeSeamTimingAnalyzer


class RifeService:
    """Coordinate interpolation engines behind stable workflow signatures."""

    def __init__(
        self,
        interpolation: RifeInterpolationEngine | None = None,
        fps_resampler: RifeFpsResampler | None = None,
        seam_analyzer: RifeSeamTimingAnalyzer | None = None,
    ) -> None:
        """Create related services around one shared model cache."""

        shared = interpolation or RifeInterpolationEngine()
        self._interpolation = shared
        self._fps_resampler = fps_resampler or RifeFpsResampler(shared)
        self._seam_analyzer = seam_analyzer or RifeSeamTimingAnalyzer(shared)

    def interpolate(
        self,
        ckpt_name: str,
        frames: torch.Tensor,
        multiplier: int = 2,
        scale_factor: float = 1.0,
        ensemble: bool = True,
        clear_cache_after_n_frames: int = 10,
        optional_interpolation_states: Any = None,
    ) -> tuple[torch.Tensor]:
        """Interpolate by a uniform output multiple."""

        states = cast(InterpolationStates | None, optional_interpolation_states)
        return (
            self._interpolation.interpolate_by_multiplier(
                frames,
                ckpt_name,
                multiplier,
                scale_factor,
                ensemble,
                clear_cache_after_n_frames,
                interpolation_states=states,
            ),
        )

    def interpolate_advanced(
        self,
        ckpt_name: str,
        frames: torch.Tensor,
        multiplier: int = 2,
        timing_mode: str = "linear",
        timing_gamma: float = 1.0,
        minimum_timing: float = 0.0,
        maximum_timing: float = 1.0,
        scale_factor: float = 1.0,
        ensemble: bool = True,
        clear_cache_after_n_frames: int = 10,
        custom_t_list_csv: str = "",
        optional_interpolation_states: Any = None,
    ) -> tuple[torch.Tensor]:
        """Interpolate by a multiple with an arbitrary timing map."""

        if multiplier <= 0:
            return (frames,)
        custom = (
            parse_custom_timings(custom_t_list_csv)
            if custom_t_list_csv and timing_mode == "custom_list"
            else []
        )

        def mapper(timestep: float) -> float:
            return map_timing(
                timestep,
                timing_mode,
                timing_gamma,
                minimum_timing,
                maximum_timing,
                custom,
                multiplier,
            )

        states = cast(InterpolationStates | None, optional_interpolation_states)
        return (
            self._interpolation.interpolate_by_multiplier(
                frames,
                ckpt_name,
                multiplier,
                scale_factor,
                ensemble,
                clear_cache_after_n_frames,
                timing_mapper=mapper,
                interpolation_states=states,
            ),
        )

    def resample_fps(
        self,
        ckpt_name: str,
        fps_in: float,
        fps_out: float,
        frames: torch.Tensor,
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
        """Convert frame rate with every established stabilization control."""

        options = FpsResampleOptions(
            model_name=ckpt_name,
            input_fps=fps_in,
            output_fps=fps_out,
            scale_factor=scale_factor,
            ensemble=ensemble,
            linearize=linearize,
            low_frequency_guardrail=lf_guardrail,
            low_frequency_sigma=lf_sigma,
            source_pair_match=source_pair_match,
            match_scale_cap=match_a_cap,
            match_offset_cap=match_b_cap,
            edge_band_lock=edge_band_lock,
            edge_low_threshold=tau_low,
            edge_high_threshold=tau_high,
            edge_band_radius=band_radius,
            edge_band_sigma=band_soft_sigma,
            clear_cache_interval=clear_cache_after_n_frames,
        )
        return (self._fps_resampler.resample(frames, options),)

    def analyze_seam(
        self,
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
        """Calibrate loop-seam timings against real adjacent motion."""

        return self._seam_analyzer.analyze(
            ckpt_name,
            scale_factor,
            ensemble,
            full_clip,
            max(0, int(multiplier)),
            use_first_two,
            use_last_two,
            use_global_median,
            "L1" if calibrate_metric == "L1" else "MSE",
            calibrate_iters,
            t_min,
            t_max,
            auto_tmax,
            t_cap,
        )


__all__ = ["RifeService"]
