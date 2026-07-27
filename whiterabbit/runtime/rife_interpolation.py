# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Shared RIFE inference engine for multiplier, timing, FPS, and seam services."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, Protocol, cast

import torch

from ..shared.tensor_validation import validate_image_batch
from .rife_architecture import required_core_alignment
from .rife_loading import LoadedRifeModel, RifeModelLoader


class InterpolationStates(Protocol):
    """Compatibility surface for ComfyUI-Frame-Interpolation skip states."""

    def is_frame_skipped(self, frame_index: int) -> bool:
        """Return whether interpolation should be skipped for one source pair."""


class ProgressBar(Protocol):
    """Subset of Comfy's progress bar used for inference."""

    def update(self, value: int) -> None:
        """Advance progress by a number of frames."""


TimingMapper = Callable[[float], float]


class RifeInterpolationEngine:
    """Load RIFE once and synthesize arbitrary positions between source pairs."""

    def __init__(self, loader: RifeModelLoader | None = None) -> None:
        """Create the engine with an injectable Comfy model loader."""

        self._loader = loader or RifeModelLoader()

    def interpolate_by_multiplier(
        self,
        frames: torch.Tensor,
        model_name: str,
        multiplier: int,
        scale_factor: float,
        ensemble: bool,
        clear_cache_interval: int,
        timing_mapper: TimingMapper | None = None,
        interpolation_states: InterpolationStates | None = None,
    ) -> torch.Tensor:
        """Insert `multiplier - 1` frames between each unskipped source pair."""

        shape = validate_image_batch(frames)
        if shape.batch_size < 2:
            raise ValueError("RIFE interpolation requires at least two frames.")
        if multiplier <= 1:
            return frames
        loaded = self._loader.load(model_name, tuple(frames.shape), scale_factor)
        progress = self._progress_bar((shape.batch_size - 1) * (multiplier - 1))
        outputs: list[torch.Tensor] = []
        pairs_since_clear = 0
        with torch.inference_mode():
            for pair_index in range(shape.batch_size - 1):
                outputs.append(frames[pair_index : pair_index + 1].to("cpu"))
                if (
                    interpolation_states is not None
                    and interpolation_states.is_frame_skipped(pair_index)
                ):
                    continue
                for middle_index in range(1, multiplier):
                    timing = middle_index / multiplier
                    if timing_mapper is not None:
                        timing = timing_mapper(timing)
                    outputs.append(
                        self.synthesize_pair(
                            frames[pair_index : pair_index + 1],
                            frames[pair_index + 1 : pair_index + 2],
                            timing,
                            scale_factor,
                            ensemble,
                            loaded,
                        )
                    )
                    progress.update(1)
                pairs_since_clear += 1
                if (
                    clear_cache_interval > 0
                    and pairs_since_clear >= clear_cache_interval
                ):
                    self._clear_cache()
                    pairs_since_clear = 0
        outputs.append(frames[-1:].to("cpu"))
        return torch.cat(outputs).float().clamp(0, 1)

    def synthesize(
        self,
        model_name: str,
        frame_0: torch.Tensor,
        frame_1: torch.Tensor,
        timestep: float,
        scale_factor: float,
        ensemble: bool,
    ) -> torch.Tensor:
        """Load a named model and synthesize one arbitrary pair position."""

        loaded = self._loader.load(model_name, tuple(frame_0.shape), scale_factor)
        with torch.inference_mode():
            return self.synthesize_pair(
                frame_0,
                frame_1,
                timestep,
                scale_factor,
                ensemble,
                loaded,
            )

    @staticmethod
    def synthesize_pair(
        frame_0: torch.Tensor,
        frame_1: torch.Tensor,
        timestep: float,
        scale_factor: float,
        ensemble: bool,
        loaded: LoadedRifeModel,
    ) -> torch.Tensor:
        """Synthesize a pair using an already resident model."""

        height, width = frame_0.shape[1:3]
        image_0 = frame_0.movedim(-1, 1).to(
            device=loaded.device,
            dtype=loaded.dtype,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        image_1 = frame_1.movedim(-1, 1).to(
            device=loaded.device,
            dtype=loaded.dtype,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        if loaded.spec.architecture == "core":
            common_dit: Any = import_module("comfy.ldm.common_dit")
            alignment = required_core_alignment(scale_factor)
            image_0 = common_dit.pad_to_patch_size(
                image_0, (alignment,) * 2, padding_mode="replicate"
            )
            image_1 = common_dit.pad_to_patch_size(
                image_1, (alignment,) * 2, padding_mode="replicate"
            )
        result = loaded.inference_model(
            image_0,
            image_1,
            timestep,
            scale_factor,
            ensemble,
        )
        return (
            result[:, :, :height, :width]
            .movedim(1, -1)
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clamp(0, 1)
        )

    @staticmethod
    def _progress_bar(total: int) -> ProgressBar:
        """Create a Comfy progress bar through a typed protocol."""

        comfy_utils = import_module("comfy.utils")
        return cast(ProgressBar, comfy_utils.ProgressBar(max(1, total)))

    @staticmethod
    def _clear_cache() -> None:
        """Ask Comfy to release safely offloadable memory."""

        management = import_module("comfy.model_management")
        management.soft_empty_cache()


__all__ = [
    "InterpolationStates",
    "RifeInterpolationEngine",
    "TimingMapper",
]
