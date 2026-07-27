# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for RIFE inference orchestration without loading neural weights."""

from __future__ import annotations

import torch

from whiterabbit.domain.rife import FpsResampleOptions, get_rife_model_spec
from whiterabbit.runtime.rife_architecture import (
    required_core_alignment,
    required_legacy_alignment,
)
from whiterabbit.runtime.rife_fps import RifeFpsResampler
from whiterabbit.runtime.rife_interpolation import RifeInterpolationEngine
from whiterabbit.runtime.rife_loading import LoadedRifeModel, RifeModelLoader
from whiterabbit.runtime.rife_seam import RifeSeamTimingAnalyzer


class _RecordingModel:
    """Linearly interpolate while recording quality and ensemble controls."""

    pad_align = 64

    def __init__(self) -> None:
        self.calls: list[tuple[float, float, bool]] = []

    def __call__(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        timestep: float | torch.Tensor,
        scale_factor: float,
        ensemble: bool,
    ) -> torch.Tensor:
        timing = float(timestep)
        self.calls.append((timing, scale_factor, ensemble))
        return torch.lerp(image_0, image_1, timing)


class _FakeLoader(RifeModelLoader):
    """Return one CPU fake model and record requested names."""

    def __init__(self, model: _RecordingModel) -> None:
        self.model = model
        self.names: list[str] = []

    def load(
        self,
        filename: str,
        frame_shape: tuple[int, ...] | None = None,
        scale_factor: float = 1.0,
    ) -> LoadedRifeModel:
        del frame_shape, scale_factor
        self.names.append(filename)
        return LoadedRifeModel(
            self.model,
            None,
            torch.device("cpu"),
            torch.float32,
            get_rife_model_spec("rife47.pth"),
        )


class _SkipFirstPair:
    """Skip interpolation only across the first source pair."""

    def is_frame_skipped(self, frame_index: int) -> bool:
        return frame_index == 0


def test_current_rife_alignment_tracks_subnative_scale_pyramids() -> None:
    """Current checkpoints receive enough padding at every quality scale."""

    assert [required_core_alignment(scale) for scale in (0.25, 0.5, 1.0, 2.0, 4.0)] == [
        256,
        128,
        64,
        64,
        64,
    ]
    assert [
        required_legacy_alignment(scale) for scale in (0.25, 0.5, 1.0, 2.0, 4.0)
    ] == [128, 64, 64, 64, 64]


def test_multiplier_forwards_scale_ensemble_and_skip_states() -> None:
    """Every synthesized frame retains WhiteRabbit controls and state behavior."""

    model = _RecordingModel()
    loader = _FakeLoader(model)
    engine = RifeInterpolationEngine(loader)
    frames = torch.tensor([0.0, 0.5, 1.0]).reshape(3, 1, 1, 1).expand(-1, 2, 2, 3)
    output = engine.interpolate_by_multiplier(
        frames,
        "rife47.pth",
        3,
        0.5,
        True,
        0,
        interpolation_states=_SkipFirstPair(),
    )
    torch.testing.assert_close(
        output[:, 0, 0, 0],
        torch.tensor([0.0, 0.5, 2 / 3, 5 / 6, 1.0]),
    )
    assert loader.names == ["rife47.pth"]
    assert model.calls == [(1 / 3, 0.5, True), (2 / 3, 0.5, True)]


def test_passthrough_multiplier_does_not_load_a_model() -> None:
    """A multiplier of one remains model-free."""

    model = _RecordingModel()
    loader = _FakeLoader(model)
    frames = torch.zeros((2, 2, 2, 3))
    output = RifeInterpolationEngine(loader).interpolate_by_multiplier(
        frames, "rife47.pth", 1, 1.0, True, 0
    )
    assert output is frames
    assert loader.names == []


class _RecordingInterpolation(RifeInterpolationEngine):
    """Record synth requests and return linear frames for FPS/seam tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, float, bool]] = []

    def synthesize(
        self,
        model_name: str,
        frame_0: torch.Tensor,
        frame_1: torch.Tensor,
        timestep: float,
        scale_factor: float,
        ensemble: bool,
    ) -> torch.Tensor:
        self.calls.append((model_name, timestep, scale_factor, ensemble))
        return torch.lerp(frame_0, frame_1, timestep)


def test_fps_resample_preserves_exact_count_scale_and_ensemble() -> None:
    """Non-integer FPS conversion makes the exact rational timeline."""

    interpolation = _RecordingInterpolation()
    frames = torch.tensor([0.0, 1.0, 2.0]).reshape(3, 1, 1, 1).expand(-1, 2, 2, 3)
    options = FpsResampleOptions(
        "rife_v4.26.safetensors",
        24.0,
        60.0,
        scale_factor=2.0,
        ensemble=False,
        clear_cache_interval=0,
    )
    output = RifeFpsResampler(interpolation).resample(frames, options)
    assert output.shape[0] == 6
    assert all(call[2:] == (2.0, False) for call in interpolation.calls)


def test_same_rate_downscale_and_zero_seam_are_model_free() -> None:
    """Fast paths avoid model downloads and neural execution."""

    interpolation = _RecordingInterpolation()
    frames = (
        torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1).expand(-1, 2, 2, 3)
    )
    resampler = RifeFpsResampler(interpolation)
    same = resampler.resample(
        frames,
        FpsResampleOptions("rife47.pth", 24, 24),
    )
    down = resampler.resample(
        frames,
        FpsResampleOptions("rife47.pth", 48, 24),
    )
    assert same is frames
    assert down[:, 0, 0, 0].tolist() == [0.0, 2.0]
    assert RifeSeamTimingAnalyzer(interpolation).analyze(
        "rife47.pth",
        1.0,
        True,
        frames,
        0,
        True,
        True,
        False,
        "MSE",
        4,
        0,
        0.96,
    ) == ("", 0)
    assert interpolation.calls == []
