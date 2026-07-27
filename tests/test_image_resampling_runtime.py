# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Execution tests for typed image and mask resampling."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from whiterabbit.domain.image_geometry import ResizeMode, build_resize_plan
from whiterabbit.runtime.image_resampling import LanczosResampler, ProgressBar


class _Progress:
    """Record completed images for the test resampler."""

    def __init__(self) -> None:
        self.updates: list[int] = []

    def update(self, value: int) -> None:
        self.updates.append(value)


class _TestResampler(LanczosResampler):
    """Replace Comfy device and TorchLanc boundaries with deterministic CPU work."""

    progress = _Progress()

    @staticmethod
    def _device() -> torch.device:
        return torch.device("cpu")

    @staticmethod
    def _progress_bar(total: int) -> ProgressBar:
        _TestResampler.progress = _Progress()
        assert total > 0
        return _TestResampler.progress

    @staticmethod
    def _lanczos(
        images: torch.Tensor,
        *,
        height: int,
        width: int,
        sinc_window: int,
        precision: str,
    ) -> torch.Tensor:
        del sinc_window, precision
        return functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )


def test_resize_executes_shared_pad_geometry_for_images_and_masks() -> None:
    """Images and a broadcast mask receive identical fit-and-pad geometry."""

    images = torch.ones((2, 2, 4, 3), dtype=torch.float32)
    mask = torch.ones((2, 4), dtype=torch.float32)
    plan = build_resize_plan(
        4,
        2,
        4,
        4,
        ResizeMode.PAD.value,
        1,
        "center",
    )

    output, width, height, output_mask = _TestResampler().resize(
        images,
        plan,
        maximum_batch_size=1,
        sinc_window=3,
        pad_color="0, 0, 0",
        precision="fp32",
        mask=mask,
    )

    assert (width, height) == (4, 4)
    assert output.shape == (2, 4, 4, 3)
    assert output_mask.shape == (2, 4, 4)
    torch.testing.assert_close(output[:, 0], torch.zeros_like(output[:, 0]))
    torch.testing.assert_close(output[:, 1:3], torch.ones_like(output[:, 1:3]))
    torch.testing.assert_close(output_mask[:, 0], torch.zeros_like(output_mask[:, 0]))
    torch.testing.assert_close(
        output_mask[:, 1:3], torch.ones_like(output_mask[:, 1:3])
    )
    assert _TestResampler.progress.updates == [1, 1]
