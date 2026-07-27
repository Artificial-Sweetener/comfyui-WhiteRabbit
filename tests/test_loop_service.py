# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for loop-frame application behavior."""

from __future__ import annotations

import pytest
import torch

from whiterabbit.services.loop_frames import LoopFrameService


def _frames(count: int) -> torch.Tensor:
    """Return a small scalar-valued NHWC frame batch."""

    return torch.arange(count, dtype=torch.float32).reshape(count, 1, 1, 1)


def test_prepare_and_assemble_match_the_characterized_contract() -> None:
    """The typed service preserves seam extraction and assembly frame order."""

    service = LoopFrameService()
    source = _frames(4)
    seam, original = service.prepare(source)
    assert seam.flatten().tolist() == [3.0, 0.0]
    assert original is source

    interpolated = torch.tensor([3.0, 3.5, 3.75, 0.0]).reshape(4, 1, 1, 1)
    (assembled,) = service.assemble(source, interpolated)
    assert assembled.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 3.5, 3.75]


def test_roll_unroll_and_trim_match_the_characterized_contract() -> None:
    """The typed service retains existing rotation and bounded trimming behavior."""

    service = LoopFrameService()
    rolled, offset = service.roll(_frames(4), 1)
    assert rolled.flatten().tolist() == [1.0, 2.0, 3.0, 0.0]
    assert offset == 1
    (unrolled,) = service.unroll(_frames(8), offset, 1)
    assert unrolled.flatten().tolist() == [6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    (trimmed,) = service.trim(_frames(5), 1, 2)
    assert trimmed.flatten().tolist() == [1.0, 2.0]


def test_assemble_rejects_incompatible_frame_geometry() -> None:
    """Mismatched loop-seam geometry fails before concatenation."""

    with pytest.raises(ValueError, match="matching spatial dimensions"):
        LoopFrameService().assemble(
            torch.zeros((2, 8, 8, 3)),
            torch.zeros((2, 16, 8, 3)),
        )
