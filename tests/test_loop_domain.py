# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for pure loop frame-index planning."""

from __future__ import annotations

from whiterabbit.domain.looping import (
    FrameSpan,
    build_trim_span,
    expanded_unroll_offset,
    normalize_roll_offset,
)


def test_normalize_roll_offset_handles_empty_negative_and_wrapped_offsets() -> None:
    """Roll offsets remain bounded to the available frame count."""

    assert normalize_roll_offset(0, 4) == 0
    assert normalize_roll_offset(4, 1) == 1
    assert normalize_roll_offset(4, -1) == 3
    assert normalize_roll_offset(4, 9) == 1


def test_expanded_unroll_offset_accounts_for_inserted_frames() -> None:
    """Inverse rotation expands the source offset by each interpolated gap."""

    assert expanded_unroll_offset(8, 1, 1) == 2
    assert expanded_unroll_offset(8, -1, 1) == 6
    assert expanded_unroll_offset(0, 1, 1) == 0


def test_build_trim_span_preserves_one_frame_when_over_trimmed() -> None:
    """Trim planning never removes every frame from a non-empty batch."""

    assert build_trim_span(5, 1, 2) == FrameSpan(1, 3)
    assert build_trim_span(5, 99, 99) == FrameSpan(4, 5)
    assert build_trim_span(1, 99, 99) == FrameSpan(0, 1)
    assert build_trim_span(0, 1, 1) == FrameSpan(0, 0)
