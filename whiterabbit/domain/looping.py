# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Pure frame-index planning for loop manipulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSpan:
    """An inclusive-exclusive span within a frame batch."""

    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of frames in the span."""

        return self.end - self.start


def normalize_roll_offset(frame_count: int, offset: int) -> int:
    """Return the left-roll offset for a frame batch."""

    if frame_count <= 0:
        return 0
    return int(offset) % frame_count


def expanded_unroll_offset(
    expanded_frame_count: int,
    base_offset: int,
    in_betweens_per_gap: int,
) -> int:
    """Return the right-roll offset after interpolation expands each frame gap."""

    if expanded_frame_count <= 0:
        return 0
    return (
        int(base_offset) * (max(0, int(in_betweens_per_gap)) + 1)
    ) % expanded_frame_count


def build_trim_span(
    frame_count: int,
    trim_start_frames: int,
    trim_end_frames: int,
) -> FrameSpan:
    """Plan a trim while retaining at least one frame when the batch is non-empty."""

    if frame_count <= 0:
        return FrameSpan(0, 0)
    if frame_count == 1:
        return FrameSpan(0, 1)

    start = max(0, int(trim_start_frames))
    end_trim = max(0, int(trim_end_frames))
    if start + end_trim >= frame_count:
        start = min(start, frame_count - 1)
        end_trim = max(0, frame_count - start - 1)

    end = frame_count - end_trim if end_trim > 0 else frame_count
    if start >= end:
        return FrameSpan(frame_count - 1, frame_count)
    return FrameSpan(start, end)
