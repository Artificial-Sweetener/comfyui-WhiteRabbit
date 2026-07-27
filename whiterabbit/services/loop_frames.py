# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Application service for loop frame manipulation."""

from __future__ import annotations

import torch

from ..domain.looping import (
    build_trim_span,
    expanded_unroll_offset,
    normalize_roll_offset,
)
from ..shared.tensor_validation import validate_image_batch


class LoopFrameService:
    """Prepare, assemble, rotate, and trim ComfyUI frame batches."""

    def prepare(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the last/first seam pair and the unchanged source batch."""

        validate_image_batch(images)
        seam_pair = torch.cat((images[-1:], images[:1]), dim=0)
        return seam_pair, images

    def assemble(
        self,
        original_images: torch.Tensor,
        interpolated_frames: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Append only synthesized seam frames to the original frame batch."""

        original_shape = validate_image_batch(original_images, name="original_images")
        interpolated_shape = validate_image_batch(
            interpolated_frames,
            name="interpolated_frames",
        )
        if (
            original_shape.height,
            original_shape.width,
            original_shape.channels,
        ) != (
            interpolated_shape.height,
            interpolated_shape.width,
            interpolated_shape.channels,
        ):
            raise ValueError(
                "original_images and interpolated_frames must have matching spatial "
                "dimensions and channels."
            )
        originals = original_images.to(interpolated_frames.device)
        in_between = interpolated_frames[1:-1]
        return (torch.cat((originals, in_between), dim=0),)

    def roll(
        self,
        images: torch.Tensor,
        offset: int,
    ) -> tuple[torch.Tensor, int]:
        """Rotate a frame batch left by the normalized offset."""

        shape = validate_image_batch(images, allow_empty=True)
        normalized = normalize_roll_offset(shape.batch_size, offset)
        if normalized == 0:
            return images, int(offset)
        return torch.roll(images, shifts=-normalized, dims=0), int(offset)

    def unroll(
        self,
        images: torch.Tensor,
        base_offset: int,
        in_betweens_per_gap: int,
    ) -> tuple[torch.Tensor]:
        """Undo a roll after each original gap has gained synthesized frames."""

        shape = validate_image_batch(images, allow_empty=True)
        offset = expanded_unroll_offset(
            shape.batch_size,
            base_offset,
            in_betweens_per_gap,
        )
        if offset == 0:
            return (images,)
        return (torch.roll(images, shifts=offset, dims=0),)

    def trim(
        self,
        clip_frames: torch.Tensor,
        trim_start_frames: int,
        trim_end_frames: int,
    ) -> tuple[torch.Tensor]:
        """Trim frames from both ends while retaining one non-empty frame."""

        shape = validate_image_batch(clip_frames, name="clip_frames", allow_empty=True)
        span = build_trim_span(
            shape.batch_size,
            trim_start_frames,
            trim_end_frames,
        )
        return (clip_frames[span.start : span.end],)
