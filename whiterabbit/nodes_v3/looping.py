# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Comfy v3 nodes for frame-loop preparation and manipulation."""

from __future__ import annotations

from typing import Any

import torch

from ..services.loop_frames import LoopFrameService
from ._api import ComfyNodeBase, io

_SERVICE = LoopFrameService()


class PrepareLoopFramesV3(ComfyNodeBase):
    """Expose seam-pair preparation to ComfyUI."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable Prepare Loop Frames schema."""

        return io.Schema(
            node_id="PrepareLoopFrames",
            display_name="🐇 Prepare Loop Frames",
            category="video utils",
            description=(
                "Prepares the wrap seam: builds a tiny 2-frame batch [last, first] "
                "for your interpolator and also passes the original clip through "
                "unchanged."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "Your clip as an IMAGE batch (frames×H×W×C, values 0–1). "
                        "Outputs: [last, first] for the seam, plus the original clip."
                    ),
                )
            ],
            outputs=[
                io.Image.Output("interp_batch"),
                io.Image.Output("original_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the loop seam through the application service."""

        return _SERVICE.prepare(images)


class AssembleLoopFramesV3(ComfyNodeBase):
    """Expose seam-frame assembly to ComfyUI."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable Assemble Loop Frames schema."""

        return io.Schema(
            node_id="AssembleLoopFrames",
            display_name="🐇 Assemble Loop Frames",
            category="video utils",
            description=(
                "Builds the final loop: appends only the new in-between seam frames "
                "to your original clip—no duplicate of frame 1."
            ),
            inputs=[
                io.Image.Input(
                    "original_images",
                    tooltip="Your original clip (frames×H×W×C).",
                ),
                io.Image.Input(
                    "interpolated_frames",
                    tooltip=(
                        "Frames that bridge last→first. The first and last of this "
                        "batch are the originals; only the middle ones get added."
                    ),
                ),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(
        cls,
        original_images: torch.Tensor,
        interpolated_frames: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Assemble the final loop through the application service."""

        return _SERVICE.assemble(original_images, interpolated_frames)


class RollFramesV3(ComfyNodeBase):
    """Expose cyclic frame rotation to ComfyUI."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable Roll Frames schema."""

        return io.Schema(
            node_id="RollFrames",
            display_name="🐇 Roll Frames",
            category="video utils",
            description=(
                "Rolls the clip in a loop by an integer amount (cyclic shift). Also "
                "returns the same offset so you can undo it later."
            ),
            inputs=[
                io.Image.Input("images", tooltip="Your clip (frames×H×W×C)."),
                io.Int.Input(
                    "offset",
                    default=1,
                    min=-9999,
                    max=9999,
                    step=1,
                    tooltip=(
                        "How far to rotate the clip. Positive = forward in time; "
                        "negative = backward."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Int.Output("offset_out"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, offset: int) -> tuple[torch.Tensor, int]:
        """Roll the frame batch through the application service."""

        return _SERVICE.roll(images, offset)


class UnrollFramesV3(ComfyNodeBase):
    """Expose interpolation-aware inverse rotation to ComfyUI."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable Unroll Frames schema."""

        return io.Schema(
            node_id="UnrollFrames",
            display_name="🐇 Unroll Frames",
            category="video utils",
            description=(
                "Undo a previous roll after interpolation by accounting for the "
                "inserted frames (rotate by base_offset × (m+1))."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Clip after interpolation (frames′×H×W×C).",
                ),
                io.Int.Input(
                    "base_offset",
                    default=1,
                    min=-9999,
                    max=9999,
                    step=1,
                    tooltip="Use the exact offset_out that came from RollFrames.",
                ),
                io.Int.Input(
                    "m",
                    default=0,
                    min=0,
                    max=9999,
                    step=1,
                    tooltip=(
                        "How many in-betweens per gap were added (the interpolation "
                        "multiple)."
                    ),
                ),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        base_offset: int,
        m: int,
    ) -> tuple[torch.Tensor]:
        """Unroll the frame batch through the application service."""

        return _SERVICE.unroll(images, base_offset, m)


class TrimBatchEndsV3(ComfyNodeBase):
    """Expose bounded frame trimming to ComfyUI."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable Trim Batch Ends schema."""

        return io.Schema(
            node_id="TrimBatchEnds",
            display_name="🐇 Trim Batch Ends",
            category="video utils",
            description=(
                "Quickly remove frames from the start and/or end of a clip. Always "
                "keeps at least one frame."
            ),
            inputs=[
                io.Image.Input(
                    "clip_frames",
                    tooltip="Your clip (frames×H×W×C, 0–1).",
                ),
                io.Int.Input(
                    "trim_start_frames",
                    default=0,
                    min=0,
                    max=100000,
                    tooltip="Frames to remove from the START.",
                ),
                io.Int.Input(
                    "trim_end_frames",
                    default=0,
                    min=0,
                    max=100000,
                    tooltip="Frames to remove from the END.",
                ),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(
        cls,
        clip_frames: torch.Tensor,
        trim_start_frames: int,
        trim_end_frames: int,
    ) -> tuple[torch.Tensor]:
        """Trim the frame batch through the application service."""

        return _SERVICE.trim(
            clip_frames,
            trim_start_frames,
            trim_end_frames,
        )


LOOP_NODES = [
    PrepareLoopFramesV3,
    AssembleLoopFramesV3,
    RollFramesV3,
    UnrollFramesV3,
    TrimBatchEndsV3,
]

__all__ = [
    "AssembleLoopFramesV3",
    "LOOP_NODES",
    "PrepareLoopFramesV3",
    "RollFramesV3",
    "TrimBatchEndsV3",
    "UnrollFramesV3",
]
