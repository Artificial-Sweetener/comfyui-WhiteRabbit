# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for automatic loop crop scoring and v3 registration."""

from __future__ import annotations

import pytest
import torch

from whiterabbit.domain.loop_autocrop import LoopAutocropOptions
from whiterabbit.nodes_v3.loop_autocrop import AutocropToLoopV3
from whiterabbit.runtime.loop_autocrop import (
    LoopAutocropRuntime,
    _translation_magnitude,
    parse_scales,
)


class DeterministicFlowRuntime(LoopAutocropRuntime):
    """Replace optional OpenCV flow with an exact test metric."""

    @staticmethod
    def _flow_magnitude(
        left: torch.Tensor,
        right: torch.Tensor,
        maximum_side: int = 256,
    ) -> float:
        del maximum_side
        return float((left - right).abs().mean().item())


def test_autocrop_runtime_matches_characterized_candidate_selection() -> None:
    """The extracted runtime retains crop choice, score, and diagnostic rows."""

    values = torch.tensor([0.0, 0.1, 0.2, 0.9, 0.0], dtype=torch.float32)
    frames = values.reshape(5, 1, 1, 1).expand(-1, 8, 8, 3).clone()
    options = LoopAutocropOptions(
        maximum_end_crop=3,
        include_first_step=True,
        include_last_step=True,
        include_global_median_step=False,
        seam_window_frames=1,
        distance_metric="L1",
        score_in_8bit=False,
        use_ssim_similarity=False,
        use_exposure_guard=False,
        use_flow_guard=False,
        weight_step_size=1.0,
        weight_similarity=0.0,
        weight_exposure=0.0,
        weight_flow=0.0,
        ssim_downsample_scales="1",
        accelerate_with_gpu=False,
        use_mixed_precision=False,
    )
    cropped, end_crop, length, score, diagnostics = LoopAutocropRuntime().find(
        frames, options
    )
    assert (end_crop, length, score) == (3, 2, 0.0)
    torch.testing.assert_close(cropped, frames[:2])
    assert diagnostics.splitlines()[-1] == (
        "3,0.000000,0.100000,0.100000,0.000000,0.000000,"
        "0.000000,0.000000,0.000000,0.000000"
    )


def test_ssim_scale_parser_is_unique_and_fault_tolerant() -> None:
    """Invalid and repeated scale entries do not destabilize scoring."""

    assert parse_scales("1, 2,2, nope, 0") == [1, 2]
    assert parse_scales("nope") == [1]


def test_flow_guard_scores_real_candidate_seams() -> None:
    """Flow diagnostics and weighting reflect each candidate's seam motion."""

    values = torch.tensor([0.0, 0.1, 0.5, 0.9], dtype=torch.float32)
    frames = values.reshape(4, 1, 1, 1).expand(-1, 4, 4, 3).clone()
    options = LoopAutocropOptions(
        maximum_end_crop=1,
        include_first_step=True,
        include_last_step=True,
        include_global_median_step=True,
        seam_window_frames=1,
        distance_metric="L1",
        score_in_8bit=False,
        use_ssim_similarity=False,
        use_exposure_guard=False,
        use_flow_guard=True,
        weight_step_size=0.0,
        weight_similarity=0.0,
        weight_exposure=0.0,
        weight_flow=1.0,
        ssim_downsample_scales="1",
        accelerate_with_gpu=False,
        use_mixed_precision=False,
    )

    _, end_crop, _, _, diagnostics = DeterministicFlowRuntime().find(frames, options)
    rows = [
        [float(value) for value in row.split(",")]
        for row in diagnostics.splitlines()[1:]
    ]

    assert end_crop == 1
    assert [row[8] for row in rows] == [0.9, 0.5]
    assert [row[9] for row in rows] == [0.4, 0.25]


def test_flow_fallback_detects_global_translation_without_opencv() -> None:
    """The torch fallback keeps flow scoring functional without optional OpenCV."""

    first = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    first[:, 4:10, 5:11] = 1
    shifted = torch.roll(first, shifts=(3, -2), dims=(1, 2))

    assert _translation_magnitude(first, shifted, 256) == pytest.approx(
        (3**2 + 2**2) ** 0.5
    )


def test_autocrop_v3_schema_preserves_public_contract_order() -> None:
    """The v3 node exposes every established scoring control in order."""

    schema = AutocropToLoopV3.define_schema()
    assert schema.node_id == "AutocropToLoop"
    assert [item.id for item in schema.inputs] == [
        "clip_frames",
        "max_end_crop_frames",
        "include_first_step",
        "include_last_step",
        "include_global_median_step",
        "seam_window_frames",
        "distance_metric",
        "score_in_8bit",
        "use_ssim_similarity",
        "use_exposure_guard",
        "use_flow_guard",
        "weight_step_size",
        "weight_similarity",
        "weight_exposure",
        "weight_flow",
        "ssim_downsample_scales",
        "accelerate_with_gpu",
        "use_mixed_precision",
    ]
