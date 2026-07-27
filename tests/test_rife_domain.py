# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for RIFE catalog, scale, and timing domain behavior."""

from __future__ import annotations

import pytest

from whiterabbit.domain.rife import (
    RIFE_MODEL_NAMES,
    get_rife_model_spec,
    map_timing,
    parse_custom_timings,
    scale_pyramid,
)


def test_catalog_retains_legacy_models_and_adds_current_comfy_models() -> None:
    """Workflows keep 4.7/4.9 while native current models are selectable."""

    assert RIFE_MODEL_NAMES == [
        "rife47.pth",
        "rife49.pth",
        "rife_v4.25.safetensors",
        "rife_v4.26.safetensors",
    ]
    assert get_rife_model_spec("rife47.pth").architecture == "legacy47"
    assert get_rife_model_spec("rife_v4.26.safetensors").architecture == "core"


def test_scale_pyramids_cover_legacy_and_current_block_counts() -> None:
    """scale_factor continues to control every refinement stage."""

    assert scale_pyramid(1.0, 4) == [8.0, 4.0, 2.0, 1.0]
    assert scale_pyramid(0.5, 4) == [16.0, 8.0, 4.0, 2.0]
    assert scale_pyramid(1.0, 5) == [16.0, 8.0, 4.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="greater than zero"):
        scale_pyramid(0, 5)


def test_custom_and_gamma_timing_match_characterized_mapping() -> None:
    """Timing schedules retain clamping, sorting, and indexed custom values."""

    custom = parse_custom_timings("0.9, -1, 0.4, 2")
    assert custom == [0.0, 0.4, 0.9, 1.0]
    assert map_timing(0.5, "gamma_in", 2, 0, 1, [], 2) == 0.25
    assert map_timing(1 / 3, "custom_list", 1, 0, 1, custom, 2) == 0.0
