# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Contract tests for all enhanced RIFE v3 nodes."""

from __future__ import annotations

from whiterabbit.domain.rife import RIFE_MODEL_NAMES
from whiterabbit.nodes_v3.rife import (
    RifeFpsResampleV3,
    RifeSeamTimingAnalyzerV3,
    RifeVfiAdvancedV3,
    RifeVfiOptV3,
)


def test_rife_node_ids_input_order_and_model_catalog() -> None:
    """RIFE schemas preserve workflow IDs and all established controls."""

    expected = {
        RifeVfiOptV3: (
            "RIFE_VFI_Opt",
            [
                "ckpt_name",
                "frames",
                "multiplier",
                "scale_factor",
                "ensemble",
                "clear_cache_after_n_frames",
                "optional_interpolation_states",
            ],
        ),
        RifeVfiAdvancedV3: (
            "RIFE_VFI_Advanced",
            [
                "ckpt_name",
                "frames",
                "multiplier",
                "t_mode",
                "t_gamma",
                "t_min",
                "t_max",
                "scale_factor",
                "ensemble",
                "clear_cache_after_n_frames",
                "custom_t_list_csv",
                "optional_interpolation_states",
            ],
        ),
        RifeSeamTimingAnalyzerV3: (
            "RIFE_SeamTimingAnalyzer",
            [
                "ckpt_name",
                "scale_factor",
                "ensemble",
                "full_clip",
                "multiplier",
                "use_first_two",
                "use_last_two",
                "use_global_median",
                "calibrate_metric",
                "calibrate_iters",
                "t_min",
                "t_max",
                "auto_tmax",
                "t_cap",
            ],
        ),
        RifeFpsResampleV3: (
            "RIFE_FPS_Resample",
            [
                "ckpt_name",
                "frames",
                "fps_in",
                "fps_out",
                "scale_factor",
                "ensemble",
                "linearize",
                "lf_guardrail",
                "lf_sigma",
                "source_pair_match",
                "match_a_cap",
                "match_b_cap",
                "edge_band_lock",
                "tau_low",
                "tau_high",
                "band_radius",
                "band_soft_sigma",
                "clear_cache_after_n_frames",
            ],
        ),
    }
    for node_class, (node_id, input_ids) in expected.items():
        schema = node_class.define_schema()
        assert schema.node_id == node_id
        assert [item.id for item in schema.inputs] == input_ids
        assert schema.inputs[0].options == RIFE_MODEL_NAMES
