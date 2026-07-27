# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Contract tests for all enhanced RIFE v3 nodes."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

from whiterabbit.domain.rife import (
    RIFE_MODEL_NAMES,
    RIFE_SCALE_FACTOR_MAXIMUM,
    RIFE_SCALE_FACTOR_MINIMUM,
    RIFE_SCALE_FACTOR_STEP,
)
from whiterabbit.nodes_v3.rife import (
    RifeFpsResampleV3,
    RifeSeamTimingAnalyzerV3,
    RifeVfiAdvancedV3,
    RifeVfiOptV3,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_embedded_workflow(example_name: str) -> dict[str, object]:
    """Read the Comfy workflow metadata embedded in a documented PNG example."""

    data = (PROJECT_ROOT / "examples" / example_name).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"{example_name} is not a PNG file.")
    offset = 8
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + length
        chunk_type = data[offset + 4 : chunk_start]
        chunk_data = data[chunk_start:chunk_end]
        if chunk_type == b"tEXt":
            keyword, separator, value = chunk_data.partition(b"\0")
            if separator and keyword == b"workflow":
                parsed = json.loads(value.decode("utf-8"))
                if isinstance(parsed, dict):
                    return cast(dict[str, object], parsed)
                break
        offset = chunk_end + 4
    raise AssertionError(f"{example_name} does not embed a Comfy workflow.")


def _rife_scale_factor_from_workflow(
    example_name: str, node_type: str, widget_index: int
) -> int | float | str:
    """Return one persisted RIFE scale factor from a documented workflow."""

    workflow = _load_embedded_workflow(example_name)
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        raise AssertionError(f"{example_name} has no workflow nodes.")
    for node in nodes:
        if not isinstance(node, dict) or node.get("type") != node_type:
            continue
        values = node.get("widgets_values")
        if not isinstance(values, list):
            raise AssertionError(
                f"{example_name} has no widget values for {node_type}."
            )
        value: object = values[widget_index]
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise AssertionError(
                f"{example_name} has an invalid scale factor for {node_type}."
            )
        return value
    raise AssertionError(f"{example_name} does not contain {node_type}.")


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


def test_rife_scale_factor_is_a_bounded_float_for_every_rife_node() -> None:
    """RIFE scale control accepts legacy numbers and stringified workflow values."""

    for node_class in (
        RifeVfiOptV3,
        RifeVfiAdvancedV3,
        RifeSeamTimingAnalyzerV3,
        RifeFpsResampleV3,
    ):
        scale_input = next(
            item
            for item in node_class.define_schema().inputs
            if item.id == "scale_factor"
        )
        assert scale_input.get_io_type() == "FLOAT"
        assert scale_input.default == 1.0
        assert scale_input.min == RIFE_SCALE_FACTOR_MINIMUM
        assert scale_input.max == RIFE_SCALE_FACTOR_MAXIMUM
        assert scale_input.step == RIFE_SCALE_FACTOR_STEP


def test_documented_workflows_preserve_rife_scale_factor_values() -> None:
    """Changing the widget type keeps existing example workflow values usable."""

    examples = (
        ("interpolate_loop_seam.png", "RIFE_SeamTimingAnalyzer", 1, 1),
        ("interpolate_loop_seam.png", "RIFE_VFI_Advanced", 6, 1),
        ("resample_framerate.png", "RIFE_FPS_Resample", 3, 1),
    )
    for example_name, node_type, widget_index, expected_value in examples:
        assert (
            _rife_scale_factor_from_workflow(example_name, node_type, widget_index)
            == expected_value
        )
