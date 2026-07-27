# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Characterize WhiteRabbit's complete Comfy v3 workflow contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import ModuleType
from typing import Any, cast

from whiterabbit.nodes_v3 import get_nodes

EXPECTED_NODE_IDS = [
    "PrepareLoopFrames",
    "AssembleLoopFrames",
    "RollFrames",
    "UnrollFrames",
    "AutocropToLoop",
    "TrimBatchEnds",
    "RIFE_VFI_Opt",
    "RIFE_VFI_Advanced",
    "RIFE_SeamTimingAnalyzer",
    "RIFE_FPS_Resample",
    "PixelHold",
    "UpscaleWithModelAdvanced",
    "BatchResizeWithLanczos",
    "BatchWatermarkSingle",
]
EXPECTED_DISPLAY_NAMES = [
    "🐇 Prepare Loop Frames",
    "🐇 Assemble Loop Frames",
    "🐇 Roll Frames",
    "🐇 Unroll Frames",
    "🐇 Autocrop to Loop",
    "🐇 Trim Batch Ends",
    "🐇 RIFE VFI Interpolate by Multiple",
    "🐇 RIFE VFI Custom Timing",
    "🐇 RIFE Seam Timing Analyzer",
    "🐇 RIFE VFI FPS Resample",
    "🐇 Pixel Hold",
    "🐇 Upscale w/ Model (Advanced)",
    "🐇 Batch Resize w/ Lanczos",
    "🐇 Watermark",
]
EXPECTED_V3_CONTRACT_SHA256 = (
    "a05b9159ce00e8cf4cca331917f59d06af1ea99234c88621d175af990747b473"
)


def _v3_contract() -> list[dict[str, Any]]:
    """Return a deterministic representation of every v3 node schema."""

    nodes: list[dict[str, Any]] = []
    for node_class in cast(list[Any], get_nodes()):
        schema = node_class.define_schema()
        inputs: list[list[Any]] = []
        for input_item in schema.inputs:
            config = input_item.as_dict()
            if (
                schema.node_id == "BatchWatermarkSingle"
                and input_item.id == "watermark"
            ):
                config["options"] = ["<dynamic-input-files>"]
            inputs.append([input_item.id, input_item.io_type, config])
        nodes.append(
            {
                "node_id": schema.node_id,
                "display_name": schema.display_name,
                "category": schema.category,
                "description": schema.description,
                "inputs": inputs,
                "outputs": [[output.id, output.io_type] for output in schema.outputs],
            }
        )
    return nodes


def test_node_ids_display_names_and_order_are_stable() -> None:
    """All serialized workflow identifiers remain registered in historical order."""

    schemas = [
        node_class.define_schema() for node_class in cast(list[Any], get_nodes())
    ]
    assert [schema.node_id for schema in schemas] == EXPECTED_NODE_IDS
    assert [schema.display_name for schema in schemas] == EXPECTED_DISPLAY_NAMES


def test_complete_v3_schema_contract_is_stable() -> None:
    """Detect unapproved schema, tooltip, default, category, or output drift."""

    canonical = json.dumps(
        _v3_contract(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert hashlib.sha256(canonical.encode()).hexdigest() == EXPECTED_V3_CONTRACT_SHA256


def test_root_entrypoint_advertises_the_same_nodes(
    extension_package: ModuleType,
) -> None:
    """The public root uses only Comfy's v3 extension entrypoint."""

    extension = asyncio.run(extension_package.comfy_entrypoint())
    advertised = asyncio.run(extension.get_node_list())
    assert [
        node.define_schema().node_id for node in cast(list[Any], advertised)
    ] == EXPECTED_NODE_IDS
    assert extension_package.__all__ == ["comfy_entrypoint"]
