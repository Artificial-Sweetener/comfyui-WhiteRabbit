# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""WhiteRabbit ComfyUI extension entry point."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast


class _NodeRegistry(Protocol):
    """Typed boundary for the lazily imported node registry."""

    def get_nodes(self) -> list[type[object]]:
        """Return every node in stable workflow order."""


async def _get_node_list(_extension: object) -> list[type[object]]:
    """Load nodes only after Comfy has initialized its runtime APIs."""

    registry = cast(
        _NodeRegistry,
        import_module(".whiterabbit.nodes_v3", package=__package__),
    )
    return registry.get_nodes()


async def comfy_entrypoint() -> object:
    """Return WhiteRabbit's fully typed Comfy v3 extension."""

    comfy_api = import_module("comfy_api.latest")
    extension_type = type(
        "WhiteRabbitExtension",
        (comfy_api.ComfyExtension,),
        {
            "__doc__": "Advertise WhiteRabbit's v3 node collection.",
            "__module__": __name__,
            "get_node_list": _get_node_list,
        },
    )
    return extension_type()


__all__ = ["comfy_entrypoint"]
