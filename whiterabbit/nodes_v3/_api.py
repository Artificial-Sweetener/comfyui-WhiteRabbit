# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Typed boundary for ComfyUI's dynamic v3 node API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    class ComfyNodeBase:
        """Static base used while type checking v3 node declarations."""

        @classmethod
        def define_schema(cls) -> Any:
            """Return the node schema supplied by each concrete node."""

            raise NotImplementedError

else:
    ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io

__all__ = ["ComfyNodeBase", "io"]
