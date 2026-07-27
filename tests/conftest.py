# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Pytest configuration for WhiteRabbit's authoritative Comfy environment."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_comfy_root() -> Path:
    """Locate the authoritative ComfyUI checkout used by the test environment."""

    configured_root = os.environ.get("WHITERABBIT_COMFY_ROOT")
    candidates = [
        Path(configured_root) if configured_root else None,
        PROJECT_ROOT.parents[1],
        PROJECT_ROOT.parents[1] / "ComfyUI",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "folder_paths.py").is_file():
            return candidate
    raise RuntimeError(
        "Unable to locate ComfyUI. Set WHITERABBIT_COMFY_ROOT to the ComfyUI root."
    )


COMFY_ROOT = _resolve_comfy_root()
for path in (PROJECT_ROOT, PROJECT_ROOT.parent, COMFY_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


@pytest.fixture(scope="session")
def extension_package() -> ModuleType:
    """Load the root extension package under a stable test-only module name."""

    module_name = "white_rabbit_extension_under_test"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing

    spec = importlib.util.spec_from_file_location(
        module_name,
        PROJECT_ROOT / "__init__.py",
        submodule_search_locations=[str(PROJECT_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create the WhiteRabbit package import spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
