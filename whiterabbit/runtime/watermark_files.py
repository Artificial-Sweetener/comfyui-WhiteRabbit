# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Comfy input-file integration for watermark images."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any, cast


class WatermarkFileResolver:
    """List and resolve watermark files through Comfy's annotated file paths."""

    def choices(self) -> list[str]:
        """Return sorted image files from Comfy's input directory."""

        folder_paths: Any = import_module("folder_paths")
        input_directory = cast(str, folder_paths.get_input_directory())
        files = [
            name
            for name in os.listdir(input_directory)
            if os.path.isfile(os.path.join(input_directory, name))
        ]
        filtered: Any = folder_paths.filter_files_content_types(files, ["image"])
        return sorted(cast(list[str], filtered))

    def resolve(self, annotated_path: str) -> Path:
        """Validate and resolve a workflow-supplied annotated input path."""

        if not annotated_path:
            raise ValueError("Select a watermark image from the list (or upload one).")
        folder_paths: Any = import_module("folder_paths")
        if not folder_paths.exists_annotated_filepath(annotated_path):
            raise ValueError(f"Invalid watermark file: {annotated_path}")
        return Path(cast(str, folder_paths.get_annotated_filepath(annotated_path)))


__all__ = ["WatermarkFileResolver"]
