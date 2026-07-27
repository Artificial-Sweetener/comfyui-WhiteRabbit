# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Resolve trusted RIFE checkpoints in Comfy's frame_interpolation model folder."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, cast

from ..domain.rife import RifeModelSpec, get_rife_model_spec
from .model_downloads import DownloadRequest, ModelDownloader


class RifeModelResolver:
    """Locate or safely download cataloged RIFE checkpoints."""

    def __init__(self, downloader: ModelDownloader | None = None) -> None:
        """Create the resolver with an injectable downloader."""

        self._downloader = downloader or ModelDownloader()

    def resolve(self, filename: str) -> tuple[Path, RifeModelSpec]:
        """Resolve one model within Comfy's registered native folder."""

        spec = get_rife_model_spec(filename)
        folder_paths: Any = import_module("folder_paths")
        existing = folder_paths.get_full_path("frame_interpolation", filename)
        if existing is not None:
            existing_path = Path(cast(str, existing))
            return (
                self._downloader.download(
                    DownloadRequest(
                        source_url=spec.source_url,
                        destination=existing_path,
                        expected_folder=existing_path.parent,
                        expected_sha256=spec.sha256,
                        description=f"RIFE {spec.version}",
                    )
                ),
                spec,
            )
        folders = cast(list[str], folder_paths.get_folder_paths("frame_interpolation"))
        if not folders:
            raise RuntimeError("ComfyUI has no registered frame_interpolation folder.")
        model_folder = Path(folders[0])
        destination = model_folder / spec.filename
        return (
            self._downloader.download(
                DownloadRequest(
                    source_url=spec.source_url,
                    destination=destination,
                    expected_folder=model_folder,
                    expected_sha256=spec.sha256,
                    description=f"RIFE {spec.version}",
                )
            ),
            spec,
        )


__all__ = ["RifeModelResolver"]
