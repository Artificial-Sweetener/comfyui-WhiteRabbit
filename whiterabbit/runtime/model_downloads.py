# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Safely download trusted model artifacts with Comfy progress reporting."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Protocol, cast

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
CHUNK_SIZE = 1024 * 1024


class ProgressReporter(Protocol):
    """Report artifact download progress."""

    def start(self, label: str, total: int | None) -> None:
        """Start one download."""

    def advance(self, current: int, total: int | None) -> None:
        """Report absolute downloaded bytes."""

    def finish(self) -> None:
        """Mark the download complete."""


class _ComfyProgressBar(Protocol):
    """Subset of Comfy's absolute progress API."""

    def update_absolute(self, value: int, total: int | None = None) -> None:
        """Set absolute progress."""


class ComfyProgressReporter:
    """Report downloads through ComfyUI's progress bar."""

    def __init__(self) -> None:
        """Create a reporter before a transfer begins."""

        self._progress: _ComfyProgressBar | None = None
        self._total = 1

    def start(self, label: str, total: int | None) -> None:
        """Create the Comfy progress bar."""

        self._total = total if total and total > 0 else 1
        comfy_utils = import_module("comfy.utils")
        self._progress = cast(_ComfyProgressBar, comfy_utils.ProgressBar(self._total))
        LOGGER.info("Downloading %s", label)

    def advance(self, current: int, total: int | None) -> None:
        """Update absolute byte progress."""

        if self._progress is not None:
            self._progress.update_absolute(current, total or self._total)

    def finish(self) -> None:
        """Mark the progress bar complete."""

        if self._progress is not None:
            self._progress.update_absolute(self._total, self._total)


@dataclass(frozen=True)
class DownloadRequest:
    """A trusted catalog download request."""

    source_url: str
    destination: Path
    expected_folder: Path
    expected_sha256: str
    description: str


class ModelDownloader:
    """Atomically download and verify model files inside one expected folder."""

    def download(
        self,
        request: DownloadRequest,
        progress: ProgressReporter | None = None,
    ) -> Path:
        """Return an existing verified file or download it through a `.part` file."""

        self._validate_destination(request.destination, request.expected_folder)
        if request.destination.is_file():
            if sha256_file(request.destination) == request.expected_sha256.lower():
                return request.destination
            raise ValueError(
                f"Existing model checksum mismatch: '{request.destination}'. "
                "Remove or replace the file before retrying."
            )
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=request.destination.parent,
            prefix=f".{request.destination.name}.",
            suffix=".part",
            delete=False,
        ) as staging_file:
            temporary = Path(staging_file.name)
        reporter = progress or ComfyProgressReporter()
        try:
            with urllib.request.urlopen(request.source_url, timeout=30) as response:
                total = _content_length(response)
                reporter.start(request.description, total)
                downloaded = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        downloaded += len(chunk)
                        reporter.advance(downloaded, total)
            actual = sha256_file(temporary)
            if actual != request.expected_sha256.lower():
                raise ValueError(
                    f"Downloaded model checksum mismatch for '{request.destination}'. "
                    f"Expected {request.expected_sha256}, got {actual}."
                )
            temporary.replace(request.destination)
            reporter.finish()
            return request.destination
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _validate_destination(destination: Path, expected_folder: Path) -> None:
        """Reject traversal outside the registered model directory."""

        try:
            destination.resolve().relative_to(expected_folder.resolve())
        except ValueError as error:
            raise ValueError(
                f"Download destination '{destination}' is outside '{expected_folder}'."
            ) from error


def sha256_file(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _content_length(response: BinaryIO) -> int | None:
    """Return an HTTP content length when supplied."""

    headers = getattr(response, "headers", None)
    value = headers.get("Content-Length") if headers is not None else None
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


__all__ = [
    "ComfyProgressReporter",
    "DownloadRequest",
    "ModelDownloader",
    "ProgressReporter",
    "sha256_file",
]
