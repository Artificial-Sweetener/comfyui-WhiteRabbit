# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Tests for atomic verified model downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from whiterabbit.runtime.model_downloads import DownloadRequest, ModelDownloader


class _Progress:
    """Record download lifecycle calls."""

    def __init__(self) -> None:
        self.started = False
        self.finished = False

    def start(self, label: str, total: int | None) -> None:
        self.started = bool(label)

    def advance(self, current: int, total: int | None) -> None:
        assert current > 0

    def finish(self) -> None:
        self.finished = True


def test_download_is_atomic_verified_and_skips_verified_existing(
    tmp_path: Path,
) -> None:
    """Trusted artifacts move into place only after checksum verification."""

    source = tmp_path / "source.bin"
    source.write_bytes(b"verified model")
    models = tmp_path / "models"
    destination = models / "model.bin"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    request = DownloadRequest(
        source.as_uri(), destination, models, digest, "test model"
    )
    progress = _Progress()
    downloader = ModelDownloader()
    assert downloader.download(request, progress) == destination
    assert destination.read_bytes() == b"verified model"
    assert progress.started and progress.finished
    assert not list(models.glob("*.part"))
    assert downloader.download(request, progress) == destination


def test_download_rejects_traversal_and_checksum_mismatch(tmp_path: Path) -> None:
    """Untrusted destinations and corrupt payloads fail closed without leftovers."""

    source = tmp_path / "source.bin"
    source.write_bytes(b"wrong")
    models = tmp_path / "models"
    downloader = ModelDownloader()
    with pytest.raises(ValueError, match="outside"):
        downloader.download(
            DownloadRequest(
                source.as_uri(), tmp_path / "elsewhere.bin", models, "0" * 64, "bad"
            ),
            _Progress(),
        )
    destination = models / "model.bin"
    with pytest.raises(ValueError, match="checksum mismatch"):
        downloader.download(
            DownloadRequest(source.as_uri(), destination, models, "0" * 64, "bad"),
            _Progress(),
        )
    assert not destination.exists()
    assert not list(models.glob("*.part"))
